# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from nneuroutil.helpers import FuzzyChoiceAction, module_logger
from nneuroutil.torch_extras import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)
from nneuroutil.visualization import figure, make_colorbar_axes

log = module_logger("lib.quadratic_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a quadratic neural network: f(z) = (W z)^2 + b.

Example:

    > {SCRIPT_PATH.name} --device cuda dataset.npz
"""


# {{{ model


class QuadraticNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        if output_size is None:
            output_size = input_size

        if dtype is None:
            dtype = torch.float32

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype

        from nneuroutil.torch_extras import (
            Bias,
            ComplexLinear,
            ComplexQuadratic,
            kaiming_uniform_,
        )

        # NOTE: we know that our function is (A x)^2 + c, so just encode that
        A = ComplexLinear(
            input_size,
            output_size,
            bias=False,
            interleaved=True,
            device=device,
            dtype=dtype,
        )
        kaiming_uniform_(A.weight, nonlinearity="cquadratic")

        self.model = nn.Sequential(
            A,
            ComplexQuadratic(interleaved=True),
            Bias(2 * output_size, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# }}}


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int = 5,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    epochs: int = 50,
    batch_size: int = 4096,
    early_stop_patience: int = 0,
    overwrite: bool = False,
) -> int:
    from commonlib import (
        determine_escape_radius,
        generate_timeseries_dataset,
        make_timeseries_datasets,
        torch_predict,
        torch_train,
    )

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    # {{{ vallidate inputs

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    try:
        data = np.load(filename)
    except Exception:
        log.error("Failed to load data from file: '%s'.", filename)
        return 1

    if datakey not in data:
        log.error("File does not contain '%s' dataset: '%s'.", datakey, filename)
        return 1

    matrices = data[datakey]
    if matrices.ndim != 3:
        log.error("Expected a 3 dimensional array: %s.", matrices.shape)
        return 1

    # NOTE: we don't really need complex128 for this example, so save some memory
    mat = torch.from_numpy(matrices[dataidx]).to(torch.complex64).to(device)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-quad.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{device.type}"
    )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ plot predict

    def plot_predict(epoch: int, m: nn.Module) -> None:
        # NOTE: plot 10 epochs and then only every 10th one to save time / memory
        if epoch > 10 and epoch % 10 != 0:
            return

        Zpredict = torch_predict(
            model,
            bbox=bbox,
            resolution=2 * resolution,
            maxit=maxit,
        )
        mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

        with figure(
            basename.with_stem(f"{basename.stem}-predict-{epoch:04d}"),
            overwrite=overwrite,
        ) as fig:
            ax = fig.gca()
            ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ train

    from torch import optim

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, device=device, dtype=mat.dtype.to_real())
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5.0e-3, weight_decay=1.0e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    log.info("Network:\n%s", model)

    weightsfile = outfile.with_suffix(".pkl")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]
        loss_test = checkpoint.get("loss_test", 0.0)
        zmask = checkpoint["zmask"]

        model.eval()
        plot_predict(epochs, model)
    else:
        # {{{ generate dataset

        x, y, _, zmask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            generate_timeseries_dataset(
                mat,
                bbox=bbox,
                resolution=resolution,
                maxit=maxit,
                escape_radius=escape_radius,
            ),
        )
        xy_train, xy_validate, xy_test = cast(
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
            make_timeseries_datasets(x, y, (0.8, 0.1, 0.1)),
        )

        mem.add_record("After data generation")

        # }}}

        # {{{ train

        _, y_train = xy_train
        target_scale = torch.mean(y_train.conj() * y_train).real.item()

        _, optimizer, losses = torch_train(
            model,
            criterion,
            optimizer,
            scheduler,
            TensorDataset(*xy_train),
            TensorDataset(*xy_validate),
            epochs=epochs,
            batch_size=batch_size,
            target_scale=target_scale,
            patience_early_stop=early_stop_patience,
            callback=plot_predict,
        )

        mem.add_record("After training")

        model.eval()
        with torch.no_grad():
            x_test, y_test = xy_test
            predictions = model(x_test)
            loss_test = criterion(predictions, y_test) / target_scale

            loss_test = loss_test.cpu().numpy()
            log.info("[%03d/%03d] test %.5e", epochs, epochs, loss_test)

        # }}}

        # {{{ checkpoint

        num_epochs_trained = len(losses)
        torch.save(
            {
                # model
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "num_epochs_trained": num_epochs_trained,
                "losses": losses,
                "loss_test": loss_test,
                "zmask": zmask,
                # parameters
                "inputfile": str(filename),
                "dataidx": dataidx,
                "bbox": bbox,
                "escape_radius": escape_radius,
                "resolution": resolution,
                "maxit": maxit,
                "early_stop_patience": early_stop_patience,
            },
            weightsfile,
        )

        # }}}

    # }}}

    # {{{ plot fractal

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(zmask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ plot loss

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        losses = losses.cpu().numpy()
        ax.semilogy(losses[:, 0, 0], label="Train")
        ax.semilogy(losses[:, 1, 0], label="Validation")
        ax.axhline(loss_test, color="k", ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    # }}}

    mem.add_record("Finished")

    # {{{ plot weight matrices

    # NOTE: the nn is trained to appproximate the trajectory, not the matrix A,
    # even though the structure is the same (`(W x)^2 + b`). We look at the
    # linear layer to get a sense of what it managed to do here. However,
    #   1. we do not expect the matrices to match
    #   2. we do expect them to have similar singular values (i.e. the nn captured
    #      the general dynamics of the problem)

    learned_A = model.model[0].weight.detach()  # ty: ignore[call-non-callable]
    A = mat.to(learned_A.dtype)

    with figure(
        basename.with_stem(f"{basename.stem}-weights"),
        nrows=2,
        ncols=2,
        overwrite=overwrite,
    ) as fig:
        ax11, ax12, ax21, ax22 = fig.axes

        # NOTE: some of the eig stuff does not work on GPU
        A = A.cpu().numpy().real
        learned_A = learned_A.cpu().numpy().real

        A_eigs = np.linalg.eigvals(A)
        learned_A_eigs = np.linalg.eigvals(learned_A)

        _, A_sigma, _ = np.linalg.svd(A)
        _, learned_A_sigma, _ = np.linalg.svd(learned_A)

        im = ax11.imshow(A.real, vmin=-1.0, vmax=1.0)
        ax11.set_title("$A$")
        fig.colorbar(im, cax=make_colorbar_axes(ax11))

        im = ax12.imshow(learned_A, vmin=-1.0, vmax=1.0)
        ax12.set_title("$A_{learned}$")
        fig.colorbar(im, cax=make_colorbar_axes(ax12))

        ax21.plot(A_eigs.real, A_eigs.imag, "o", label=r"$\lambda(A)$")
        ax21.plot(
            learned_A_eigs.real,
            learned_A_eigs.imag,
            "o",
            label=r"$\lambda(A_{learned})$",
        )
        ax21.set_xlabel(r"$\lambda_r$")
        ax21.set_ylabel(r"$\lambda_i$")
        ax21.set_title("EIG")
        ax21.legend()

        ax22.plot(A_sigma, "o", label=r"$\sigma(A)$")
        ax22.plot(learned_A_sigma, "o", label=r"$\sigma(A_{learned})$")
        ax22.set_ylabel(r"$\sigma$")
        ax22.set_title("SVD")
        ax22.legend()

    # }}}

    log.info("Memory:\n%s", mem)

    return 0


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "--dataidx",
        type=int,
        default=5,
        help="Index of the matrix to reconstruct",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Resolution (in the c complex plane) of the images",
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default deviced used by the framework",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training loss does not improve for this many epochs (0 disables)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        raise SystemExit(
            main(
                args.filename,
                dataidx=args.dataidx,
                resolution=args.resolution,
                early_stop_patience=args.early_stop_patience,
                outfile=args.outfile,
                overwrite=args.force,
            )
        )

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from commonlib import Config, make_argument_parser
from nneuroutil.helpers import machine_info, module_logger
from nneuroutil.torch_extras import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)
from nneuroutil.visualization import figure, make_colorbar_axes

log = module_logger("lib.quadratic_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains or runs inference with a quadratic neural network: f(z) = (W z)^2 + b.

Examples:

    > {SCRIPT_PATH.name} --device cpu train dataset.npz
    > {SCRIPT_PATH.name} --device cpu infer dataset-quad.pkl
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

    # ty: ignore[invalid-mutable-override]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# }}}


# {{{ helpers


def _plot_predict(
    filename: pathlib.Path,
    model: QuadraticNetwork,
    config: Config,
    *,
    tag: str,
    overwrite: bool,
) -> None:
    from commonlib import torch_predict

    Zpredict = torch_predict(
        model,
        bbox=config.bbox,
        resolution=2 * config.resolution,
        maxit=config.maxit,
    )

    escape_radius = config.escape_radius
    mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

    with figure(
        filename.with_stem(f"{filename.stem}-predict-{tag}"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask.cpu().numpy(), extent=config.bbox, cmap="binary")


def _plot_dataset(
    filename: pathlib.Path,
    zmask: torch.Tensor,
    *,
    bbox: tuple[float, float, float, float],
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(zmask.cpu().numpy(), extent=bbox, cmap="binary")


def _plot_loss(
    filename: pathlib.Path,
    losses: torch.Tensor,
    loss_test: float,
    *,
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        losses = losses.cpu().numpy()
        ax.semilogy(losses[:, 0, 0], label="Train")
        ax.semilogy(losses[:, 1, 0], label="Validation")
        ax.axhline(loss_test, color="k", ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()


def _plot_weights(
    filename: pathlib.Path,
    model: QuadraticNetwork,
    mat: torch.Tensor,
    *,
    overwrite: bool,
) -> None:
    # NOTE: the nn is trained to appproximate the trajectory, not the matrix A,
    # even though the structure is the same (`(W x)^2 + b`). We look at the
    # linear layer to get a sense of what it managed to do here. However,
    #   1. we do not expect the matrices to match
    #   2. we do expect them to have similar singular values (i.e. the nn captured
    #      the general dynamics of the problem)

    learned_A = model.model[0].weight.detach()  # ty: ignore[call-non-callable]
    A = mat.to(learned_A.dtype)

    with figure(
        filename.with_stem(f"{filename.stem}-weights"),
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


# {{{ train


def train(
    filename: pathlib.Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    weights: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from commonlib import (
        generate_timeseries_dataset,
        load_matrix_datasets,
        split_timeseries_datasets,
        torch_train,
    )

    machine = machine_info(SCRIPT_PATH)
    log.info("Machine:\n%s", machine.pretty())

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    provenance = {**machine._asdict(), "device": device}

    # {{{ load parameters

    if overrides is None:
        overrides = {}

    checkpoint: dict[str, Any] | None = None
    if weights is not None:
        if not weights.exists():
            log.error("Weights file does not exist: '%s'.", weights)
            return 1

        checkpoint = torch.load(weights, map_location=device, weights_only=True)
        config = Config.from_sources(checkpoint["config"], overrides)
        mat = checkpoint["mat"]
    else:
        config = Config.from_sources(overrides)

        try:
            matrices = cast(
                torch.Tensor,
                load_matrix_datasets(
                    filename, key=config.datakey, device=device, xp=torch
                ),
            )
            mat = matrices[config.dataidx].to(torch.complex64)
        except Exception as exc:
            log.error("Failed to load dataset '%s': %s", filename.name, exc)
            return 1

    mem.add_record("After data load")

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-quad.npz"

    outfile = outfile.with_stem(f"{outfile.stem}-{config.suffix}").with_suffix("")
    weightsfile = outfile.with_suffix(".pkl")

    if not overwrite and weightsfile.exists():
        log.error(
            "Weights file already exist (use --force to overwrite): '%s'", weightsfile
        )
        return 1

    # }}}

    # {{{ train

    from torch import optim

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, device=device, dtype=mat.dtype.to_real())
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1.0e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    log.info("Network:\n%s", model)

    losses_previous: torch.Tensor | None = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        losses_previous = checkpoint["losses"]
        log.info(
            "Resuming from '%s' (%d epochs trained)",
            weights,
            checkpoint["num_epochs_trained"],
        )

    # {{{ generate dataset

    x, y, _, zmask = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        generate_timeseries_dataset(
            mat,
            bbox=config.bbox,
            resolution=config.resolution,
            maxit=config.maxit,
            escape_radius=config.escape_radius,
        ),
    )
    xy_train, xy_validate, xy_test = cast(
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        split_timeseries_datasets(
            x, y, config.split, rng=np.random.default_rng(config.seed)
        ),
    )

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    def plot_predict(epoch: int, m: nn.Module) -> None:
        # NOTE: plot 10 epochs and then only every 10th one to save time / memory
        if epoch > 10 and epoch % 10 != 0:
            return

        _plot_predict(
            outfile,
            m,  # ty: ignore[invalid-argument-type]
            config,
            tag=f"{epoch:04d}",
            overwrite=overwrite,
        )

    _, y_train = xy_train
    target_scale = torch.mean(y_train.conj() * y_train).real.item()

    _, optimizer, losses = torch_train(
        model,
        criterion,
        optimizer,
        scheduler,
        TensorDataset(*xy_train),
        TensorDataset(*xy_validate),
        epochs=config.epochs,
        batch_size=config.batch_size,
        target_scale=target_scale,
        patience_early_stop=config.early_stop_patience,
        # callback=plot_predict,
    )

    mem.add_record("After training")

    model.eval()
    with torch.no_grad():
        x_test, y_test = xy_test
        predictions = model(x_test)
        loss_test = criterion(predictions, y_test) / target_scale

        loss_test = loss_test.cpu().item()
        log.info("[%03d/%03d] test %.5e", config.epochs, config.epochs, loss_test)

    mem.add_record("After test loss")

    # }}}

    # {{{ checkpoint

    from dataclasses import asdict

    if losses_previous is not None:
        losses = torch.cat((losses_previous.to(losses.device), losses))

    num_epochs_trained = len(losses)
    torch.save(
        {
            # model
            "provenance": provenance,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # loss
            "num_epochs_trained": num_epochs_trained,
            "losses": losses,
            "loss_test": loss_test,
            # dataset
            "inputfile": str(filename),
            "mat": mat,
            "zmask": zmask,
            # parameters
            "config": asdict(config),
        },
        weightsfile,
    )
    log.info("Saved checkpoint: '%s'.", weightsfile)

    mem.add_record("After checkpoint")

    # }}}

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, loss_test, overwrite=overwrite)
    _plot_weights(outfile, model, mat, overwrite=overwrite)

    # }}}

    mem.add_record("Finished")
    log.info("Memory:\n%s", mem)

    return 0


# }}}


# {{{ infer


def infer(
    weightsfile: pathlib.Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not weightsfile.exists():
        log.error("Weights file does not exist: '%s'.", weightsfile)
        return 1

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    # {{{ load parameters

    if overrides is None:
        overrides = {}

    checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)
    config = Config.from_sources(checkpoint["config"], overrides)

    losses = checkpoint["losses"]
    loss_test = checkpoint["loss_test"]
    zmask = checkpoint["zmask"]
    mat = cast(torch.Tensor, checkpoint["mat"])

    if outfile is None:
        outfile = weightsfile.with_suffix("")
    outfile = outfile.with_suffix("")

    # }}}

    # {{{ load model

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, device=device, dtype=mat.dtype.to_real())
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mem.add_record("After model load")

    # }}}

    # {{{ predict

    _plot_predict(
        outfile,
        model,
        config,
        tag="final",
        overwrite=overwrite,
    )

    mem.add_record("After prediction")

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, loss_test, overwrite=overwrite)
    _plot_weights(outfile, model, mat, overwrite=overwrite)

    # }}}

    mem.add_record("Finished")

    log.info("Memory:\n%s", mem)

    return 0


# }}}


if __name__ == "__main__":
    parser = make_argument_parser(
        description=SCRIPT_LONG_HELP,
        devices=available_device_names(),
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        if args.command == "train":
            raise SystemExit(
                train(
                    args.filename,
                    overrides=vars(args),
                    weights=args.weights,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        elif args.command == "infer":
            raise SystemExit(
                infer(
                    args.filename,
                    overrides=vars(args),
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise SystemExit(1)

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from nneuroutil.helpers import TicTocTimer, module_logger, tictoc

log = module_logger("lib.quadratic_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a quadratic neural network.

Example:

    > {SCRIPT_PATH.name} dataset.npz
"""


# {{{ model


class QuadraticNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
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

        from nneuroutil.torch_extras import Bias, ComplexLinear, ComplexQuadratic

        # NOTE: we know that our function is (A x)^2 + c, so just encode that
        self.model = nn.Sequential(
            ComplexLinear(input_size, output_size, bias=False, dtype=dtype),
            ComplexQuadratic(),
            Bias(2 * output_size, dtype=dtype),
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
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    overwrite: bool = False,
) -> int:
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
    mat = torch.from_numpy(matrices[dataidx]).to(torch.complex64)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        from commonlib import determine_escape_radius

        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = (
            filename.parent
            / f"{filename.stem}-quad-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}.npz"
        )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ generate dataset

    d = mat.shape[0]
    bbox = (-0.03, 0.03, -0.03, 0.03)
    log.info("Generating dataset on %s", bbox)

    with tictoc("generate_dataset"):
        from commonlib import generate_dataset

        Z, c = generate_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            escape_radius=escape_radius,
            maxit=maxit,
        )

    mask = torch.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, dim=-1) < escape_radius**2
    log.info("Points:")
    log.info("    Converged: %d", torch.sum(mask))
    log.info("    Max: %.5e (radius %.5e)", Z.abs().max(), escape_radius)
    log.info("    Mean: %.5e", Z.abs().mean())

    from nneuroutil.visualization import figure, make_colorbar_axes

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

    # {{{ build torch dataset

    from commonlib import iter_batches, split_dataset

    # Compute target scale for Normalized MSE (NMSE) loss to resolve vanishing gradients
    X, Y = split_dataset(Z, C=c, escape_radius=escape_radius, split_real_imaginary=True)
    target_scale = torch.mean(Y.conj() * Y).real
    log.info("Dataset: %s", X.shape)

    # }}}

    # {{{ train

    from torch import optim

    model = QuadraticNetwork(d, d, dtype=X.dtype)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5.0e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    log.info("Network:\n%s", model)

    weightsfile = outfile.with_suffix(".pkl")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]

        compiled_model = cast("QuadraticNetwork", torch.compile(model))
    else:
        timer = TicTocTimer()

        epochs = 50
        batch_size = 4096
        num_batches = X.shape[0] // batch_size
        losses = torch.empty(epochs)

        compiled_model = cast("QuadraticNetwork", torch.compile(model))

        for epoch in range(epochs):
            loss_epoch = torch.tensor(0.0)
            timer.tic()

            for batch_x, batch_y in iter_batches(X, Y, batch_size=batch_size):
                optimizer.zero_grad()

                # compute loss function
                predictions = compiled_model(batch_x)
                loss = criterion(predictions, batch_y) / target_scale

                # compute gradient
                loss.backward()

                # update weights
                optimizer.step()

                loss_epoch += loss.detach()

            losses[epoch] = loss_epoch.item() / num_batches
            scheduler.step(losses[epoch])

            timer.toc()
            log.info("[%03d/%03d] loss %.5e (%s)", epoch, epochs, loss_epoch, timer)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "losses": losses,
            },
            weightsfile,
        )

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        ax.semilogy(losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    # }}}

    # {{{ weight error

    # NOTE: the nn is trained to appproximate the trajectory, not the matrix A,
    # even though the structure is the same (`(W x)^2 + b`). We look at the
    # linear layer to get a sense of what it managed to do here. However,
    #   1. we do not expect the matrices to match
    learned_A = model.model[0].weight.detach()  # ty: ignore[call-non-callable]
    A = mat.to(learned_A.dtype)
    # E = torch.abs(learned_A - A)

    with figure(
        basename.with_stem(f"{basename.stem}-weights"),
        nrows=2,
        ncols=2,
        overwrite=overwrite,
    ) as fig:
        ax11, ax12, ax21, ax22 = fig.axes

        A_eigs = torch.linalg.eigvals(A)
        learned_A_eigs = torch.linalg.eigvals(learned_A)

        _, A_sigma, _ = torch.linalg.svd(A)
        _, learned_A_sigma, _ = torch.linalg.svd(learned_A)

        im = ax11.imshow(A.real, vmin=-1.0, vmax=1.0)
        ax11.set_title("$A$")
        fig.colorbar(im, cax=make_colorbar_axes(ax11))

        im = ax12.imshow(learned_A.real, vmin=-1.0, vmax=1.0)
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

    # {{{ forecast

    from commonlib import torch_predict

    Zpredict = torch_predict(
        compiled_model,
        bbox=bbox,
        resolution=2 * resolution,
        maxit=maxit,
    )
    mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

    with figure(
        basename.with_stem(f"{basename.stem}-predict"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

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
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.filename,
            overwrite=args.force,
        )
    )

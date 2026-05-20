# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from dmdaelib import Array3D, make_default_logger

log = make_default_logger(__file__)

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a quadratic neural network.

Example:

    > {SCRIPT_PATH.name}
"""


# {{{ model


class ComplexMSELoss(nn.Module):
    def forward(self, y_hat, y):
        dy = y_hat - y
        return (dy.conj() * dy).real.mean()


class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x**2


class BiasLayer(nn.Module):
    def __init__(self, size: int, *, dtype: Any = None) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class QuadraticNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        dtype: Any = None,
    ) -> None:
        if output_size is None:
            output_size = input_size

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype

        # NOTE: we know that our function is (A x)^2 + c, so just encode that
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False, dtype=dtype),
            QuadraticActivation(),
            BiasLayer(output_size, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# }}}

# {{{ predict


def predict(
    model: nn.Module,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
) -> Array3D:
    import array_api_compat

    from dmdaelib import generate_grid

    with torch.no_grad():
        z = torch.zeros((resolution, resolution, model.input_size), dtype=model.dtype)  # ty: ignore[no-matching-overload]
        xp = array_api_compat.array_namespace(z)

        c = generate_grid(*bbox, nx=resolution, ny=resolution, dtype=model.dtype, xp=xp)
        c = c[:, :, None]

        for _ in range(1, maxit):
            z = model(z) + c

    return z


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
    from dmdaelib import determine_escape_radius, generate_dataset, iter_batches, timeit

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

    with timeit("generate_dataset"):
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

    from dmdaelib import figure, make_colorbar_axes

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

    # {{{ build torch dataset

    # NOTE: we have an array of shape (maxit, n, n, d) and we want to
    #   * make our input/output be (Z_n, Z_{n + 1}) so that we can learn the dynamics
    #   * take only the converged points (based on mask from above)
    #   * squish everything down to use d as the "features"

    X = Z[:-1, mask, :].reshape(-1, d).contiguous()
    C = c.unsqueeze(0).unsqueeze(-1).expand(maxit - 1, -1, -1, d)
    Y = (Z[1:, mask, :] - C[:, mask, :]).reshape(-1, d).contiguous()
    log.info("Dataset size: %s", X.shape)

    # }}}

    # {{{ train

    from torch import optim

    model = QuadraticNetwork(d, d, dtype=Z.dtype)
    criterion = ComplexMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    weightsfile = outfile.with_suffix(".pth")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]

        compiled_model = cast("QuadraticNetwork", torch.compile(model))
    else:
        from dmdaelib import TicTocTimer

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
                loss = criterion(predictions, batch_y)

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

    # NOTE: the learned linear layer should approximate A (the input matrix),
    #   so compare the two to get a sense of training quality
    learned_A = model.model[0].weight.detach()  # ty: ignore[call-non-callable]
    A = mat.to(learned_A.dtype)
    E = torch.abs(learned_A - A) / torch.abs(A)

    with figure(
        basename.with_stem(f"{basename.stem}-weights"),
        nrows=2,
        ncols=2,
        overwrite=overwrite,
    ) as fig:
        ax11, ax12, ax21, ax22 = fig.axes

        A_eigs = torch.linalg.eigvals(A)
        learned_A_eigs = torch.linalg.eigvals(learned_A)

        im = ax11.imshow(A.real, vmin=-1.0, vmax=1.0)
        ax11.set_title("$A$")
        fig.colorbar(im, cax=make_colorbar_axes(ax11))

        im = ax12.imshow(learned_A.real, vmin=-1.0, vmax=1.0)
        ax12.set_title("$A_{learned}$")
        fig.colorbar(im, cax=make_colorbar_axes(ax12))

        im = ax21.imshow(torch.log10(E))
        ax21.set_title(r"$\log_{10}$(Error)")
        fig.colorbar(im, cax=make_colorbar_axes(ax21))

        ax22.plot(A_eigs.real, A_eigs.imag, "o", label="eigs($A$)")
        ax22.plot(
            learned_A_eigs.real,
            learned_A_eigs.imag,
            "o",
            label="eigs($A_{learned}$)",
        )
        ax22.set_xlabel(r"$\lambda_r$")
        ax22.set_ylabel(r"$\lambda_i$")
        ax22.set_title("Eigenvalues")

    # }}}

    # {{{ forecast

    Zpredict = predict(
        compiled_model, bbox=bbox, resolution=2 * resolution, maxit=maxit
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

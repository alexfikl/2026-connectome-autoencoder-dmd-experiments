# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import numpy as np
import torch

from nneuroutil.helpers import FuzzyChoiceAction, module_logger, tictoc
from nneuroutil.torch_extras import available_device_names, torch_default_device
from nneuroutil.visualization import figure, make_colorbar_axes

log = module_logger("lib.dmd")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Uses a classic DMD to learn the dynamics of a quadratic iteration

    z <- (A z)^2 + c

Example:

    > {SCRIPT_PATH.name} dataset.npz
"""


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int = 5,
    bbox: tuple[float, float, float, float] = (-0.04, 0.04, -0.04, 0.04),
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 192,
    overwrite: bool = False,
) -> int:
    from commonlib import determine_escape_radius, dmd_predict, generate_dataset

    device = torch.get_default_device()
    log.info("Device: %s", device)

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

    mat = torch.from_numpy(matrices[dataidx]).to(torch.complex64).to(device)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = (
            filename.parent
            / f"{filename.stem}-dmd-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}.npz"
        )

    basename = outfile.with_suffix("")
    log.info("DMD")

    # }}}

    # {{{ generate dataset

    d = mat.shape[0]
    with tictoc(f"Generated dataset on {bbox}"):
        Z, c = generate_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            escape_radius=escape_radius,
            maxit=maxit,
        )

    mask = torch.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, dim=-1) < escape_radius**2
    log.info("Points:")
    log.info("Converged: %d", torch.sum(mask))
    log.info("Max: %.5e (radius %.5e)", torch.max(torch.abs(Z)), escape_radius)
    log.info("Mean: %.5e", torch.mean(torch.abs(Z)))

    with figure(basename.with_stem(f"{basename.stem}-dataset")) as fig:
        ax = fig.gca()
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

    # {{{ build dataset

    # NOTE: To build the dataset we
    #   1. Add c as an extra variable (DMD can't learn affine systems, so we help it)
    #   2. Remove all entries that escaped (no point there)
    #   3. Reshape and get the `X1, X2` blocks for DMD

    C = torch.tile(c[None, :, :, None], (maxit, 1, 1, 1))
    X = torch.concatenate([Z, C], dim=3)
    X = X[:, mask, :]
    X = X.reshape(maxit, -1)

    X1 = X[:-1].reshape(-1, d + 1).T
    X2 = X[1:].reshape(-1, d + 1).T
    log.info("Dataset: %s", X1.shape)

    # }}}

    # {{{ train

    from nneuroutil.dmd import build_dmd, total_least_squares

    with tictoc("Constructed DMD model"):
        X1, X2 = total_least_squares(X1, X2, eps=1.0e-6)
        model = build_dmd(X1, X2, eps=1.0e-6)

    with figure(basename.with_stem(f"{basename.stem}-sigma")) as fig:
        ax = fig.gca()

        ax.semilogy(model.S.real)
        ax.set_xlabel("n")
        ax.set_ylabel(r"$\sigma$")

    # }}}

    # {{{ forecast

    # FIXME:
    #   - Why is the "prediction" a lot slower than "generate_dataset"?
    #   - What's a good tolerance for the escaped points based on the error?
    #   - bumping the resolution to 128x128 seems to fail (everybody escapes?)

    with tictoc("Predicting"):
        Zpredict = dmd_predict(model, Z, bbox=bbox)

    # NOTE: this is a linear operator, so we expect everybody to converge
    escaped = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=0) < escape_radius**2
    log.info("Escaped %d (points %d)", torch.sum(escaped), resolution * resolution)

    # NOTE: try to tell who escaped by looking at the error
    error = torch.linalg.norm(Z[-1].permute(2, 0, 1) - Zpredict, dim=0)

    with figure(basename.with_stem(f"{basename.stem}-error")) as fig:
        ax = fig.gca()

        im = ax.imshow(torch.log10(error), extent=bbox)
        fig.colorbar(im, cax=make_colorbar_axes(ax))

    with figure(basename.with_stem(f"{basename.stem}-predict")) as fig:
        ax = fig.gca()

        # NOTE: this threshold is matrix-dependent, so we'll need to play with it
        # to actually get the result back, but it can theoretically work
        ax.imshow(error < 6.0e-1, extent=bbox, cmap="binary")

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
        "--dataidx",
        type=int,
        default=5,
        help="Index of the matrix to reconstruct",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=64,
        help="Resolution (in the c complex plane) of the images",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default deviced used by the framework",
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

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        raise SystemExit(
            main(
                args.filename,
                dataidx=args.dataidx,
                resolution=args.resolution,
                overwrite=args.force,
            )
        )

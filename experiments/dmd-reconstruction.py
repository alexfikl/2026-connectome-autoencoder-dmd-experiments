# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Literal, cast

import numpy as np
import torch

from nneuroutil.helpers import FuzzyChoiceAction, module_logger, tictoc
from nneuroutil.torch_extras import (
    available_device_names,
    torch_default_device,
    view_as_complex,
)
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
    method: Literal["dmd", "edmd"] = "edmd",
    bbox: tuple[float, float, float, float] = (-0.04, 0.04, -0.04, 0.04),
    escape_radius: float | None = 100.0,
    resolution: int = 64,
    maxit: int = 192,
    overwrite: bool = False,
) -> int:
    from commonlib import determine_escape_radius

    device = torch.get_default_device()
    log.info("Device: %s", device)

    # {{{ validate inputs

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
        outfile = filename.parent / (
            f"{filename.stem}-{method}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}.npz"
        )

    basename = outfile.with_suffix("")
    log.info("DMD")

    # }}}

    # {{{ generate dataset

    d = mat.shape[0]
    with tictoc(f"Generated data on {bbox}"):
        from commonlib import generate_dataset

        Z, c = cast(
            tuple[torch.Tensor, torch.Tensor],
            generate_dataset(
                mat,
                bbox=bbox,
                resolution=resolution,
                escape_radius=escape_radius,
                maxit=maxit,
            ),
        )

    mask = torch.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, dim=-1) < escape_radius**2
    log.info("Points:")
    log.info("Converged: %d", torch.sum(mask))
    log.info(
        "Max: %.5e (radius %.5e)",
        torch.max(torch.abs(Z)),
        escape_radius,
    )
    log.info("Mean: %.5e", torch.mean(torch.abs(Z)))

    with figure(basename.with_stem(f"{basename.stem}-dataset")) as fig:
        ax = fig.gca()
        ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ build dataset

    # NOTE: To build the dataset we
    #   1. Add c as an extra variable (DMD can't learn affine systems, so we help it)
    #   2. Remove all entries that escaped (no point there)
    #   3. Reshape and get the `X1, X2` blocks for DMD

    with tictoc("Built dataset"):
        from commonlib import post_process_dataset

        # NOTE: Z has shape (ntrajectories, maxit, d)
        X, mask = cast(
            tuple[torch.Tensor, torch.Tensor],
            post_process_dataset(Z, escape_radius, min_escape_iters=None),
        )
        X = view_as_complex(X)

        C = torch.tile(c[mask][:, None, None], (1, maxit, 1))
        X = torch.concatenate([X, C], dim=-1)
        X = X.permute(1, 0, 2)

        X1 = X[:-1].reshape(-1, d + 1)
        X2 = X[1:].reshape(-1, d + 1)
        log.info("Dataset: %s", X1.shape)

    # }}}

    # {{{ train

    from nneuroutil.dmd import build_full_dmd, build_full_extended_dmd

    with tictoc("Constructed DMD model"):
        if method == "dmd":
            dmd = build_full_dmd(X1, X2, method="ridge")
        elif method == "edmd":
            dmd = build_full_extended_dmd(
                [
                    lambda x, A=mat: x[..., :d],
                    lambda x, A=mat: torch.einsum("...j,ij->...i", x[..., :d], A) ** 2,
                    lambda x, A=mat: torch.einsum("...j,ij->...i", x[..., :d], A) ** 4,
                    lambda x, A=mat: x[..., -1:],
                ],
                X1,
                X2,
                first_observable_is_state=True,
                method="ridge",
            )
        else:
            raise ValueError(f"unknown method: {method!r}")

    with figure(basename.with_stem(f"{basename.stem}-sigma")) as fig:
        ax = fig.gca()

        S = torch.linalg.svdvals(mat)
        # S = torch.cumsum(S, dim=0) / torch.sum(S)
        ax.semilogy(S.cpu().numpy(), label=r"$\sigma_A$")
        S = torch.linalg.svdvals(dmd.A)
        # S = torch.cumsum(S, dim=0) / torch.sum(S)
        ax.semilogy(S.cpu().numpy(), label=r"$\sigma_{{DMD}}$")
        S = torch.linalg.svdvals(X1)
        # S = torch.cumsum(S, dim=0) / torch.sum(S)
        ax.semilogy(S.cpu().numpy(), label=r"$\sigma_{{X}}$")
        ax.set_xlabel("n")
        ax.legend()

    with figure(basename.with_stem(f"{basename.stem}-lambda")) as fig:
        ax = fig.gca()

        theta = np.linspace(0, 2.0 * np.pi, 512)
        ax.plot(np.sin(theta), np.cos(theta), "k-", lw=1)

        S = torch.linalg.eigvals(mat.cpu()).numpy()
        ax.plot(S.real, S.imag, "o", label=r"$\lambda_A$")
        S = torch.linalg.eigvals(dmd.A.cpu()).numpy()  # ty: ignore[unresolved-attribute]
        ax.plot(S.real, S.imag, "o", label=r"$\lambda_{{DMD}}$")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_xlabel(r"$\lambda_i$")
        ax.set_xlim([-3, 3])
        ax.set_aspect("equal")
        ax.legend()

    # }}}

    # {{{ forecast

    with tictoc("Predicting"):
        Zpredict = torch.concat([Z[0], c[:, :, None]], dim=-1)
        Zpredict = dmd.encode(Zpredict)

        for _ in range(1, maxit):
            Zpredict = dmd.evolve(Zpredict)
            Zpredict[:, :, -1] = c

        Zpredict = cast("torch.Tensor", dmd.decode(Zpredict))

    # NOTE: this is a linear operator, so we expect everybody to converge
    escaped = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2
    total_escaped = torch.sum(escaped)
    log.info("Escaped %d (points %d)", total_escaped, resolution * resolution)

    from nneuroutil.vision import as_grayscale, otsu_threshold_from_image

    # NOTE: try to tell who escaped by looking at the error
    error = torch.linalg.norm(Z[-1] - Zpredict[:, :, :d], dim=-1)

    # binarize error in a nice way
    gray = as_grayscale(error, dtype=torch.uint8)
    thresh = otsu_threshold_from_image(gray)

    with figure(basename.with_stem(f"{basename.stem}-error")) as fig:
        ax = fig.gca()

        im = ax.imshow(torch.log10(error).cpu().numpy(), extent=bbox)
        fig.colorbar(im, cax=make_colorbar_axes(ax))

    with figure(basename.with_stem(f"{basename.stem}-predict")) as fig:
        ax = fig.gca()

        ax.imshow((gray < thresh).cpu().numpy(), extent=bbox, cmap="binary")  # ty: ignore[unsupported-operator]

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

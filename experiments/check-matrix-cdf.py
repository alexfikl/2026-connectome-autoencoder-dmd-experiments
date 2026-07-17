# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, TypeAlias

import numpy as np

from nneuroutil.helpers import module_logger

log = module_logger("lib.check_matrix_cdf")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Plot CDF of absolute edge weights for connectome matrices.

Example:

    > {SCRIPT_PATH.name} data/Emotion_LR_Task.npz
"""

DEFAULT_PERCENTILES = (0.80, 0.85, 0.90, 0.95, 0.99)

Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
ArrayND: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]]


def compute_cdf(matrices: ArrayND) -> tuple[Array1D, Array1D]:
    amat = np.abs(matrices.ravel())
    amat = amat[amat > 0.0]
    amat_sorted = np.sort(amat)
    cdf = np.arange(1, len(amat_sorted) + 1) / len(amat_sorted)
    return amat_sorted, cdf


def main(
    filename: pathlib.Path,
    *,
    datakey: str = "matrices",
    outfile: pathlib.Path | None = None,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    overwrite: bool = False,
) -> int:
    if not filename.exists():
        log.error("Input file does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_stem(f"{filename.stem}-cdf").with_suffix("")

    if not overwrite and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
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

    amat_sorted, cdf = compute_cdf(matrices)

    from commonlib import set_matplotlib_style  # ruff:ignore[unused-import]
    from nneuroutil.visualization import figure

    with figure(outfile) as fig:
        ax = fig.gca()
        ax.plot(amat_sorted, cdf, linewidth=2, color="C0")

        colors = ["C1", "C2", "C3", "C4", "C5"]
        for p, color in zip(percentiles, colors, strict=True):
            q = np.quantile(amat_sorted, p)
            ax.axvline(
                q, linestyle="--", alpha=0.7, color=color, label=f"{p:.0%}: {q:.5f}"
            )
        ax.legend(loc="lower right")

        ax.set_xlabel("weight")
        ax.set_ylabel("CDF")

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
        help="Output file name",
    )
    parser.add_argument(
        "-k",
        "--datakey",
        default="matrices",
        help="Name of the variable in the file to use",
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
            datakey=args.datakey,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

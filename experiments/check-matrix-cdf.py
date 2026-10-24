# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, TypeAlias

import numpy as np

log = logging.getLogger("lib.check_matrix_cdf")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Plot CDF of absolute edge weights for connectome matrices.

This also plots the per-matrix relative deviation from the pooled quantile
function, i.e. how each matrix's weight distribution differs from the overall
one as a function of the CDF level q.

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
        outfile = filename.with_suffix("")

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

    with figure(outfile.with_stem(f"{outfile.stem}-cdf"), overwrite=overwrite) as fig:
        ax = fig.gca()
        ax.plot(amat_sorted, cdf, linewidth=2, color="C0")

        colors = ["C1", "C2", "C3", "C4", "C5"]
        for p, color in zip(percentiles, colors, strict=True):
            q = np.quantile(amat_sorted, p)
            ax.axvline(
                float(q),
                linestyle="--",
                alpha=0.7,
                color=color,
                label=f"{p:.0%}: {q:.5f}",
            )
        ax.legend(loc="lower right")

        ax.set_xlabel("weight")
        ax.set_ylabel("CDF")

    with figure(
        outfile.with_stem(f"{outfile.stem}-cdf-ratio"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        q_grid = np.linspace(0.0, 1.0, 2001)[1:-1]
        w_ref = np.quantile(amat_sorted, q_grid)

        wmats = []
        for mat in matrices:
            wmat = np.abs(mat)
            wmat = np.sort(wmat[wmat > 0.0])
            wmats.append(np.quantile(wmat, q_grid) / w_ref - 1.0)

        wmat = np.asarray(wmats)
        median = np.median(wmat, axis=0)
        p5, p95 = np.percentile(wmat, [5, 95], axis=0)

        ax.axhline(0.0, linestyle="--", lw=1, color="k")

        ax.plot(q_grid, wmat.T * 100.0, lw=0.5, alpha=0.2, color="C0")
        ax.fill_between(
            q_grid,
            p5 * 100.0,
            p95 * 100.0,
            color="C0",
            alpha=0.2,
            label="5% - 95%",
        )
        ax.plot(q_grid, median * 100.0, color="C0", label="median")

        vmax = np.percentile(np.abs(wmat) * 100.0, 99.5)
        ax.set_ylim((-vmax, vmax))

        ax.legend(loc="lower right")
        ax.set_xlabel("CDF level $q$")
        ax.set_ylabel(r"relative deviation $w(q) / w_{\mathrm{ref}}(q) - 1$ [\%]")

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

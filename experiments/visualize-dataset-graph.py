# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

import commonlib  # ruff: ignore[unused-import]
from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array3D
from nneuroutil.visualization import figure

log = module_logger("lib.visualize_dataset_graph")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Example:

    > {SCRIPT_PATH.name} task-emotion.npz
"""

# {{{ visualize_strength


def visualize_strength(
    filename: pathlib.Path,
    matrices: Array3D[np.floating[Any]],
    *,
    overwrite: bool = False,
) -> None:
    from orbitkit.metrics import (
        compute_negative_weighted_degree,
        compute_positive_weighted_degree,
        compute_weighted_degree,
    )
    from orbitkit.utils import increment_mean_and_var

    _, n, m = matrices.shape
    if n != m:
        raise ValueError(f"matrices are not square: {matrices.shape}")

    s_mean = np.zeros(n, dtype=matrices.dtype)
    s_var = np.zeros(n, dtype=matrices.dtype)
    s_pos_mean = np.zeros(n, dtype=matrices.dtype)
    s_pos_var = np.zeros(n, dtype=matrices.dtype)
    s_neg_mean = np.zeros(n, dtype=matrices.dtype)
    s_neg_var = np.zeros(n, dtype=matrices.dtype)

    for i, mat in enumerate(matrices):
        s = compute_weighted_degree(mat)
        s_mean, s_var = increment_mean_and_var(i, s, s_mean, s_var)
        s_pos = compute_positive_weighted_degree(mat)
        s_pos_mean, s_pos_var = increment_mean_and_var(i, s_pos, s_pos_mean, s_pos_var)
        s_neg = compute_negative_weighted_degree(mat)
        s_neg_mean, s_neg_var = increment_mean_and_var(i, s_neg, s_neg_mean, s_neg_var)

    s_std = np.sqrt(s_var / (n - 1))
    s_pos_std = np.sqrt(s_pos_var / (n - 1))
    s_neg_std = np.sqrt(s_neg_var / (n - 1))

    ymin = (
        min(
            np.min(s_mean - s_std),
            np.min(s_pos_mean - s_pos_std),
            np.min(s_neg_mean - s_neg_std),
        )
        - 1
    )
    ymax = (
        max(
            np.max(s_mean + s_std),
            np.max(s_pos_mean + s_pos_std),
            np.max(s_neg_mean + s_neg_std),
        )
        + 1
    )

    with figure(
        filename.with_stem(f"{filename.stem}-strength"),
        ncols=3,
        figsize=(15, 5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2, ax3 = fig.axes
        ns = np.arange(n)

        (line,) = ax1.plot(ns, s_mean)
        ax1.fill_between(
            ns,
            s_mean - s_std,
            s_mean + s_std,
            color=line.get_color(),
            alpha=0.45,
        )
        ax1.set_xlabel("Index")
        ax1.set_title("Strength")
        ax1.set_ylim([ymin, ymax])

        ax2.plot([], [])
        (line,) = ax2.plot(ns, s_pos_mean)
        ax2.fill_between(
            ns,
            s_pos_mean - s_pos_std,
            s_pos_mean + s_pos_std,
            color=line.get_color(),
            alpha=0.45,
        )
        ax2.set_xlabel("Index")
        ax2.set_title("Strength (+)")
        ax1.set_ylim([ymin, ymax])

        ax3.plot([], [])
        ax3.plot([], [])
        (line,) = ax3.plot(ns, s_neg_mean)
        ax3.fill_between(
            ns,
            s_neg_mean - s_neg_std,
            s_neg_mean + s_neg_std,
            color=line.get_color(),
            alpha=0.45,
        )
        ax3.set_xlabel("Index")
        ax3.set_title("Strength (-)")
        ax1.set_ylim([ymin, ymax])


# }}}


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    show_strength: bool = True,
    overwrite: bool = False,
) -> int:
    if not filename.exists():
        raise FileNotFoundError(f"file does not exist: '{filename}'.")
        return 1

    try:
        data = np.load(filename)
    except Exception:
        log.error("failed to load data from file: '%s'.", filename)
        return 1

    if datakey not in data:
        log.error("file does not contain '%s' dataset: '%s'.", datakey, filename)
        return 1

    matrices = data[datakey]
    if matrices.ndim != 3:
        log.error("expected a 3d array in '%s' dataset: %s.", datakey, matrices.shape)
        return 1

    if outfile is None:
        outfile = filename.with_suffix("")

    if show_strength:
        visualize_strength(outfile, matrices, overwrite=overwrite)

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
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

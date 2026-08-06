# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

import commonlib  # ruff: ignore[unused-import]
from nneuroutil.helpers import module_logger
from nneuroutil.visualization import figure
from orbitkit.typing import Array1D, Array2D, Array3D

log = module_logger("lib.visualize_dataset_graph")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Example:

    > {SCRIPT_PATH.name} task-emotion.npz
"""

# {{{ run


def run(
    matrices: Array3D[np.floating[Any]],
    func: Callable[[Array2D[np.floating[Any]]], Array1D[np.floating[Any]]],
) -> tuple[Array1D[np.floating[Any]], Array1D[np.floating[Any]]]:
    from orbitkit.utils import increment_mean_and_var

    _, n, m = matrices.shape
    if n != m:
        raise ValueError(f"matrices are not square: {matrices.shape}")

    x_mean = np.zeros(n, dtype=matrices.dtype)
    x_var = np.zeros(n, dtype=matrices.dtype)

    for i, mat in enumerate(matrices):
        x = func(mat)
        x_mean, x_var = increment_mean_and_var(i, x, x_mean, x_var)

    x_std = np.sqrt(x_var / (n - 1))
    return x_mean, x_std


def plot_mean_std(
    ax: Any,
    x_mean: Array1D[np.floating[Any]],
    x_std: Array1D[np.floating[Any]],
    xlabel: str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    ns = np.arange(np.size(x_mean))

    (line,) = ax.plot(ns, x_mean)
    ax.fill_between(
        ns,
        x_mean - x_std,
        x_mean + x_std,
        color=line.get_color(),
        alpha=0.45,
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)


# }}}


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

    s_mean, s_std = run(matrices, compute_weighted_degree)
    s_pos_mean, s_pos_std = run(matrices, compute_positive_weighted_degree)
    s_neg_mean, s_neg_std = run(matrices, compute_negative_weighted_degree)

    with figure(
        filename.with_stem(f"{filename.stem}-strength"),
        ncols=3,
        figsize=(15, 5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2, ax3 = fig.axes

        plot_mean_std(
            ax1,
            s_mean,
            s_std,
            xlabel="Index",
            title="Strength",
            ylim=(-5.0, 30.0),
        )

        ax2.plot([], [])
        plot_mean_std(
            ax2,
            s_pos_mean,
            s_pos_std,
            xlabel="Index",
            title="Strength (+)",
            ylim=(-5.0, 30.0),
        )

        ax3.plot([], [])
        ax3.plot([], [])
        plot_mean_std(
            ax2,
            s_neg_mean,
            s_neg_std,
            xlabel="Index",
            title="Strength (-)",
            ylim=(-5.0, 30.0),
        )


# }}}


# {{{ visualize_clustering_coefficient


def visualize_clustering_coefficient(
    filename: pathlib.Path,
    matrices: Array3D[np.floating[Any]],
    *,
    overwrite: bool = False,
) -> None:
    from orbitkit.metrics import compute_weighted_clustering_coefficient_costantini

    cc6_mean, cc6_std = run(
        matrices,
        partial(compute_weighted_clustering_coefficient_costantini, variant=6),
    )
    cc7_mean, cc7_std = run(
        matrices,
        partial(compute_weighted_clustering_coefficient_costantini, variant=7),
    )
    cc8_mean, cc8_std = run(
        matrices,
        partial(compute_weighted_clustering_coefficient_costantini, variant=8),
    )

    with figure(
        filename.with_stem(f"{filename.stem}-clustering-coefficient"),
        ncols=3,
        figsize=(15, 5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2, ax3 = fig.axes

        plot_mean_std(
            ax1,
            cc6_mean,
            cc6_std,
            xlabel="Index",
            title="Variant 6",
            ylim=(0.0, 0.5),
        )

        plot_mean_std(
            ax2,
            cc7_mean,
            cc7_std,
            xlabel="Index",
            title="Variant 7",
            ylim=(0.0, 0.5),
        )

        plot_mean_std(
            ax3,
            cc8_mean,
            cc8_std,
            xlabel="Index",
            title="Variant 8",
            ylim=(0.0, 0.5),
        )


# }}}


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    show_strength: bool = True,
    show_clustering_coefficient: bool = True,
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

    if show_clustering_coefficient:
        visualize_clustering_coefficient(outfile, matrices, overwrite=overwrite)

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

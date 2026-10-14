# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, NamedTuple

import numpy as np

from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array3D, Array4D

log = module_logger("lib.check_dataset")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Analyzes the timeseries dataset used by the DMD autoencoder scripts.

Constructs the dataset in the same way as autoencoder-dmd-quadratic-network.py
and plots statistics about it: how many points escape, how fast they escape and
how many steps are spent in the pinned (clipped) state.

Example:

    > {SCRIPT_PATH.name} task-emotion.npz
"""


# {{{ statistics


class DatasetStatistics(NamedTuple):
    escape_time: Array3D[np.integer[Any]]
    """First iteration at which each trajectory escaped (``n_levels`` if never)."""
    escaped: Array3D[np.bool]
    """Boolean mask of pinned (clipped) states of shape ``(n_levels, nx, ny)``."""
    znorm: Array3D[np.floating[Any]]
    """Trajectory norms of shape ``(n_levels, nx, ny)``."""


def compute_statistics(
    z: Array4D[np.complexfloating[Any]],
    escape_radius: float,
) -> DatasetStatistics:
    """Compute escape statistics for the trajectories *z*.

    This mirrors the escape time computation in
    :func:`commonlib.post_process_dataset`.

    :arg z: trajectories of shape ``(n_levels, nx, ny, d)`` as returned by
        :func:`commonlib.generate_timeseries`.
    """
    sqr_norm = np.sum(np.real(z) ** 2 + np.imag(z) ** 2, axis=-1)
    escaped = sqr_norm > escape_radius**2

    never_escaped = np.all(~escaped, axis=0)
    first_escaped = np.argmax(escaped.astype(np.int32), axis=0)
    escape_time = np.where(never_escaped, z.shape[0], first_escaped)

    return DatasetStatistics(
        escape_time=escape_time,
        escaped=escaped,
        znorm=np.sqrt(sqr_norm),
    )


# }}}

# {{{ main


def main(
    filename: pathlib.Path,
    *,
    datakey: str = "matrices",
    dataidx: int = 5,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 10.0,
    resolution: int = 128,
    maxit: int = 32,
    min_escape_iters: int | None = None,
    basename: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from commonlib import generate_timeseries, post_process_dataset
    from nneuroutil.visualization import figure, make_colorbar_axes

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

    if not 0 <= dataidx < matrices.shape[0]:
        log.error("Index out of range: %d (max %d)", dataidx, matrices.shape[0] - 1)
        return 1

    mat = np.asarray(matrices[dataidx], dtype=np.complex64)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if min_escape_iters is None:
        # NOTE: same as autoencoder-dmd-quadratic-network.py
        min_escape_iters = 1 * (maxit + 1) // 4

    if basename is None:
        basename = filename.with_suffix("")

    basename = basename.with_stem(
        f"{basename.stem}-dataset-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}"
    )

    # }}}

    # {{{ construct dataset (same as autoencoder-dmd-quadratic-network.py)

    d = mat.shape[0]
    n_levels = maxit + 1
    max_z = (escape_radius + 1) / d**0.5

    z, _c = generate_timeseries(
        mat,
        bbox=bbox,
        escape_radius=escape_radius,
        resolution=resolution,
        # NOTE: we add 1 so that z.shape == (maxit + 1, nx, ny, d), matching the
        # trajectories in the training dataset
        maxit=maxit + 1,
    )

    # NOTE: this is what generate_timeseries_dataset uses to get the training mask
    z_kept, kept = post_process_dataset(
        z, escape_radius, interleaved=True, min_escape_iters=min_escape_iters
    )
    assert z_kept.shape[0] == np.sum(kept)

    stats = compute_statistics(z, escape_radius)
    escape_time = stats.escape_time

    converged = escape_time == n_levels
    escapers = np.logical_not(converged)

    # }}}

    # {{{ log statistics

    npoints = np.size(escape_time)
    nconverged = np.sum(converged)
    nescapers = np.sum(escapers)
    nkept = np.sum(kept)

    log.info("Dataset '%s' (index %d, d = %d)", filename.stem, dataidx, d)
    log.info("    Grid:        %d points on %s", npoints, bbox)
    log.info(
        "    Converged:   %d/%d (%.2f%%)",
        nconverged,
        npoints,
        100 * nconverged / npoints,
    )
    log.info(
        "    Escaped:     %d/%d (%.2f%%)", nescapers, npoints, 100 * nescapers / npoints
    )

    if nescapers:
        te = escape_time[escapers]
        p10, p90 = np.percentile(te, [10, 90])
        log.info(
            "    Escape time: mean %.1f median %.1f (p10 %.1f, p90 %.1f)",
            np.mean(te),
            np.median(te),
            p10,
            p90,
        )

    log.info(
        "    Kept:        %d/%d (%.2f%%) (t_e >= %d)",
        nkept,
        npoints,
        100 * nkept / npoints,
        min_escape_iters,
    )

    # transitions n -> n + 1: pinned if the state at level n is pinned, escaping
    # if the state at level n + 1 is the first pinned state
    n_transitions = maxit * nkept
    n_escape_transitions = np.sum(kept & escapers)
    n_pinned_transitions = np.sum((n_levels - 1) - escape_time[kept & escapers])
    n_active_transitions = n_transitions - n_escape_transitions - n_pinned_transitions

    log.info("    Transitions (kept trajectories): %d total", n_transitions)
    log.info(
        "        active-active: %d (%.2f%%)",
        n_active_transitions,
        100 * n_active_transitions / n_transitions,
    )
    log.info(
        "        escape:        %d (%.2f%%)",
        n_escape_transitions,
        100 * n_escape_transitions / n_transitions,
    )
    log.info(
        "        pinned-pinned: %d (%.2f%%)",
        n_pinned_transitions,
        100 * n_pinned_transitions / n_transitions,
    )
    log.info(
        "    Pinned state: norm %.3f (per-component max_z %.3e, R %.3f)",
        np.sqrt(d) * max_z,
        max_z,
        escape_radius,
    )

    if nconverged:
        zconv = stats.znorm[:, converged]
        log.info(
            "    Converged |z|: median %.3e, max %.3e",
            np.median(zconv),
            np.max(zconv),
        )

    # }}}

    # {{{ plot masks

    with figure(
        basename.with_stem(f"{basename.stem}-masks"),
        nrows=1,
        ncols=3,
        figsize=(15, 4.5),
        overwrite=overwrite,
    ) as fig:
        fig.set_layout_engine("tight")
        ax1, ax2, ax3 = fig.axes

        im = ax1.imshow(escape_time, extent=bbox, cmap="viridis")
        fig.colorbar(im, cax=make_colorbar_axes(ax1), label="Escape iteration $t_e$")
        ax1.set_title("Escape time")

        ax2.imshow(converged, extent=bbox, cmap="binary")
        ax2.set_title(rf"Converged ({100 * nconverged / npoints:.1f}\%)")

        ax3.imshow(kept, extent=bbox, cmap="binary")
        ax3.set_title(
            f"Kept $t_e \\geq {min_escape_iters}$ ({100 * nkept / npoints:.1f}\\%)"
        )

    # }}}

    # {{{ plot escape time histogram

    with figure(
        basename.with_stem(f"{basename.stem}-escape-time"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        bins = np.arange(1, n_levels + 2) - 0.5
        ax.hist(
            escape_time[escapers].ravel(),
            bins=bins,  # ty: ignore[invalid-argument-type]
            color="C0",
            label="Escaped",
        )
        ax.bar([n_levels + 1], [nconverged], color="C2", label="Never escaped")

        ax.axvline(
            min_escape_iters - 0.5,
            color="k",
            lw=1,
            ls="--",
            label=f"Kept $t_e \\geq {min_escape_iters}$",
        )

        xticks = list(range(8, n_levels, 8))
        ax.set_xticks([1, *xticks, n_levels + 2])
        ax.set_xticklabels([str(1), *(str(t) for t in xticks), "never"])
        ax.set_xlabel("Escape iteration $t_e$")
        ax.set_ylabel("Count")

        ax.legend()

    # }}}

    # {{{ plot pinned states

    levels = np.arange(n_levels)
    te_kept = escape_time[kept]
    pinned_frac = np.array([np.mean(te_kept <= n) for n in levels])

    with figure(
        basename.with_stem(f"{basename.stem}-pinned"),
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
        overwrite=overwrite,
    ) as fig:
        fig.set_layout_engine("tight")
        ax1, ax2 = fig.axes

        ax1.stackplot(
            levels,
            pinned_frac,
            1.0 - pinned_frac,
            labels=("Pinned", "Active"),
            colors=("C3", "C0"),
            alpha=0.8,
        )
        ax1.set_xlabel("Iteration $n$")
        ax1.set_ylabel("Fraction of kept trajectories")
        ax1.set_xlim((0, n_levels - 1))
        ax1.set_ylim((0, 1))
        ax1.legend(loc="upper left")
        ax1.set_title("Trajectory composition")

        fracs = (
            100
            * np.array([
                n_active_transitions,
                n_escape_transitions,
                n_pinned_transitions,
            ])
            / n_transitions
        )
        bars = ax2.bar(
            ("active-active", "escape", "pinned-pinned"),
            fracs,
            color=("C0", "C2", "C3"),
        )
        ax2.bar_label(bars, fmt="%.1f%%")
        ax2.set_ylabel("% of all kept transitions")
        ax2.set_ylim((0, 100))
        ax2.set_title("Transition types")

    # }}}

    # {{{ plot norms

    with figure(
        basename.with_stem(f"{basename.stem}-norms"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        for mask, color, label in [
            (converged, "C0", "Converged"),
            (kept & escapers, "C3", "Escaped"),
        ]:
            zgroup = stats.znorm[:, mask]
            if not zgroup.size:
                continue

            median = np.median(zgroup, axis=1)
            q25, q75 = np.percentile(zgroup, [25, 75], axis=1)

            (line,) = ax.semilogy(levels, median, color=color, label=label)
            ax.fill_between(levels, q25, q75, color=line.get_color(), alpha=0.3)

        ax.axhline(escape_radius, color="k", lw=1, ls="--")
        ax.text(
            0,
            escape_radius,
            f"escape radius $R = {escape_radius:g}$",
            va="bottom",
        )
        ax.axhline(np.sqrt(d) * max_z, color="k", lw=1, ls=":")
        ax.text(
            0,
            np.sqrt(d) * max_z,
            f"pinned state ($\\sqrt{{d}}\\, m = {np.sqrt(d) * max_z:.3f}$)",
            va="bottom",
        )

        ax.set_xlabel("Iteration $n$")
        ax.set_ylabel(r"$\|z_n\|_2$")
        ax.set_xlim((0, n_levels - 1))
        ax.legend(loc="lower right")

    # }}}

    return 0


# }}}

if __name__ == "__main__":
    import argparse
    from typing import cast

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
        "--datakey",
        type=str,
        default="matrices",
        help="Key in the npz file containing the matrices",
    )
    parser.add_argument(
        "--dataidx",
        type=int,
        default=5,
        help="Index of the matrix to analyze",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Resolution (in the c complex plane) of the dataset",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-0.03, 0.03, -0.03, 0.03),
        metavar=("RE_MIN", "RE_MAX", "IM_MIN", "IM_MAX"),
        help="Bounding box in the c complex plane",
    )
    parser.add_argument(
        "--escape-radius",
        type=float,
        default=10.0,
        help="Escape radius for the iteration",
    )
    parser.add_argument(
        "--maxit",
        type=int,
        default=32,
        help="Number of iterations per trajectory",
    )
    parser.add_argument(
        "--min-escape-iters",
        type=int,
        default=None,
        help="Minimum iterations before a trajectory is allowed to escape "
        "(default: (maxit + 1) // 4)",
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
            dataidx=args.dataidx,
            bbox=cast("tuple[float, float, float, float]", tuple(args.bbox)),
            escape_radius=args.escape_radius,
            resolution=args.resolution,
            maxit=args.maxit,
            min_escape_iters=args.min_escape_iters,
            basename=args.outfile,
            overwrite=args.force,
        )
    )

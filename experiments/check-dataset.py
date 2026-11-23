# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, NamedTuple

import numpy as np

from nneuroutil.typing import Array1D, Array2D, Array3D, Array4D

log = logging.getLogger("lib.check_dataset")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Analyzes the timeseries dataset used by the DMD autoencoder scripts.

Constructs the dataset in the same way as autoencoder-dmd-quadratic-network.py
and plots statistics about it: how many points escape, how fast they escape and
how many steps are spent in the pinned (clipped) state. Pass ``--cache FILE``
to load the data for the plots from a previously saved file instead of
recomputing it.

Example:

    > {SCRIPT_PATH.name} task-emotion.npz
    > {SCRIPT_PATH.name} --cache task-emotion-cache.npz task-emotion.npz
"""


# {{{ statistics


class DatasetStatistics(NamedTuple):
    escape_time: Array2D[np.integer[Any]]
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
    from commonlib import compute_escaped_mask

    escaped = compute_escaped_mask(z, escape_radius)

    never_escaped = np.all(np.logical_not(escaped), axis=0)
    first_escaped = np.argmax(np.asarray(escaped, dtype=np.int32), axis=0)
    escape_time = np.where(never_escaped, z.shape[0], first_escaped)

    return DatasetStatistics(
        escape_time=escape_time,
        escaped=escaped,
        znorm=np.linalg.norm(z, axis=-1),
    )


def compute_cumulative_energy(
    z: Array4D[np.complexfloating[Any]],
    escaped: Array3D[np.bool],
) -> Array1D[np.floating[Any]]:
    """Compute cumulative spectral energy of the unpinned trajectory snapshot matrix.

    :arg z: trajectories of shape ``(n_levels, nx, ny, d)``.
    :arg escaped: boolean mask of shape ``(n_levels, nx, ny)`` denoting pinned states.
    """
    d = z.shape[-1]
    snapshots = z[np.logical_not(escaped)].reshape(-1, d)

    s = np.linalg.svd(snapshots, compute_uv=False)
    s_sq = s**2
    return np.cumsum(s_sq) / (np.sum(s_sq) + 1e-15)


# }}}

# {{{ cache loading


def load_cache(cache: pathlib.Path, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """Load the given *keys* from a cache file."""
    if not cache.exists():
        log.error("Cache file does not exist: '%s'.", cache)
        return None

    with np.load(cache) as data:
        missing = [key for key in keys if key not in data]
        if missing:
            log.error("Cache file '%s' does not contain %s.", cache, missing)
            return None

        return {key: data[key] for key in keys}


# }}}

# {{{ compute and plot dataset statistics


def compute_dataset_statistics(
    mat: Array2D[np.complexfloating[Any]],
    *,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 10.0,
    resolution: int = 128,
    maxit: int = 32,
    min_escape_iters: int,
) -> dict[str, Any]:
    """Compute the data needed by the single-connectome plots."""
    from commonlib import generate_timeseries

    z, _ = generate_timeseries(
        mat,
        bbox=bbox,
        escape_radius=escape_radius,
        resolution=resolution,
        # NOTE: we add 1 so that z.shape == (maxit + 1, nx, ny, d), matching the
        # trajectories in the training dataset
        maxit=maxit + 1,
    )

    stats = compute_statistics(z, escape_radius)
    escape_time = stats.escape_time

    return {
        "escape_time": escape_time,
        "kept": np.asarray(escape_time) >= min_escape_iters,
        "znorm": stats.znorm,
        "energy": compute_cumulative_energy(z, stats.escaped),
    }


def plot_dataset_statistics(
    basename: pathlib.Path,
    *,
    escape_time: Array2D[np.integer[Any]],
    kept: Array2D[np.bool],
    znorm: Array3D[np.floating[Any]],
    energy: Array1D[np.floating[Any]],
    d: int,
    bbox: tuple[float, float, float, float],
    escape_radius: float,
    maxit: int,
    min_escape_iters: int,
    dataidx: int,
    filename: pathlib.Path,
    overwrite: bool = False,
) -> None:
    """Log the dataset statistics and plot them."""
    from nneuroutil.visualization import figure, make_colorbar_axes

    n_levels = maxit + 1
    max_z = (escape_radius + 1) / d**0.5
    converged = escape_time == n_levels
    escapers = np.logical_not(converged)

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
        zconv = znorm[:, converged]
        log.info(
            "    Converged |z|: median %.3e, max %.3e",
            np.median(zconv),
            np.max(zconv),
        )

    r9999 = np.searchsorted(energy, 0.9999) + 1
    log.info("    Cumulative energy: r_99.9%% = %d of %d", r9999, d)

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
            np.ravel(np.asarray(escape_time)[escapers]),
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
            zgroup = znorm[:, mask]
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

    # {{{ plot cumulative energy

    with figure(
        basename.with_stem(f"{basename.stem}-energy"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        k = np.arange(1, len(energy) + 1)

        ax.plot(k, energy, "o-", ms=3)
        ax.axhline(0.999, ls="--", color="k", alpha=0.7, label=f"$k = {r9999}$")
        ax.axvline(r9999, ls=":", color="k", alpha=0.7)
        ax.plot([r9999], [0.999], ms=4)
        ax.set_xlabel("Rank truncation $k$")
        ax.set_ylabel("Cumulative spectral energy")
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="lower right")

    # }}}


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
    cache: pathlib.Path | None = None,
    basename: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from commonlib import set_matplotlib_style

    set_matplotlib_style()

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

    if basename is None:
        basename = filename.with_suffix("")

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

    outfile = basename.with_stem(
        f"{basename.stem}-dataset-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}"
    )

    # }}}

    # {{{ construct or load dataset statistics

    if cache is not None:
        keys = (
            "escape_time",
            "kept",
            "znorm",
            "energy",
            "d",
            "bbox",
            "escape_radius",
            "maxit",
            "min_escape_iters",
        )
        data = load_cache(cache, keys)
        if data is None:
            return 1

        d = int(data.pop("d"))
        bbox_data = data.pop("bbox")
        bbox = (
            float(bbox_data[0]),
            float(bbox_data[1]),
            float(bbox_data[2]),
            float(bbox_data[3]),
        )
        escape_radius = float(data.pop("escape_radius"))
        maxit = int(data.pop("maxit"))
        min_escape_iters = int(data.pop("min_escape_iters"))
        log.info("Loading dataset statistics from '%s'.", cache)
    else:
        d = mat.shape[0]
        data = compute_dataset_statistics(
            mat,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
            min_escape_iters=min_escape_iters,
        )
        np.savez(
            outfile.with_suffix(".npz"),
            **data,
            d=d,
            bbox=np.asarray(bbox),
            escape_radius=escape_radius,
            maxit=maxit,
            min_escape_iters=min_escape_iters,
        )

    # }}}

    plot_dataset_statistics(
        outfile,
        **data,
        d=d,
        bbox=bbox,
        escape_radius=escape_radius,
        maxit=maxit,
        min_escape_iters=min_escape_iters,
        dataidx=dataidx,
        filename=filename,
        overwrite=overwrite,
    )

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
        "--cache",
        type=pathlib.Path,
        default=None,
        metavar="FILE",
        help="Load the data for the plots from FILE instead of computing it",
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
        default=100.0,
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
            cache=args.cache,
            basename=args.outfile,
            overwrite=args.force,
        )
    )

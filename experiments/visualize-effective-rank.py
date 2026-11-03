# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.typing import Array1D, Array2D, Array3D, Array4D

log = logging.getLogger("lib.visualize_effective_rank")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Computes the effective rank of every matrix in the dataset used by the DMD
autoencoder scripts.

Constructs the dataset in the same way as autoencoder-dmd-quadratic-network.py
and computes the effective rank (at a few cumulative spectral energy
thresholds) for every matrix in the file, then plots it. Pass ``--cache FILE``
to load the data for the plots from a previously saved file instead of
recomputing it.

Example:

    > {SCRIPT_PATH.name} task-emotion.npz
    > {SCRIPT_PATH.name} --cache task-emotion-cache.npz task-emotion.npz
"""


# {{{ compute effective ranks for a whole dataset


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


def compute_effective_rank(
    z: Array4D[np.complexfloating[Any]],
    escaped: Array3D[np.bool],
    *,
    thresholds: tuple[float, ...] = (0.999, 0.9999, 0.99999),
) -> Array1D[np.integer[Any]]:
    """Compute the effective rank of the real interleaved snapshot matrix.

    :arg z: trajectories of shape ``(n_levels, nx, ny, d)``.
    :arg escaped: boolean mask of shape ``(n_levels, nx, ny)`` denoting pinned
        states.
    :arg thresholds: cumulative spectral energy thresholds in ``(0, 1)``.
    """

    # NOTE: same real interleaved layout as nn-autoencoder-dmd.py
    active = z[np.logical_not(escaped)]
    snapshots = np.stack([active.real, active.imag], axis=-1)
    snapshots = snapshots.reshape(snapshots.shape[0], -1)

    s = np.linalg.svd(snapshots, compute_uv=False)
    s_sq = s**2
    energy = np.cumsum(s_sq) / (np.sum(s_sq) + 1e-15)

    return np.minimum(np.searchsorted(energy, thresholds) + 1, energy.size)


def compute_ranks(
    matrices: Array3D[np.number[Any]],
    *,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 10.0,
    resolution: int = 128,
    maxit: int = 32,
    thresholds: tuple[float, ...] = (0.999, 0.9999, 0.99999),
) -> Array2D[np.integer[Any]]:
    """Compute the effective rank for every matrix in *matrices*."""
    from commonlib import compute_escaped_mask, generate_timeseries

    # NOTE: the dataset generation is the slow part, so we try to do it on the GPU
    xp = array_api_compat.array_namespace(np.empty(0))
    device = "cpu"
    try:
        import torch

        from nneuroutil.interop.torch import available_device_names

        if any(dev.startswith("cuda") for dev in available_device_names()):
            xp = array_api_compat.array_namespace(torch.empty(0))
            device = "cuda"
    except ImportError:
        pass

    log.info("Generating dataset using '%s' (%s).", xp.__name__, device)

    labels = [f"{100.0 * t:g}%" for t in thresholds]
    ranks = np.empty((matrices.shape[0], len(thresholds)), dtype=np.int64)

    for i, mat in enumerate(matrices):
        cmat = xp.astype(xp.asarray(mat), xp.complex64)
        cmat = array_api_compat.to_device(cmat, device)

        z, _ = generate_timeseries(
            cmat,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            # NOTE: we add 1 so that z.shape == (maxit + 1, nx, ny, d)
            maxit=maxit + 1,
            xp=xp,
        )
        z = np.asarray(array_api_compat.to_device(z, "cpu"))

        escaped = compute_escaped_mask(z, escape_radius)

        # NOTE: nn-autoencoder-dmd.py drops the last timestep (`x = z[:, :-1]`)
        ranks[i] = compute_effective_rank(z[:-1], escaped[:-1], thresholds=thresholds)
        log.info(
            "[%02d/%d] %s",
            i + 1,
            len(matrices),
            "  ".join(
                f"{label} = {rank:4d}"
                for label, rank in zip(labels, ranks[i], strict=True)
            ),
        )

        del z

    return ranks


# }}}

# {{{ main


def main(
    filename: pathlib.Path,
    *,
    datakey: str = "matrices",
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 10.0,
    resolution: int = 128,
    maxit: int = 32,
    thresholds: tuple[float, ...] = (0.999, 0.9999, 0.99999),
    cache: pathlib.Path | None = None,
    basename: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from matplotlib.ticker import MaxNLocator

    from commonlib import set_matplotlib_style
    from nneuroutil.visualization import figure

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

    if matrices.shape[1] != matrices.shape[2]:
        log.error("Expected square matrices: %s.", matrices.shape)
        return 1

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    if np.any(thresholds_arr <= 0.0) or np.any(thresholds_arr >= 1.0):
        log.error("'thresholds' must be in (0, 1): %s", thresholds)
        return 1

    if basename is None:
        basename = filename.with_suffix("")

    outfile = basename.with_stem(
        f"{basename.stem}-energy-ranks-r{resolution:04d}-n{maxit:04d}"
    )

    # }}}

    # {{{ load or compute ranks

    if cache is not None:
        data = load_cache(cache, ("ranks", "thresholds"))
        if data is None:
            return 1

        ranks = data["ranks"]
        thresholds = tuple(data["thresholds"].tolist())
        log.info("Loading ranks from '%s'.", cache)
    else:
        ranks = compute_ranks(
            matrices,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
            thresholds=thresholds,
        )
        np.savez(
            outfile.with_suffix(".npz"),
            ranks=ranks,
            thresholds=np.asarray(thresholds),
            bbox=np.asarray(bbox),
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
        )

    # }}}

    # {{{ plot ranks

    labels = [f"{100.0 * t:g}%" for t in thresholds]
    positions = np.arange(1, len(labels) + 1)

    with figure(outfile, overwrite=overwrite) as fig:
        ax = fig.gca()

        boxplot = ax.boxplot(
            [ranks[:, j] for j in range(len(labels))],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
        )
        for j, box in enumerate(boxplot["boxes"]):
            box.set_facecolor(f"C{j}")
            box.set_alpha(0.4)

        rng = np.random.default_rng(0)
        for j, position in enumerate(positions):
            r = ranks[:, j]
            ax.plot(
                position + rng.uniform(-0.15, 0.15, size=r.size),
                r,
                "o",
                ms=5,
                alpha=0.6,
                color=f"C{j}",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Cumulative energy threshold")
        ax.set_ylabel("Effective rank")
        ax.set_ylim((0, 2 * matrices.shape[1]))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

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
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.999, 0.9999, 0.99999),
        metavar="T",
        help="Cumulative spectral energy thresholds",
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
            bbox=cast("tuple[float, float, float, float]", tuple(args.bbox)),
            escape_radius=args.escape_radius,
            resolution=args.resolution,
            maxit=args.maxit,
            thresholds=tuple(args.thresholds),
            cache=args.cache,
            basename=args.outfile,
            overwrite=args.force,
        )
    )

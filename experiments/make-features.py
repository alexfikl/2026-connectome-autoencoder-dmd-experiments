# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal

import array_api_compat
import numpy as np
import rich.logging
from typing_extensions import TypeAliasType, TypeVar

from commonlib import FeatureDataset

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Compute features for connectome matrices.

Example:

    > {SCRIPT_PATH.name} -k matrices --outfile result.npz data.npz
"""


ScalarTypeT = TypeVar("ScalarTypeT", bound=np.generic, default=Any)
Array1D = TypeAliasType(
    "Array1D",
    np.ndarray[tuple[int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
Array2D = TypeAliasType(
    "Array2D",
    np.ndarray[tuple[int, int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)


# {{{ make_matrix_features


def make_matrix_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    rows, cols = np.tril_indices(n, k=-1)

    features = mat[rows, cols]
    starts = np.array([0, features.size])
    descrs = ("Lower triangular matrix entries",)

    return FeatureDataset(
        x=features, labels=np.array(0), label_names=(), starts=starts, descrs=descrs
    )


# }}}


# {{{ make_eig_features


def make_eig_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    # NOTE: normalizing the eigenvalues does not seem to work very well. The
    # matrices themselves are already symmetric, in [-1, 1], with zero trace,
    # so they're normalized enough.
    eigs, _ = np.linalg.eigh(mat)

    return FeatureDataset(
        x=eigs,
        labels=np.array(0),
        label_names=(),
        starts=np.array([0, eigs.size]),
        descrs=("Eigenvalues",),
    )


# }}}


# {{{ make_unsigned_graph_features


def _global_efficiency(distances: Array2D[np.floating[Any]]) -> float:
    """Average global efficiency from an unweighted distance matrix."""
    n, _ = distances.shape
    if n < 2:
        return 0.0

    d = distances[np.triu_indices(n, k=1)]
    mask = np.isfinite(d) & (d > 0)
    with np.errstate(divide="ignore"):
        reciprocal = np.where(mask, 1.0 / d, 0.0)

    return float(2.0 * np.sum(reciprocal) / (n * (n - 1)))


def _local_efficiency(g: Any, distances: Array2D[np.floating[Any]]) -> float:
    """Average local efficiency from an unweighted distance matrix."""
    n, _ = distances.shape
    if n == 0:
        return 0.0

    total = 0.0
    for i in range(n):
        nbrs = np.asarray(g.neighbors(i))
        k = nbrs.size
        if k < 2:
            continue

        d = distances[np.ix_(nbrs, nbrs)][np.triu_indices(k, k=1)]
        mask = np.isfinite(d) & (d > 0)
        with np.errstate(divide="ignore"):
            reciprocal = np.where(mask, 1.0 / d, 0.0)

        total += 2.0 * np.sum(reciprocal) / (k * (k - 1))

    return float(total / n)


def _small_worldness(g: Any, *, niter: int = 1, nrand: int = 1) -> float:
    """Small-worldness coefficient (matching :func:`networkx.sigma`)."""
    if g.vcount() < 4 or g.ecount() < 2:
        return np.nan

    clustering = g.transitivity_undirected()
    path_length = g.average_path_length(unconn=True)

    random_clustering = []
    random_path_length = []
    for _ in range(nrand):
        null = g.copy()
        null.rewire(n=niter * g.ecount(), allowed_edge_types="simple")
        random_clustering.append(null.transitivity_undirected())
        random_path_length.append(null.average_path_length(unconn=True))

    cr = float(np.mean(random_clustering))
    lr = float(np.mean(random_path_length))
    if cr <= 0.0 or lr <= 0.0:
        return np.nan

    return float((clustering / cr) / (path_length / lr))


def _make_unsigned_graph_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    import random

    import igraph as ig

    from orbitkit.metrics import (
        compute_rich_club_coefficient,
        compute_weighted_clustering_coefficient_nx,
    )

    ig.set_random_number_generator(random.Random(42))  # ruff: ignore[suspicious-non-cryptographic-random-usage]

    # construct graph
    # NOTE: distance = 1/weight for some measures (shortest path on cost)
    g = ig.Graph.Weighted_Adjacency(mat, mode="undirected", attr="weight", loops=False)
    g.es["distance"] = (1.0 / np.asarray(g.es["weight"])).tolist()

    # unweighted shortest paths are reused by the efficiency measures
    distances = np.asarray(g.distances())

    features = []
    starts = [0]
    descrs = []

    # Strength: average weighted degree
    features.append(np.mean(g.strength(weights="weight")))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Strength")

    # Clustering Coefficient
    features.append(compute_weighted_clustering_coefficient_nx(mat))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Clustering Coefficient")

    # Modularity
    communities = g.community_leiden(
        objective_function="modularity", weights="weight", n_iterations=-1
    )
    features.append(g.modularity(communities.membership, weights="weight"))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Modularity")

    # Eigenvector Centrality
    ec = np.asarray(g.eigenvector_centrality(weights="weight"))
    norm = np.linalg.norm(ec)
    features.append(np.mean(ec / norm) if norm > 0.0 else 0.0)
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Eigenvector Centrality")

    # Rich-club Coefficient (mean across degrees)
    features.append(np.mean(list(compute_rich_club_coefficient(mat).values())))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Rich-club Coefficient")

    # Transitivity (unweighted)
    features.append(g.transitivity_undirected())
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Transitivity")

    # Characteristic Path Length
    features.append(g.average_path_length(weights="distance", unconn=True))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Characteristic Path Length")

    # Global Efficiency
    features.append(_global_efficiency(distances))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Global Efficiency")

    # Local Efficiency
    features.append(_local_efficiency(g, distances))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Local Efficiency")

    # Betweenness Centrality (mean across nodes, normalized as in networkx)
    n, _ = mat.shape
    if n > 2:
        bc = np.asarray(g.betweenness(weights="distance", directed=False))
        features.append(float(np.mean(bc)) * 2.0 / ((n - 1) * (n - 2)))
    else:
        features.append(0.0)
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Betweenness Centrality")

    # Small-worldness (requires connected graph)
    features.append(_small_worldness(g, niter=1, nrand=1))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Small-worldness")

    return FeatureDataset(
        x=np.array(features),
        labels=np.array(0),
        label_names=(),
        starts=np.array(starts),
        descrs=tuple(descrs),
    )


def make_unsigned_graph_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    return _make_unsigned_graph_features(np.abs(mat))


# }}}


# {{{ make_signed_graph_features


def make_signed_graph_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    import orbitkit.metrics as okm
    from orbitkit.clusters import signed_leiden_communities

    features = []
    starts = [0]
    descrs = []

    # Strength
    features.append(okm.compute_weighted_degree(mat))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Strength")

    # Clustering Coefficient
    features.append(
        okm.compute_weighted_clustering_coefficient_costantini(mat, variant=6)
    )
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Clustering Coefficient")

    # Local Assortativity
    features.append(
        okm.compute_local_assortativity_sabek(np.abs(mat), alpha=1.0, beta=1.0)
    )
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Local Assortativity")

    # Eigenvector Centrality
    features.append(okm.compute_eigenvector_centrality(mat).score)
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Eigenvector Centrality")

    # Modularity
    communities = signed_leiden_communities(mat)
    features.append(np.array(okm.compute_modularity(mat, communities)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Modularity")

    return FeatureDataset(
        x=np.hstack(features),
        labels=np.array(0),
        label_names=(),
        starts=np.array(starts),
        descrs=tuple(descrs),
    )


# }}}


# {{{ make_dmd_features


def make_dmd_features(
    mat: Array2D[np.floating[Any]],
    *,
    top_k: int = 3,
    bbox: tuple[float, float, float, float] = (-0.04, 0.04, -0.04, 0.04),
    escape_radius: float = 100.0,
    resolution: int = 64,
    maxit: int = 192,
) -> FeatureDataset:
    # NOTE: the DMD run is pretty slow, so we try to do it on the GPU
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        import torch

        from nneuroutil.torch_extras import available_device_names

        if any(dev.startswith("cuda") for dev in available_device_names()):
            torch.set_default_device("cuda")
            xp = array_api_compat.array_namespace(torch.empty(0))
        xp = array_api_compat.array_namespace(np.array(0))
    except ImportError:
        xp = array_api_compat.array_namespace(np.array(0))

    from commonlib import generate_dataset
    from nneuroutil.dmdc import build_dense_dmdc

    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    if top_k > n:
        raise ValueError(f"'top_k' must be smaller than matrix size: {top_k}")

    # {{{ generate data

    dtype = xp.result_type(mat.dtype, xp.complex64)

    Z, c = generate_dataset(
        xp.astype(xp.asarray(mat), dtype),
        bbox=bbox,
        resolution=resolution,
        escape_radius=escape_radius,
        maxit=maxit,
    )

    # NOTE: only keep the data we actually use from here on out
    mask = xp.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, axis=-1) < escape_radius**2
    Z = Z[:, mask, :]

    # }}}

    # {{{ build dataset

    # NOTE: Z has shape (maxit, converged, d)
    X1 = xp.reshape(Z[:-1], (-1, n))
    X2 = xp.reshape(Z[1:], (-1, n))
    U = xp.reshape(
        xp.broadcast_to(c[mask, None], (maxit - 1, xp.sum(mask), 1)),
        (-1, 1),
    )

    # }}}

    # {{{ features

    dmd = build_dense_dmdc(X1, U, X2, method="ridge")
    eigs, Phi = xp.linalg.eig(dmd.A.T)  # ty: ignore[unresolved-attribute]

    P = xp.linalg.pinv(Phi)
    G = xp.conj(X1).T @ X1
    coeff_energy = xp.maximum(xp.real(xp.diag(P @ G @ xp.conj(P).T)), 0.0)
    strengths = xp.linalg.norm(Phi, axis=0) * xp.sqrt(coeff_energy)

    order = xp.argsort(-strengths)[:top_k]
    eigs = eigs[order]
    strengths = strengths[order] / xp.sum(strengths)
    Phi = Phi[:, order]

    features = xp.concatenate([
        strengths,
        eigs.real,
        eigs.imag,
        xp.abs(Phi).T.ravel(),
    ])

    starts = np.asarray([
        *range(top_k),
        *(top_k + i for i in range(top_k)),
        *(2 * top_k + i for i in range(top_k)),
        *(3 * top_k + i * n for i in range(top_k)),
        features.size,
    ])
    descrs = (
        *(f"DMD Strength {i}" for i in range(top_k)),
        *(f"DMD Eigenvalue.re {i}" for i in range(top_k)),
        *(f"DMD Eigenvalue.im {i}" for i in range(top_k)),
        *(f"DMD Mode {i}" for i in range(top_k)),
    )

    # }}}

    return FeatureDataset(
        x=np.asarray(features),
        labels=np.array(0),
        label_names=(),
        starts=starts,
        descrs=descrs,
    )


# }}}


def make_features(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    ftype: Literal["sgraph", "ugraph", "matrix", "dmd", "eig"] = "matrix",
    key: str = "matrices",
    eps: float | None = None,
    density: float | None = None,
    overwrite: bool = False,
) -> int:
    if not filenames:
        log.warning("No files given.")
        return 0

    if outfile is None:
        outfile = filenames[0].parent / f"dataset-features-{ftype}.npz"

    if not overwrite and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
        return 1

    ret = 0
    datasets = {}

    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            ret = 1
            continue

        try:
            data = np.load(filename)
        except Exception:
            log.error("Failed to load data from file: '%s'.", filename)
            ret = 1
            continue

        if key not in data:
            log.error("File does not contain '%s' dataset: '%s'.", key, filename)
            ret = 1
            continue

        matrices = data[key]
        if matrices.ndim != 3:
            log.error("Expected a 3 dimensional array: %s.", matrices.shape)
            ret = 1
            continue

        feps = eps
        if feps is None:
            feps = 10.0 * np.finfo(matrices.dtype).eps

        log.info("Gathering '%s' features for '%s'.", ftype, filename)

        nsamples, _, _ = matrices.shape
        result: Array2D | None = None

        for i in range(nsamples):
            # ensure no self-connections
            mat_i = matrices[i].copy()
            np.fill_diagonal(mat_i, 0.0)

            # thresholding
            amat_i = np.abs(mat_i)
            if density is not None:
                cutoff = np.quantile(amat_i[amat_i > feps], 1 - density)
                mat_i = np.where(amat_i > cutoff, mat_i, 0.0)

            # NOTE: ensure the matrix is symmetric (some algorithms need that)
            mat_i = (mat_i + mat_i.T) / 2.0

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                if ftype == "matrix":
                    features = make_matrix_features(mat_i)
                elif ftype == "eig":
                    features = make_eig_features(mat_i)
                elif ftype == "sgraph":
                    features = make_signed_graph_features(mat_i)
                elif ftype == "ugraph":
                    features = make_unsigned_graph_features(mat_i)
                elif ftype == "dmd":
                    features = make_dmd_features(mat_i)
                else:
                    raise ValueError(f"unknown feature type: {ftype!r}")
            except Exception as exc:
                log.error(
                    "Failed to compute features for '%s' sample %d: %s",
                    filename,
                    i,
                    exc,
                )
                ret = 1
                continue

            nfeatures = np.size(features.x)
            if result is None:
                result = np.full((nsamples, nfeatures), np.nan, dtype=features.x.dtype)

            assert features.x.dtype == np.float64
            assert result.shape[1] == nfeatures
            assert features.starts[-1] == nfeatures
            assert np.sum(np.diff(features.starts)) == nfeatures

            result[i] = features.x

        assert filename.stem not in datasets
        datasets[filename.stem] = result

    np.savez(outfile, datasets=datasets, starts=features.starts, descrs=features.descrs)
    log.info("Saving computed features: '%s'.", outfile)

    return ret


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
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--features",
        choices=("sgraph", "matrix", "dmd", "eig", "ugraph"),
        default="matrix",
        help="Type of the features to compute",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Output file name",
    )
    parser.add_argument(
        "-k",
        "--key",
        default="matrices",
        help="Name of the variable in the file to use",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.95,
        help="Proportional threshold: keep top fraction of surviving edges (e.g. 0.20)",
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
        make_features(
            args.filenames,
            ftype=args.features,
            key=args.key,
            density=args.density,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

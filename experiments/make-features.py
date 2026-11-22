# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import random
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import array_api_compat
import numpy as np
import rich.logging
from typing_extensions import TypeAliasType, TypeVar

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
Array3D = TypeAliasType(
    "Array3D",
    np.ndarray[tuple[int, int, int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)


class FeatureDataset(NamedTuple):
    x: Array1D[np.floating[Any]]
    starts: Array1D[np.integer[Any]]
    descrs: tuple[str, ...]


# {{{ make_matrix_features


def make_matrix_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    rows, cols = np.tril_indices(n, k=-1)

    features = mat[rows, cols]
    starts = np.array([0, features.size])
    descrs = ("Lower triangular matrix entries",)

    return FeatureDataset(x=features, starts=starts, descrs=descrs)


# }}}


# {{{ make_scale_features


def make_scale_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    fro = np.linalg.norm(mat, ord="fro")
    eigs = np.linalg.eigvalsh(mat)
    spectral_radius = np.max(np.abs(eigs))
    entry_range = np.max(mat) - np.min(mat)

    features = np.array([fro, spectral_radius, entry_range], dtype=np.float64)
    starts = np.array([0, 1, 2, 3])
    descrs = ("Frobenius Norm", "Spectral Radius", "Entry Range")

    return FeatureDataset(x=features, starts=starts, descrs=descrs)


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
        x=eigs, starts=np.array([0, eigs.size]), descrs=("Eigenvalues",)
    )


# }}}


# {{{ make_local_graph_features


def _global_efficiency(distances: Array2D[np.floating[Any]]) -> float:
    """Average global efficiency from a weighted distance matrix."""
    n, _ = distances.shape
    if n < 2:
        return 0.0

    d = distances[np.triu_indices(n, k=1)]
    mask = np.isfinite(d) & (d > 0)
    with np.errstate(divide="ignore"):
        reciprocal = np.where(mask, 1.0 / d, 0.0)

    return float(2.0 * np.sum(reciprocal) / (n * (n - 1)))


def _make_community_labels(mat: Array2D[np.floating[Any]]) -> Array1D[np.integer[Any]]:
    """Detect communities on the unsigned graph of a single connectome."""
    import igraph as ig

    ig.set_random_number_generator(random.Random(42))  # ruff: ignore[suspicious-non-cryptographic-random-usage]

    weights = np.abs(np.array(mat, copy=True))
    np.fill_diagonal(weights, 0.0)
    weights = (weights + weights.T) / 2.0

    if not np.any(weights > 0.0):
        return np.zeros(weights.shape[0], dtype=np.int32)

    graph = ig.Graph.Weighted_Adjacency(
        weights, mode="undirected", attr="weight", loops=False
    )
    communities = graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=1.0,
        n_iterations=-1,
    )

    return np.asarray(communities.membership, dtype=np.int32)


def _make_local_graph_features(
    mat: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
) -> FeatureDataset:
    features = []
    starts = [0]
    descrs = []

    import orbitkit.metrics as okm

    # Strength: weighted degree
    features.append(okm.compute_weighted_degree(mat))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Strength")

    # Disparity
    features.append(okm.compute_disparity_serrano(mat))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Disparity")

    # Participation Coefficient
    features.append(okm.compute_participation_coefficient(mat, labels))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Participation Coefficient")

    # Eigenvector Centrality
    features.append(okm.compute_eigenvector_centrality(mat).score)
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Eigenvector Centrality")

    # Clustering Coefficient
    max_weight = np.max(mat) if mat.size > 0 else 0.0
    if max_weight > 0.0:
        wcc = np.nan_to_num(
            okm.compute_weighted_clustering_coefficient_costantini(
                mat / max_weight, variant=7
            )
        )
    else:
        wcc = np.zeros(mat.shape[0])

    features.append(wcc)
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Clustering Coefficient")

    # Local Assortativity
    features.append(
        np.nan_to_num(
            okm.compute_local_assortativity_sabek(mat, alpha=1.0, beta=1.0),
            nan=0.0,
        )
    )
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Local Assortativity")

    return FeatureDataset(
        x=np.hstack(features), starts=np.array(starts), descrs=tuple(descrs)
    )


def make_local_graph_features(mat: Array2D[np.floating[Any]]) -> FeatureDataset:
    amat = np.abs(mat)
    labels = _make_community_labels(amat)

    fabs = _make_local_graph_features(amat, labels)
    fpos = _make_local_graph_features(np.maximum(mat, 0.0), labels)
    fneg = _make_local_graph_features(np.maximum(-mat, 0.0), labels)

    return FeatureDataset(
        x=np.concatenate([fabs.x, fpos.x, np.nan_to_num(fneg.x)]),
        starts=np.concatenate([
            fabs.starts,
            fabs.starts[-1] + fpos.starts[1:],
            fabs.starts[-1] + fpos.starts[-1] + fneg.starts[1:],
        ]),
        descrs=(
            *(f"Abs {name}" for name in fabs.descrs),
            *(f"Pos {name}" for name in fpos.descrs),
            *(f"Neg {name}" for name in fneg.descrs),
        ),
    )


# }}}


# {{{ make_global_graph_features


def _greedy_vertex_cover(
    rows: Array1D[np.integer[Any]],
    cols: Array1D[np.integer[Any]],
    size: int,
) -> Array1D[np.integer[Any]]:
    """Greedily cover all the given edges with as few vertices as possible."""
    alive = np.ones(rows.size, dtype=bool)
    cover = []
    while np.any(alive):
        degree = np.bincount(np.concatenate([rows[alive], cols[alive]]), minlength=size)
        vertex = int(np.argmax(degree))
        if degree[vertex] == 0:
            break

        cover.append(vertex)
        alive &= (rows != vertex) & (cols != vertex)

    return np.array(cover, dtype=np.int64)


def _make_local_distances(
    u: int,
    neighbors: Array1D[np.integer[Any]],
    adjacency: Array2D[np.bool_],
    lengths: Any,
    full: Array2D[np.floating[Any]],
    *,
    tol: float = 1.0e-10,
) -> Array2D[np.floating[Any]]:
    """Distances in the subgraph induced by the neighbours of ``u``.

    The induced subgraph is obtained from the full graph by removing ``u`` and
    all the nodes that are not adjacent to it. Only a small number of shortest
    paths passes through the removed nodes, so the full distances are reused
    and only the affected pairs are recomputed with Dijkstra.
    """
    from scipy.sparse.csgraph import dijkstra

    k = neighbors.size
    distances = full[np.ix_(neighbors, neighbors)]

    # NOTE: a shortest path between two neighbours stays the same unless one of
    # the removed nodes lies on it, which is the case if and only if going
    # through that node is as short as the full distance.
    flagged = np.zeros((k, k), dtype=bool)
    for v in np.flatnonzero(~adjacency[u, :]):
        dv = full[v, neighbors]
        flagged |= (dv[:, None] + dv[None, :]) <= distances + tol
    flagged &= np.isfinite(distances)

    rows, cols = np.nonzero(np.triu(flagged, k=1))
    if rows.size == 0:
        return distances

    # NOTE: Dijkstra from one endpoint of every affected pair is enough, so
    # only a vertex cover of the affected pairs is used as a source set.
    cover = _greedy_vertex_cover(rows, cols, k)
    covered = np.zeros(k, dtype=bool)
    covered[cover] = True
    sources = np.where(covered[rows], rows, cols)
    targets = np.where(covered[rows], cols, rows)

    corrections = dijkstra(
        lengths[neighbors][:, neighbors], directed=False, indices=cover
    )

    positions = np.full(k, -1, dtype=np.int64)
    positions[cover] = np.arange(cover.size)
    values = corrections[positions[sources], targets]

    distances[rows, cols] = values
    distances[cols, rows] = values

    return distances


def _local_efficiency(
    mat: Array2D[np.floating[Any]],
    graph: Any,
    *,
    eps: float = 1.0e-12,
    dense_threshold: float = 0.8,
) -> Array1D[np.floating[Any]]:
    """Compute the per-node local efficiency of a weighted graph."""
    import scipy.sparse as sp

    n, _ = mat.shape
    adjacency = mat > eps
    cbrt_weights = np.cbrt(mat)

    # NOTE: for dense graphs the neighbourhood of every node is almost the full
    # graph, so the distances in all the neighbourhoods can be derived from a
    # single all-pairs shortest path computation.
    dense = n > 1 and float(np.mean(adjacency)) > dense_threshold
    lengths = None
    full = None
    if dense:
        rows, cols = np.nonzero(adjacency)
        lengths = sp.csr_matrix((1.0 / mat[rows, cols], (rows, cols)), shape=(n, n))
        full = np.asarray(graph.distances(weights="length"))

    efficiency = np.zeros(n)
    for u in range(n):
        neighbors = np.flatnonzero(adjacency[u, :])
        if neighbors.size < 2:
            continue

        if dense:
            assert full is not None
            distances = _make_local_distances(u, neighbors, adjacency, lengths, full)
        else:
            subgraph = graph.induced_subgraph(neighbors.tolist())
            distances = np.asarray(subgraph.distances(weights="length"))

        with np.errstate(divide="ignore"):
            inverse = np.where(
                np.isfinite(distances) & (distances > 0), 1.0 / distances, 0.0
            )

        np.fill_diagonal(inverse, 0.0)
        se = inverse ** (1.0 / 3.0) + inverse.T ** (1.0 / 3.0)

        sw = cbrt_weights[u, neighbors] + cbrt_weights[neighbors, u].T
        numerator = np.sum(np.outer(sw, sw) * se) / 2.0
        if numerator == 0.0:  # ruff: ignore[float-equality-comparison]
            continue

        sa = adjacency[u, neighbors].astype(np.float64)
        sa += adjacency[neighbors, u].T.astype(np.float64)
        denominator = np.sum(sa) ** 2 - np.sum(sa * sa)
        if denominator != 0.0:  # ruff: ignore[float-equality-comparison]
            efficiency[u] = numerator / denominator

    return efficiency


def _make_global_graph_features(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float = 1.0e-12,
) -> FeatureDataset:
    """Average the paper's graph measures over all nodes.

    This follows the Brain Connectivity Toolbox definitions: weights are
    interpreted as connection strengths and converted to lengths with
    ``1 / w``.
    """
    import igraph as ig

    ig.set_random_number_generator(random.Random(42))  # ruff: ignore[suspicious-non-cryptographic-random-usage]

    # construct graph
    g = ig.Graph.Weighted_Adjacency(mat, mode="undirected", attr="weight", loops=False)
    g.es["length"] = [1.0 / w if w > eps else np.inf for w in g.es["weight"]]
    distances = np.asarray(g.distances(weights="length"))

    features = []
    starts = [0]
    descrs = []

    import orbitkit.metrics as okm

    # Betweenness Centrality: average weighted betweenness
    features.append(2.0 * np.mean(g.betweenness(weights="length")))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Betweenness Centrality")

    # Assortativity: weighted strength assortativity
    features.append(okm.compute_assortativity_newman(mat, eps=eps))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Assortativity")

    # Eigencentrality: average absolute eigenvector centrality
    _, eigvecs = np.linalg.eigh(mat)
    features.append(np.mean(np.abs(eigvecs[:, -1])))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Eigencentrality")

    # Clustering Coefficient: average weighted clustering
    degree = np.count_nonzero(mat > eps, axis=1)
    cbrt_weights = np.cbrt(mat)
    cyc3 = np.einsum("ij,jk,ki->i", cbrt_weights, cbrt_weights, cbrt_weights)
    wcc = np.zeros(mat.shape[0])
    mask = (cyc3 != 0.0) & (degree >= 2)  # ruff: ignore[float-equality-comparison]
    wcc[mask] = cyc3[mask] / (degree[mask] * (degree[mask] - 1))
    features.append(np.mean(wcc))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Clustering Coefficient")

    # Strength: average weighted degree
    features.append(np.mean(np.sum(mat, axis=0)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Strength")

    # Global Efficiency
    features.append(_global_efficiency(distances))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Global Efficiency")

    # Local Efficiency
    features.append(np.mean(_local_efficiency(mat, g, eps=eps)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Local Efficiency")

    # Transitivity: ratio of weighted triangles to triplets
    denominator = np.sum(degree * (degree - 1))
    transitivity = np.sum(cyc3) / denominator if denominator > 0 else 0.0
    features.append(transitivity)
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Transitivity")

    return FeatureDataset(
        x=np.nan_to_num(np.hstack(features)),
        starts=np.array(starts),
        descrs=tuple(descrs),
    )


def make_global_graph_features(
    mat: Array2D[np.floating[Any]],
) -> FeatureDataset:
    fabs = _make_global_graph_features(np.abs(mat))
    fpos = _make_global_graph_features(np.maximum(mat, 0.0))
    fneg = _make_global_graph_features(np.maximum(-mat, 0.0))

    return FeatureDataset(
        x=np.concatenate([fabs.x, fpos.x, fneg.x]),
        starts=np.concatenate([
            fabs.starts,
            fabs.starts[-1] + fpos.starts[1:],
            fabs.starts[-1] + fpos.starts[-1] + fneg.starts[1:],
        ]),
        descrs=(
            *(f"Abs {name}" for name in fabs.descrs),
            *(f"Pos {name}" for name in fpos.descrs),
            *(f"Neg {name}" for name in fneg.descrs),
        ),
    )


# }}}


# {{{ make_dmd_features


def make_dmd_features(
    mat: Array2D[np.floating[Any]],
    *,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 100.0,
    resolution: int = 64,
    maxit: int = 32,
    full_matrix: bool = False,
) -> FeatureDataset:
    # NOTE: the DMD run is pretty slow, so we try to do it on the GPU
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        import torch

        from nneuroutil.interop.torch import available_device_names

        if any(dev.startswith("cuda") for dev in available_device_names()):
            device = "cuda"
            torch.set_default_device(device)
            xp = array_api_compat.array_namespace(torch.empty(0))
        else:
            device = "cpu"
            xp = array_api_compat.array_namespace(np.array(0))
    except ImportError:
        device = "cpu"
        xp = array_api_compat.array_namespace(np.array(0))

    # log.info("Computing DMD modes using '%s' (%s).", xp.__name__, device)

    from commonlib import generate_timeseries
    from nneuroutil.dmdc import build_dense_dmdc

    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    # {{{ generate data

    # NOTE: complex64 is both smaller and faster -- we need some speed here
    cmat = xp.astype(xp.asarray(mat, device=device), xp.complex64)

    Z, c = generate_timeseries(
        cmat,
        bbox=bbox,
        resolution=resolution,
        escape_radius=escape_radius,
        maxit=maxit,
    )

    from commonlib import compute_trajectory_mask
    from nneuroutil.array_api_extras import view_as_real

    # NOTE: permute to (num_traj, maxit, n) and keep the complex states
    Z_perm = xp.permute_dims(xp.reshape(Z, (maxit, -1, n)), (1, 0, 2))
    c_flat = xp.reshape(c, (-1,))

    num_samples = Z_perm.shape[0]

    # }}}

    # {{{ build dataset

    # NOTE: `compute_trajectory_mask` expects interleaved real states
    tmask = compute_trajectory_mask(
        view_as_real(Z_perm), escape_radius=escape_radius, xp=xp
    )

    X1 = Z_perm[:, :-1, :][tmask.transition]
    X2 = Z_perm[:, 1:, :][tmask.transition]
    U = xp.broadcast_to(c_flat[:, None, None], (num_samples, maxit - 1, 1))[
        tmask.transition
    ]

    # NOTE: X1, X2 and U are copies, so the trajectories can be released early
    del Z, Z_perm, tmask, c_flat, c, cmat

    # }}}

    # {{{ features

    dmd = build_dense_dmdc(X1, U, X2, method="ridge")
    del U, X2

    G = xp.conj(X1).T @ X1
    del X1

    G = np.asarray(array_api_compat.to_device(G, "cpu"))
    A = np.asarray(array_api_compat.to_device(dmd.A, "cpu"))

    if full_matrix:
        ds = make_matrix_features(A)
    else:
        from threadpoolctl import threadpool_limits

        # NOTE: openmp does not seem to help much here, so we turn it off
        with threadpool_limits(limits=1):
            eigs, Phi = np.linalg.eig(A.T)

        P = np.linalg.pinv(Phi)
        coeff_energy = np.maximum(np.real(np.sum((P @ G) * np.conj(P), axis=1)), 0.0)
        strengths = np.linalg.norm(Phi, axis=0) * np.sqrt(coeff_energy)

        order = np.argsort(-strengths)
        strengths = strengths[order]

        order = np.argsort(np.abs(eigs))
        eigs = np.abs(eigs[order])

        features = np.concatenate([strengths, eigs], dtype=mat.dtype)
        starts = np.cumsum([0, strengths.size, eigs.size])
        descrs = ("DMD Strengths", "DMD Eigs")

        ds = FeatureDataset(x=features, starts=starts, descrs=descrs)

    # }}}

    return ds


# }}}


# {{{ make_image_features


def make_image_features(
    binary: Array2D[np.integer[Any]],
    *,
    size: int = 32,
) -> FeatureDataset:
    """Block-average a binary equi-M set image to a ``size x size`` image."""
    height, width = binary.shape
    if height % size != 0 or width % size != 0:
        raise ValueError(f"image size {height}x{width} is not divisible by {size}")

    blocks = binary.reshape(size, height // size, size, width // size)
    features = blocks.mean(axis=(1, 3), dtype=np.float64).reshape(-1)

    return FeatureDataset(
        x=features,
        starts=np.array([0, features.size]),
        descrs=("Image",),
    )


# }}}


# {{{ _make_contour


def _make_contour(
    binary: Array2D[np.integer[Any]],
    bbox: tuple[float, float, float, float],
) -> tuple[Any, float, float, float]:
    """Retrieve the largest outer contour of the binary mask.

    Returns the contour together with the pixel scales ``dx`` and ``dy`` and
    the contour perimeter in pixels.
    """
    import cv2  # ty: ignore[import-not-found]

    xmin, xmax, ymin, ymax = bbox
    height, width = binary.shape
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    mask = np.where(binary == 0, np.uint8(255), np.uint8(0))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("could not find contours")

    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        raise RuntimeError("invalid contour")

    perimeter_px = cv2.arcLength(c, closed=True)
    return c, dx, dy, perimeter_px


# }}}


# {{{ make_fourier_features


def make_fourier_features(
    binary: Array2D[np.integer[Any]],
    bbox: tuple[float, float, float, float],
    *,
    ncoeffs: int = 5,
    nsamples: int = 128,
    eps: float = 5.0e-4,
    normalize: bool = True,
) -> FeatureDataset:
    """Compute Fourier descriptors of the boundary contour.

    The groups contain the real and imaginary parts of the low-order boundary
    modes ``m_{-ncoeffs}, ..., m_{-1}, m_1, ..., m_{ncoeffs}``.
    """
    import cv2  # ty: ignore[import-not-found]

    if ncoeffs < 1:
        raise ValueError(f"'ncoeffs' must be positive: {ncoeffs}")

    if nsamples < 2 * ncoeffs + 1:
        raise ValueError(
            f"'nsamples' must be at least 2 * ncoeffs + 1: "
            f"{nsamples} < {2 * ncoeffs + 1}"
        )

    xmin, xmax, ymin, ymax = bbox
    height, width = binary.shape
    c, _, _, perimeter_px = _make_contour(binary, bbox)

    # approximate polygon and resample to equidistant arc-length points
    approx = cv2.approxPolyDP(c, perimeter_px * eps, closed=True)
    pts = approx[:, 0, :]

    # NOTE: image row 0 corresponds to ymax, as in the paper's MATLAB code
    x = xmin + pts[:, 0] * ((xmax - xmin) / width)
    y = ymax - pts[:, 1] * ((ymax - ymin) / height)

    z = x + 1j * y
    z = np.append(z, z[0])
    dists = np.insert(np.cumsum(np.abs(np.diff(z))), 0, 0.0)
    target_dists = np.linspace(0.0, dists[-1], nsamples, endpoint=False)

    z = np.interp(target_dists, dists, z.real) + 1j * np.interp(
        target_dists, dists, z.imag
    )

    # symmetric low-order boundary descriptors
    # NOTE: centering makes the m_0 coefficient zero, so it is omitted
    z = z - np.mean(z)
    if normalize:
        scale = np.sqrt(np.mean(np.abs(z) ** 2))
        if scale < 1.0e-12:
            scale = 1.0
        z = z / scale

    # NOTE: the phases depend on where the contour starts, so anchor it at
    # the rightmost point, which is stable across all equi-M sets
    z = np.roll(z, -int(np.argmax(z.real)))

    F = np.fft.fft(z)
    freqs = np.rint(np.fft.fftfreq(nsamples, d=1.0 / nsamples)).astype(int)
    index = {int(k): i for i, k in enumerate(freqs)}
    modes = [*range(-ncoeffs, 0), *range(1, ncoeffs + 1)]
    values = F[[index[k] for k in modes]]

    descriptors = np.concatenate([values.real, values.imag])
    starts = np.cumsum([0, values.size, values.size])

    return FeatureDataset(
        x=descriptors,
        starts=starts,
        descrs=("Descriptor Real", "Descriptor Imag"),
    )


# }}}


# {{{ make_geometry_features


def make_geometry_features(
    binary: Array2D[np.integer[Any]],
    bbox: tuple[float, float, float, float],
) -> FeatureDataset:
    """Compute geometric measures of the boundary contour.

    The groups contain the real-axis chord, the bounding box, area and
    perimeter, and orientation-invariant landmark measures from Radulescu
    et al. (2025). No landmark-side assumptions are made.
    """
    import cv2  # ty: ignore[import-not-found]

    xmin, _, _, ymax = bbox
    height = binary.shape[0]
    c, dx, dy, perimeter_px = _make_contour(binary, bbox)

    # geometry
    area = cv2.contourArea(c) * dx * dy
    perimeter = perimeter_px * np.sqrt(dx * dy)

    px = xmin + c[:, 0, 0] * dx
    py = ymax - c[:, 0, 1] * dy

    inside = binary == 0
    ys, xs = np.nonzero(inside)
    x_max = xmin + np.max(xs) * dx
    y_min = ymax - np.max(ys) * dy
    y_max = ymax - np.min(ys) * dy

    # real-axis chord: intersection of the set with y = 0
    i0 = int(np.clip(np.rint((ymax - 0.0) / dy), 0, height - 1))
    cols = np.flatnonzero(inside[i0, :])
    if cols.size == 0:
        rows = np.flatnonzero(inside.any(axis=1))
        i0 = int(rows[np.argmin(np.abs(rows - i0))])
        cols = np.flatnonzero(inside[i0, :])
    x_min_y0 = xmin + np.min(cols) * dx
    x_max_y0 = xmin + np.max(cols) * dx

    diameter_x = x_max_y0 - x_min_y0
    diameter_y = y_max - y_min
    epsilon = diameter_y / max(diameter_x, 1.0e-12)

    x_at_y_max = px[int(np.argmax(py))]
    y_at_x_max = abs(py[int(np.argmax(px))])

    features = np.array([
        x_min_y0,
        x_max_y0,
        x_max,
        y_max,
        area,
        perimeter,
        diameter_x,
        diameter_y,
        epsilon,
        x_at_y_max,
        y_at_x_max,
    ])
    starts = np.cumsum([0, *([1] * features.size)])

    return FeatureDataset(
        x=features,
        starts=starts,
        descrs=(
            "x_min_y0",
            "x_max_y0",
            "x_max",
            "y_max",
            "area",
            "perimeter",
            "d_TC",
            "d_RR",
            "epsilon",
            "x_at_y_max",
            "y_at_x_max",
        ),
    )


# }}}


def make_features(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    ftype: Literal[
        "local",
        "global",
        "matrix",
        "dmd",
        "eig",
        "geometry",
        "fourier",
        "image",
        "scale",
    ] = "matrix",
    key: str | None = None,
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

    if key is None:
        key = (
            "images_binary" if ftype in {"geometry", "fourier", "image"} else "matrices"
        )

    # {{{ load matrices

    ret = 0
    inputs = {}
    bboxes = {}
    scan_types: dict[str, str] = {}

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

        if "scan_type" not in data:
            log.error("File does not contain 'scan_type' dataset: '%s'.", filename)
            ret = 1
            continue

        scan_type = str(data["scan_type"])

        feps = eps
        if feps is None:
            try:
                feps = 10.0 * np.finfo(matrices.dtype).eps
            except ValueError:
                feps = -1.0

        if ftype in {"geometry", "fourier", "image"}:
            result = matrices
            bbox = data["images_bbox"]
        else:
            # NOTE: ensure no self-connections in any matrix of the batch
            mat = np.array(matrices, copy=True)
            rows, cols = np.diag_indices(mat.shape[-1])
            mat[:, rows, cols] = 0.0

            if density is not None:
                # NOTE: the proportional threshold is applied to each matrix
                # separately (i.e. cost matching), keeping the top 'density'
                # fraction of its weights
                amat = np.abs(mat)
                irows, icols = np.triu_indices(amat.shape[-1], k=1)

                values = amat[:, irows, icols]
                with np.errstate(all="ignore"):
                    cutoff = np.nanquantile(
                        np.where(values > feps, values, np.nan),
                        1.0 - density,
                        axis=1,
                    )

                cutoff = np.where(np.isnan(cutoff), np.inf, cutoff)
                mat = np.where(amat > cutoff[:, None, None], mat, 0.0)

            # NOTE: symmetrize each matrix (swap the last two axes only)
            result = (mat + np.swapaxes(mat, -1, -2)) / 2.0
            bbox = data["images_bbox"]

        assert filename.stem not in inputs
        inputs[filename.stem] = result
        bboxes[filename.stem] = bbox
        scan_types[filename.stem] = scan_type

    # }}}

    # {{{ compute features

    datasets = {}

    for name, matrices in inputs.items():
        log.info("Gathering '%s' features for '%s'.", ftype, name)

        nsamples, _, _ = matrices.shape
        result: Array2D | None = None

        for i in range(nsamples):
            mat_i = matrices[i]
            bbox_i = bboxes[name][i] if np.ndim(bboxes[name]) == 2 else bboxes[name]
            xmin, xmax, ymin, ymax = bbox_i

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                if ftype == "matrix":
                    features = make_matrix_features(mat_i)
                elif ftype == "scale":
                    features = make_scale_features(mat_i)
                elif ftype == "eig":
                    features = make_eig_features(mat_i)
                elif ftype == "local":
                    features = make_local_graph_features(mat_i)
                elif ftype == "global":
                    features = make_global_graph_features(mat_i)
                elif ftype == "dmd":
                    features = make_dmd_features(mat_i, bbox=(xmin, xmax, ymin, ymax))
                elif ftype == "geometry":
                    features = make_geometry_features(
                        mat_i,
                        (xmin, xmax, ymin, ymax),
                    )
                elif ftype == "fourier":
                    features = make_fourier_features(
                        mat_i,
                        (xmin, xmax, ymin, ymax),
                    )
                elif ftype == "image":
                    features = make_image_features(mat_i)
                else:
                    raise ValueError(f"unknown feature type: {ftype!r}")
            except Exception as exc:
                log.error(
                    "Failed to compute features for '%s' sample %d: %s",
                    name,
                    i,
                    exc,
                )
                ret = 1
                continue

            nfeatures = np.size(features.x)
            if result is None:
                result = np.full((nsamples, nfeatures), np.nan, dtype=features.x.dtype)

            assert np.all(np.isfinite(features.x))
            assert features.x.dtype == np.float64
            assert result.shape[1] == nfeatures
            assert features.starts[-1] == nfeatures
            assert np.sum(np.diff(features.starts)) == nfeatures

            result[i] = features.x

        assert name not in datasets
        datasets[name] = result

    # }}}

    np.savez(
        outfile,
        datasets=datasets,
        scan_types=scan_types,
        starts=features.starts,
        descrs=features.descrs,
        density=density,
    )
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
        choices=(
            "local",
            "global",
            "matrix",
            "dmd",
            "eig",
            "geometry",
            "fourier",
            "image",
            "scale",
        ),
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
        default=None,
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

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
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


# {{{ make_unsigned_graph_features


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


def _make_unsigned_graph_features(
    mat: Array2D[np.floating[Any]],
    communities: Communities,
    *,
    eps: float = 1.0e-12,
) -> FeatureDataset:
    import igraph as ig

    ig.set_random_number_generator(random.Random(42))  # ruff: ignore[suspicious-non-cryptographic-random-usage]

    # construct graph
    g = ig.Graph.Weighted_Adjacency(mat, mode="undirected", attr="weight", loops=False)

    # unweighted shortest paths are reused by the efficiency measures
    g.es["distance"] = [-np.log(w) if w > eps else np.inf for w in g.es["weight"]]
    distances = np.asarray(g.distances(weights="distance"))

    features = []
    starts = [0]
    descrs = []

    import orbitkit.metrics as okm

    # Strength: average weighted degree
    features.append(communities.mean(okm.compute_weighted_degree(mat)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Strength")

    # Strength: average excess weighted degree
    features.append(communities.mean(okm.compute_average_excess_strength(mat)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Excess Strength")

    # Disparity
    features.append(communities.mean(okm.compute_disparity_serrano(mat)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Disparity")

    # Spectral Radius
    eigs = np.linalg.eigvalsh(mat)
    features.append(eigs[-1])
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Spectral Radius")

    # Spectral Participation Ratio
    features.append(np.sum(eigs**2) ** 2 / np.sum(eigs**4))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Spectral Participation Ratio")

    # Spectral Entropy
    p = eigs**2 / np.sum(eigs**2)
    features.append(-np.sum(p * np.log(p + 1.0e-16)))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Spectral Entropy")

    # Eigenvector Centrality
    features.append(communities.mean(okm.compute_eigenvector_centrality(mat).score))
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

    features.append(communities.mean(wcc))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Clustering Coefficient")

    # Global Efficiency
    features.append(_global_efficiency(distances))
    starts.append(starts[-1] + np.size(features[-1]))
    descrs.append("Global Efficiency")

    return FeatureDataset(
        x=np.hstack(features), starts=np.array(starts), descrs=tuple(descrs)
    )


def make_unsigned_graph_features(
    mat: Array2D[np.floating[Any]], communities: Communities
) -> FeatureDataset:
    fabs = _make_unsigned_graph_features(np.abs(mat), communities)
    fpos = _make_unsigned_graph_features(np.maximum(mat, 0.0), communities)
    fneg = _make_unsigned_graph_features(np.maximum(-mat, 0.0), communities)

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
    # NOTE: this can become nan sometimes when the node has no neighbors
    features.append(
        np.nan_to_num(
            okm.compute_weighted_clustering_coefficient_costantini(mat, variant=6),
            nan=0.0,
        )
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
        x=np.hstack(features), starts=np.array(starts), descrs=tuple(descrs)
    )


# }}}


# {{{ make_dmd_features


def make_dmd_features(
    mat: Array2D[np.floating[Any]],
    *,
    bbox: tuple[float, float, float, float] = (-0.04, 0.04, -0.04, 0.04),
    escape_radius: float = 100.0,
    resolution: int = 64,
    maxit: int = 128,
    stride: int = 2,
    full_matrix: bool = False,
) -> FeatureDataset:
    # NOTE: the DMD run is pretty slow, so we try to do it on the GPU
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        import torch

        from nneuroutil.torch_extras import available_device_names

        if any(dev.startswith("cuda") for dev in available_device_names()):
            torch.set_default_device("cuda")
            device = "CUDA"
            xp = array_api_compat.array_namespace(torch.empty(0))
        else:
            device = "CPU"
            xp = array_api_compat.array_namespace(np.array(0))
    except ImportError:
        device = "CPU"
        xp = array_api_compat.array_namespace(np.array(0))

    log.info("Computing DMD modes using '%s' (%s).", xp.__name__, device)

    from commonlib import generate_dataset
    from nneuroutil.dmdc import build_dense_symmetric_dmdc

    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    # {{{ generate data

    # NOTE: complex64 is both smaller and faster -- we need some speed here
    cmat = xp.astype(xp.asarray(mat), xp.complex64)

    Z, c = generate_dataset(
        cmat,
        bbox=bbox,
        resolution=resolution,
        escape_radius=escape_radius,
        maxit=maxit,
    )

    # NOTE: only keep the data we actually use from here on out
    mask = xp.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, axis=-1) < escape_radius**2
    Z = Z[:, mask, :]
    c = c[mask]

    # cull down the solutions, since a lot of them converge to the same value
    Z = Z[:, ::stride, :]
    c = c[::stride]
    num_samples = Z.shape[1]

    # }}}

    # {{{ build dataset

    # NOTE: Z has shape (maxit, num_samples, d)
    X1 = xp.reshape(Z[:-1], (-1, n))
    X2 = xp.reshape(Z[1:], (-1, n))
    U = xp.reshape(
        xp.broadcast_to(c[None, :, None], (maxit - 1, num_samples, 1)),
        (-1, 1),
    )

    # }}}

    # {{{ features

    dmd = build_dense_symmetric_dmdc(X1, U, X2, method="ridge")
    G = xp.conj(X1).T @ X1

    G = np.asarray(array_api_compat.to_device(G, "cpu"), dtype=mat.dtype)
    A = np.asarray(array_api_compat.to_device(dmd.A, "cpu"), dtype=mat.dtype)

    if full_matrix:
        ds = make_matrix_features(A)
    else:
        from threadpoolctl import threadpool_limits

        # NOTE: openmp does not seem to help much here, so we turn it off
        with threadpool_limits(limits=1):
            eigs, P = np.linalg.eigh(A.T)

        coeff_energy = np.maximum(
            np.real(np.sum((np.conj(P).T @ G) * P.T, axis=1)), 0.0
        )
        strengths = np.sqrt(coeff_energy)

        order = np.argsort(-strengths)
        strengths = strengths[order] / np.sum(strengths)

        features = np.concatenate([strengths, eigs])
        starts = np.cumsum([0, strengths.size, eigs.size])
        descrs = ("DMD Strengths", "DMD Eigs")

        ds = FeatureDataset(x=features, starts=starts, descrs=descrs)

    # }}}

    return ds


# }}}


# {{{ make_fourier_features


def make_fourier_features(
    binary: Array2D[np.integer[Any]],
    bbox: tuple[float, float, float, float],
    *,
    nmodes: int = 32,
    eps: float = 5.0e-4,
    normalize: bool = True,
) -> FeatureDataset:
    import cv2  # ty: ignore[unresolved-import]

    xmin, xmax, ymin, ymax = bbox
    height, width = binary.shape
    mask = np.where(binary == 0, np.uint8(255), np.uint8(0))

    # 1. retrieve outer contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("could not find contours")

    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, closed=True)
    if len(c) < 3:
        raise RuntimeError("invalid contour")

    # 2. approximate polygon
    approx = cv2.approxPolyDP(c, perimeter * eps, closed=True)
    pts = approx[:, 0, :]

    x = xmin + pts[:, 0] * ((xmax - xmin) / width)
    y = ymin + pts[:, 1] * ((ymax - ymin) / height)

    # 3. resample to equidistant arc-length points
    z = x + 1j * y
    z = np.append(z, z[0])
    dists = np.insert(np.cumsum(np.abs(np.diff(z))), 0, 0.0)
    target_dists = np.linspace(0.0, dists[-1], nmodes, endpoint=False)

    z = np.interp(target_dists, dists, z.real) + 1j * np.interp(
        target_dists, dists, z.imag
    )

    # 4. Fourier descriptors
    z = z - np.mean(z)
    if normalize:
        scale = np.sqrt(np.mean(np.abs(z) ** 2))
        if scale < 1.0e-12:
            scale = 1.0
        z = z / scale

    fourier = np.abs(np.fft.fft(z)[2:])

    # 5. shape and moment descriptors of the region
    area = cv2.contourArea(c)
    (_bx, _by, bw, bh) = cv2.boundingRect(c)
    hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0e-12)

    circularity = 4.0 * np.pi * area / max(perimeter**2, 1.0e-12)
    aspect = bw / max(bh, 1)
    solidity = area / hull_area
    extent = area / max(bw * bh, 1)

    if normalize:
        # NOTE: drop the absolute size and keep only scale invariant ratios
        morphology = np.array([circularity, aspect, solidity, extent])
    else:
        morphology = np.array([
            np.log1p(area),
            np.log1p(perimeter),
            circularity,
            aspect,
            solidity,
            extent,
        ])

    moments = cv2.moments((binary == 0).astype(np.uint8), binaryImage=True)
    hu = np.asarray(cv2.HuMoments(moments)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1.0e-30)

    x = np.concatenate([fourier, morphology, hu])
    starts = np.cumsum([0, fourier.size, morphology.size, hu.size])

    return FeatureDataset(
        x=x, starts=starts, descrs=("Fourier", "Morphology", "HuMoments")
    )


# }}}


@dataclass(frozen=True)
class Communities:
    size: int
    indices: Array1D[np.integer[Any]]
    counts: Array1D[np.integer[Any]]

    def mean(self, x: Array1D[np.floating[Any]]) -> Array1D[np.floating[Any]]:
        xm = np.bincount(self.indices, weights=x, minlength=self.size)
        return xm / np.maximum(self.counts, 1)


def make_template_communities(
    inputs: dict[str, Array3D[np.floating[Any]]],
    *,
    resolution: float = 3.0,
    seed: int = 42,
) -> Communities | None:
    from orbitkit.clusters import community_labels, signed_leiden_communities

    if not inputs:
        return None

    means = []
    for matrices in inputs.values():
        means.append(np.mean(matrices, axis=0))

    mat = np.mean(means, axis=0)
    communities = signed_leiden_communities(mat, resolution=resolution, seed=seed)

    indices = community_labels(communities, mat.shape[0])
    ncommunities = np.max(indices)
    counts = np.bincount(indices, minlength=ncommunities)

    return Communities(size=int(ncommunities), indices=indices, counts=counts)


def make_features(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    ftype: Literal["sgraph", "ugraph", "matrix", "dmd", "eig", "fourier"] = "matrix",
    key: str | None = None,
    eps: float | None = None,
    density: float | None = None,
    community_resolution: float = 3.0,
    community_seed: int = 42,
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
        key = "images_binary" if ftype == "fourier" else "matrices"

    # {{{ load matrices

    ret = 0
    inputs = {}

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
            try:
                feps = 10.0 * np.finfo(matrices.dtype).eps
            except ValueError:
                feps = -1.0

        if ftype == "fourier":
            result = matrices
        else:
            mat = np.array(matrices, copy=True)
            np.fill_diagonal(mat, 0.0)

            amat = np.abs(mat)
            valid = amat > eps
            if density is not None and np.any(valid):
                cutoff = np.quantile(amat[valid], 1 - density)
                mat = np.where(amat > cutoff, mat, 0.0)

            return (mat + mat.T) / 2.0

        assert filename.stem not in matrices
        inputs[filename.stem] = result

    # }}}

    # {{{ compute communities

    coms = None
    if ftype == "ugraph":
        log.info("Computing the shared community template.")
        coms = make_template_communities(
            inputs,
            resolution=community_resolution,
            seed=community_seed,
        )

        if coms is None:
            log.error("Could not compute a community template from the given files.")
            return 1

        log.info("Template has %d communities.", coms.size + 1)

    # }}}

    # {{{ compute features

    datasets = {}

    for name, matrices in inputs.items():
        log.info("Gathering '%s' features for '%s'.", ftype, name)

        nsamples, _, _ = matrices.shape
        result: Array2D | None = None

        for i in range(nsamples):
            mat_i = matrices[i]
            if ftype == "fourier":
                xmin, xmax, ymin, ymax = data["images_bbox"][i]
            else:
                xmin, xmax, ymin, ymax = data["images_bbox"]

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                if ftype == "matrix":
                    features = make_matrix_features(mat_i)
                elif ftype == "eig":
                    features = make_eig_features(mat_i)
                elif ftype == "sgraph":
                    features = make_signed_graph_features(mat_i)
                elif ftype == "ugraph":
                    assert coms is not None
                    features = make_unsigned_graph_features(mat_i, communities=coms)
                elif ftype == "dmd":
                    features = make_dmd_features(mat_i, bbox=(xmin, xmax, ymin, ymax))
                elif ftype == "fourier":
                    features = make_fourier_features(mat_i, (xmin, xmax, ymin, ymax))
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

            assert np.all(np.isfinite(features.x))
            assert features.x.dtype == np.float64
            assert result.shape[1] == nfeatures
            assert features.starts[-1] == nfeatures
            assert np.sum(np.diff(features.starts)) == nfeatures

            result[i] = features.x

        assert filename.stem not in datasets
        datasets[filename.stem] = result

    # }}}

    np.savez(
        outfile,
        datasets=datasets,
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
        choices=("sgraph", "matrix", "dmd", "eig", "ugraph", "fourier"),
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

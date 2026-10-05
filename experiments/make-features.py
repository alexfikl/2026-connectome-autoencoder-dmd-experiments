# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

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


class Features(NamedTuple):
    features: Array1D[np.floating[Any]]
    starts: Array1D[np.integer[Any]]
    descrs: tuple[str, ...]


# {{{ make_matrix_features


def make_matrix_features(mat: Array2D[np.floating[Any]]) -> Features:
    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    rows, cols = np.tril_indices(n, k=-1)

    features = mat[rows, cols]
    starts = np.array([0, features.size])
    descrs = ("Lower triangular matrix entries",)

    return Features(features, starts, descrs)


# }}}


# {{{ make_eig_features


def make_eig_features(
    mat: Array2D[np.floating[Any]],
    *,
    top_k: int = 3,
) -> Features:
    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    if top_k > n:
        raise ValueError(f"'top_k' must be smaller than matrix size: {top_k}")

    eigs, eigvs = np.linalg.eigh(mat)

    # get the largest eigenvalues in magnitude
    order = np.argsort(-np.abs(eigs))[:top_k]
    eigs = eigs[order]
    eigvs = eigvs[:, order]

    # NOTE: fix sign ambiguity
    idx = np.argmax(np.abs(eigvs), axis=0)
    eigvs = eigvs * np.sign(eigvs[idx, np.arange(top_k)])

    features = np.concatenate([
        eigs,
        eigvs.T.ravel(),
    ])

    k = top_k + 1
    starts = np.array([*range(k), *[top_k + i * n for i in range(1, k)]])
    descrs = (
        *(f"Eigenvalue {i}" for i in range(top_k)),
        *(f"Eigenvector {i}" for i in range(top_k)),
    )

    return Features(features, starts, descrs)


# }}}


# {{{ make_signed_graph_features


def _make_signed_graph_features(mat: Array2D[np.floating[Any]]) -> Features:
    import networkx as nx

    # construct graphs
    # NOTE: distance = 1/weight for some measures (shortest path on cost)
    G = nx.from_numpy_array(mat)
    for u, v, d in G.edges(data=True):
        if d["weight"] > 0:
            G[u][v]["distance"] = 1.0 / d["weight"]
    log.info("        Nodes %d Edges %d", len(G.nodes()), len(G.edges()))

    features = []
    starts = [0]
    descrs = []

    # Strength: average weighted degree
    strengths = np.array([d for _, d in G.degree(weight="weight")])
    features.append(np.mean(strengths))
    starts.append(features[-1].size)
    descrs.append("Strength")

    # Clustering Coefficient
    features.append(nx.average_clustering(G, weight="weight"))
    starts.append(features[-1].size)
    descrs.append("Clustering Coefficient")

    # Modularity
    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    features.append(nx.community.modularity(G, communities, weight="weight"))
    starts.append(features[-1].size)
    descrs.append("Modularity")

    # Eigenvector Centrality
    ec = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    features.append(np.mean(list(ec.values())))
    starts.append(features[-1].size)
    descrs.append("Eigenvector Centrality")

    # Rich-club Coefficient (mean across degrees)
    rc = nx.rich_club_coefficient(G, normalized=False)
    features.append(np.mean(list(rc.values())))
    starts.append(features[-1].size)
    descrs.append("Rich-club Coefficient")

    # Transitivity (unweighted)
    features.append(nx.transitivity(G))
    starts.append(features[-1].size)
    descrs.append("Transitivity")

    # Characteristic Path Length
    features.append(nx.average_shortest_path_length(G, weight="distance"))
    starts.append(features[-1].size)
    descrs.append("Characteristic Path Length")

    # Global Efficiency
    features.append(nx.global_efficiency(G))
    starts.append(features[-1].size)
    descrs.append("Global Efficiency")

    # Local Efficiency
    features.append(nx.local_efficiency(G))
    starts.append(features[-1].size)
    descrs.append("Local Efficiency")

    # Betweenness Centrality (mean across nodes)
    bc = nx.betweenness_centrality(G, weight="distance", normalized=True)
    features.append(np.mean(list(bc.values())))
    starts.append(features[-1].size)
    descrs.append("Betweenness Centrality")

    # Small-worldness (requires connected graph)
    features.append(nx.sigma(G, niter=1, nrand=1, seed=42))
    starts.append(features[-1].size)
    descrs.append("Small-worldness")

    return Features(np.array(features), np.array(starts), descrs=tuple(descrs))


def make_signed_graph_features(mat: Array2D[np.floating[Any]]) -> Features:
    return _make_signed_graph_features(np.abs(mat))


# }}}


# {{{ make_unsigned_graph_features


def make_unsigned_graph_features(mat: Array2D[np.floating[Any]]) -> Features:
    import orbitkit.metrics as okm
    from orbitkit.clusters import signed_leiden_communities

    features = []
    starts = [0]
    descrs = []

    # Strength
    features.append(okm.compute_weighted_degree(mat))
    starts.append(features[-1].size)
    descrs.append("Strength")

    # Clustering Coefficient
    features.append(
        okm.compute_weighted_clustering_coefficient_costantini(mat, variant=6)
    )
    starts.append(features[-1].size)
    descrs.append("Clustering Coefficient")

    # Local Assortativity
    features.append(
        okm.compute_local_assortativity_sabek(np.abs(mat), alpha=1.0, beta=1.0)
    )
    starts.append(features[-1].size)
    descrs.append("Local Assortativity")

    # Eigenvector Centrality
    features.append(okm.compute_eigenvector_centrality(mat).score)
    starts.append(features[-1].size)
    descrs.append("Eigenvector Centrality")

    # Modularity
    communities = signed_leiden_communities(mat)
    features.append(np.array(okm.compute_modularity(mat, communities)))
    starts.append(features[-1].size)
    descrs.append("Modularity")

    return Features(np.vstack(features), np.array(starts), descrs=tuple(descrs))


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
) -> Features:
    from commonlib import generate_dataset
    from nneuroutil.dmdc import build_dense_dmdc

    n, m = mat.shape
    if n != m:
        raise ValueError(f"'mat' is not square: {mat.shape}")

    if top_k > n:
        raise ValueError(f"'top_k' must be smaller than matrix size: {top_k}")

    # {{{ generate data

    dtype = np.result_type(mat.dtype, np.complex64)

    Z, c = generate_dataset(
        np.astype(mat, dtype),
        bbox=bbox,
        resolution=resolution,
        escape_radius=escape_radius,
        maxit=maxit,
    )

    # NOTE: only keep the data we actually use from here on out
    mask = np.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, axis=-1) < escape_radius**2
    Z = Z[:, mask, :]

    # }}}

    # {{{ build dataset

    # NOTE: Z has shape (maxit, converged, d)
    X1 = Z[:-1].reshape(-1, n)
    X2 = Z[1:].reshape(-1, n)
    U = np.broadcast_to(c[mask, None], (maxit - 1, mask.sum(), 1)).reshape(-1, 1)

    # }}}

    # {{{ features

    dmd = build_dense_dmdc(X1, U, X2, method="ridge")
    eigs, Phi = np.linalg.eig(dmd.A.T)  # ty: ignore[unresolved-attribute]

    P = np.linalg.pinv(Phi)
    G = X1.conj().T @ X1
    coeff_energy = np.maximum(np.real(np.diag(P @ G @ P.conj().T)), 0.0)
    strengths = np.linalg.norm(Phi, axis=0) * np.sqrt(coeff_energy)

    order = np.argsort(-strengths)[:top_k]
    eigs = eigs[order]
    strengths = strengths[order] / np.sum(strengths)
    Phi = Phi[:, order]

    features = np.concatenate([
        strengths,
        eigs.real,
        eigs.imag,
        np.abs(Phi).T.ravel(),
    ])

    starts = np.array([
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

    return Features(features, starts, descrs)


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
        log.warning("No files give.")
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

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                if ftype == "matrix":
                    features, starts, descrs = make_matrix_features(mat_i)
                elif ftype == "eig":
                    features, starts, descrs = make_eig_features(mat_i)
                elif ftype == "sgraph":
                    features, starts, descrs = make_signed_graph_features(mat_i)
                elif ftype == "ugraph":
                    features, starts, descrs = make_unsigned_graph_features(mat_i)
                elif ftype == "dmd":
                    features, starts, descrs = make_dmd_features(mat_i)
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

            if result is None:
                result = np.full(
                    (nsamples, features.size), np.nan, dtype=features.dtype
                )

            assert result.shape[1] == features.size
            assert starts[-1] == features.size
            assert np.sum(np.diff(starts)) == features.size

            result[i] = features

        assert filename.stem not in datasets
        datasets[filename.stem] = result

    np.savez(outfile, datasets=datasets, starts=starts, descrs=descrs)
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
    parser.add_argument("filenames", nargs="*", type=pathlib.Path)
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

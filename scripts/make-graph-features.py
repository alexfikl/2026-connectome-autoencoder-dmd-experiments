# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import rich.logging
from typing_extensions import TypeAliasType, TypeVar

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Compute graph measures for connectome matrices.

Example:

    > {SCRIPT_PATH.name} --variable-name matrices --outfile result.npz data.npz
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


MEASURE_INDEX = {
    "Strength": 0,
    "Clustering Coefficient": 1,
    "Modularity": 2,
    "Participation Coefficient": 3,
    "Gateway Coefficient": 4,
    "Assortativity": 5,
    "Diversity Coefficient": 6,
    "Eigenvector Centrality": 7,
    "Rich-club Coefficient": 8,
    "Transitivity": 9,
    "Characteristic Path Length": 10,
    "Global Efficiency": 11,
    "Local Efficiency": 12,
    "Betweenness Centrality": 13,
    "Small-worldness": 14,
}
assert len(set(MEASURE_INDEX.keys())) - len(set(MEASURE_INDEX.values())) == 0, (
    "Indices do not match"
)


# {{{ compute_graph_measures


def compute_graph_measures(mat: Array2D[np.floating]) -> Array1D[np.floating]:
    import networkx as nx

    # construct graphs
    # NOTE: distance = 1/weight for some measures (shortest path on cost)
    G = nx.from_numpy_array(np.abs(mat))
    for u, v, d in G.edges(data=True):
        if d["weight"] > 0:
            G[u][v]["distance"] = 1.0 / d["weight"]

    nfeatures = len(MEASURE_INDEX)
    result = np.zeros(nfeatures, dtype=mat.dtype)

    # Strength: average weighted degree
    strengths = np.array([d for _, d in G.degree(weight="weight")])
    result[MEASURE_INDEX["Strength"]] = np.mean(strengths)

    # Clustering Coefficient
    result[MEASURE_INDEX["Clustering Coefficient"]] = nx.average_clustering(
        G, weight="weight"
    )

    # Modularity
    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    result[MEASURE_INDEX["Modularity"]] = nx.community.modularity(
        G, communities, weight="weight"
    )

    # Participation Coefficient - not implemented
    result[MEASURE_INDEX["Participation Coefficient"]] = 0.0

    # Gateway Coefficient - not implemented
    result[MEASURE_INDEX["Gateway Coefficient"]] = 0.0

    result[MEASURE_INDEX["Assortativity"]] = nx.degree_assortativity_coefficient(
        G, weight="weight"
    )

    # Diversity Coefficient - not implemented
    result[MEASURE_INDEX["Diversity Coefficient"]] = 0.0

    # Eigenvector Centrality
    ec = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    result[MEASURE_INDEX["Eigenvector Centrality"]] = np.mean(list(ec.values()))

    # Rich-club Coefficient (mean across degrees)
    rc = nx.rich_club_coefficient(G, normalized=False)
    result[MEASURE_INDEX["Rich-club Coefficient"]] = np.mean(list(rc.values()))

    # Transitivity (unweighted)
    result[MEASURE_INDEX["Transitivity"]] = nx.transitivity(G)

    # Characteristic Path Length
    result[MEASURE_INDEX["Characteristic Path Length"]] = (
        nx.average_shortest_path_length(G, weight="distance")
    )

    # Global Efficiency
    result[MEASURE_INDEX["Global Efficiency"]] = nx.global_efficiency(G)

    # Local Efficiency
    result[MEASURE_INDEX["Local Efficiency"]] = nx.local_efficiency(G)

    # Betweenness Centrality (mean across nodes)
    bc = nx.betweenness_centrality(G, weight="distance", normalized=True)
    result[MEASURE_INDEX["Betweenness Centrality"]] = np.mean(list(bc.values()))

    # Small-worldness (requires connected graph)
    result[MEASURE_INDEX["Small-worldness"]] = nx.sigma(G, niter=1, nrand=1, seed=42)

    return result


# }}}


# {{{ make_graph_features


def make_graph_features(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    key: str = "matrices",
    eps: float | None = None,
    density: float | None = None,
    overwrite: bool = False,
) -> int:
    if not filename.exists():
        log.error("Input file does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_stem(f"{filename.stem}-graph-features")

    if not overwrite and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
        return 1

    try:
        data = np.load(filename)
    except Exception:
        log.error("Failed to load data from file: '%s'.", filename)
        return 1

    if key not in data:
        log.error("File does not contain '%s' dataset: '%s'.", key, filename)
        return 1

    matrices = data[key]
    if matrices.ndim != 3:
        log.error("Expected a 3 dimensional array: %s.", matrices.shape)
        return 1

    if eps is None:
        eps = 10.0 * np.finfo(matrices.dtype).eps

    nsamples = matrices.shape[0]
    nfeatures = len(MEASURE_INDEX)

    datasets = np.zeros((3, nsamples, nfeatures), dtype=np.float64)
    for i, mat in enumerate(matrices):
        # ensure no self-connections
        mat_i = mat.copy()
        np.fill_diagonal(mat_i, 0.0)

        # thresholding
        amat_i = np.abs(mat_i)
        if density is not None:
            cutoff = np.quantile(amat_i[amat_i > eps], 1 - density)
            mat_i = np.where(amat_i > cutoff, mat_i, 0.0)

        # absolute value
        datasets[0, i] = compute_graph_measures(amat_i)

        # positive only
        amat_i = np.where(mat_i > 0.0, mat_i, 0.0)
        datasets[1, i] = compute_graph_measures(amat_i)

        # negative only
        amat_i = np.where(mat_i < 0.0, -mat_i, 0.0)
        datasets[2, i] = compute_graph_measures(amat_i)

    np.savez(outfile, datasets=datasets, descr=MEASURE_INDEX)
    log.info("Written graph measures to file: '%s'.", outfile)

    return 0


# }}}


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
        "--key",
        default="matrices",
        help="Name of the variable in the file to use",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
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
        make_graph_features(
            args.filename,
            key=args.key,
            density=args.density,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

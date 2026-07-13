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
    # signed
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
    # unsigned - positive
    "Characteristic Path Length (P)": 10,
    "Global Efficiency (P)": 11,
    "Local Efficiency (P)": 12,
    "Betweenness Centrality (P)": 13,
    "Small-worldness (P)": 14,
    # unsigned - negative
    "Characteristic Path Length (N)": 15,
    "Global Efficiency (N)": 16,
    "Local Efficiency (N)": 17,
    "Betweenness Centrality (N)": 18,
    "Small-worldness (N)": 19,
}
assert len(set(MEASURE_INDEX.keys())) - len(set(MEASURE_INDEX.values())) == 0, (
    "Indices do not match"
)


# {{{ compute_graph_measures


def compute_graph_measures(
    mat: Array2D[np.floating],
    *,
    percentile: float = 0.95,
    eps: float | None = None,
) -> Array1D[np.floating]:
    import networkx as nx

    if eps is None:
        eps = 10.0 * np.finfo(mat.dtype).eps

    # ensure there are no self-connections
    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)

    # remove noise
    amat = np.abs(mat)
    cutoff = np.quantile(amat[amat > eps], percentile)
    mat = np.where(amat > cutoff, mat, 0.0)

    # construct graphs
    G_all = nx.from_numpy_array(np.abs(mat))
    G_pos = nx.from_numpy_array(np.where(mat > 0, mat, 0.0))
    G_neg = nx.from_numpy_array(np.where(mat < 0, -mat, 0.0))

    # distance = 1/weight for some measures (shortest path on cost)
    for G in [G_all, G_pos, G_neg]:
        for u, v, d in G.edges(data=True):
            if d["weight"] > 0:
                G[u][v]["distance"] = 1.0 / d["weight"]

    nfeatures = len(MEASURE_INDEX)
    result = np.zeros(nfeatures, dtype=mat.dtype)

    # {{{ signed measures (G_all - absolute weights)

    # Strength: average weighted degree
    strengths = np.array([d for _, d in G_all.degree(weight="weight")])
    result[MEASURE_INDEX["Strength"]] = np.mean(strengths)

    # Clustering Coefficient
    result[MEASURE_INDEX["Clustering Coefficient"]] = nx.average_clustering(
        G_all, weight="weight"
    )

    # Modularity
    communities = nx.community.greedy_modularity_communities(G_all, weight="weight")
    result[MEASURE_INDEX["Modularity"]] = nx.community.modularity(
        G_all, communities, weight="weight"
    )

    # Participation Coefficient - not implemented
    result[MEASURE_INDEX["Participation Coefficient"]] = 0.0

    # Gateway Coefficient - not implemented
    result[MEASURE_INDEX["Gateway Coefficient"]] = 0.0

    result[MEASURE_INDEX["Assortativity"]] = nx.degree_assortativity_coefficient(
        G_all, weight="weight"
    )

    # Diversity Coefficient - not implemented
    result[MEASURE_INDEX["Diversity Coefficient"]] = 0.0

    # Eigenvector Centrality
    ec = nx.eigenvector_centrality(G_all, weight="weight", max_iter=1000)
    result[MEASURE_INDEX["Eigenvector Centrality"]] = np.mean(list(ec.values()))

    # Rich-club Coefficient (mean across degrees)
    rc = nx.rich_club_coefficient(G_all, normalized=True, seed=42, weight="weight")
    result[MEASURE_INDEX["Rich-club Coefficient"]] = np.mean(list(rc.values()))

    # Transitivity (unweighted)
    result[MEASURE_INDEX["Transitivity"]] = nx.transitivity(G_all)

    # }}}

    # {{{ unsigned positive measures (G_pos)

    if G_pos.number_of_edges() > 0:
        result[MEASURE_INDEX["Characteristic Path Length (P)"]] = (
            nx.average_shortest_path_length(G_pos, weight="distance")
        )

        result[MEASURE_INDEX["Global Efficiency (P)"]] = nx.global_efficiency(
            G_pos, weight="distance"
        )
        result[MEASURE_INDEX["Local Efficiency (P)"]] = nx.local_efficiency(
            G_pos, weight="distance"
        )

        bc = nx.betweenness_centrality(G_pos, weight="distance", normalized=True)
        result[MEASURE_INDEX["Betweenness Centrality (P)"]] = np.mean(list(bc.values()))

        result[MEASURE_INDEX["Small-worldness (P)"]] = nx.sigma(
            G_pos, niter=100, nrand=10, seed=42
        )
    else:
        log.error("Graph has no positive edges")

    # }}}

    # {{{ unsigned negative measures (G_neg)

    if G_neg.number_of_edges() > 0:
        result[MEASURE_INDEX["Characteristic Path Length (N)"]] = (
            nx.average_shortest_path_length(G_neg, weight="distance")
        )

        result[MEASURE_INDEX["Global Efficiency (N)"]] = nx.global_efficiency(
            G_neg, weight="distance"
        )
        result[MEASURE_INDEX["Local Efficiency (N)"]] = nx.local_efficiency(
            G_neg, weight="distance"
        )

        bc = nx.betweenness_centrality(G_neg, weight="distance", normalized=True)
        result[MEASURE_INDEX["Betweenness Centrality (N)"]] = np.mean(list(bc.values()))

        result[MEASURE_INDEX["Small-worldness (N)"]] = nx.sigma(
            G_neg, niter=100, nrand=10, seed=42
        )
    else:
        log.error("Graph has no negative edges")

    # }}}

    return result


# }}}


# {{{ make_graph_features


def make_graph_features(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    key: str = "matrices",
    percentile: float = 0.95,
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

    nsamples = matrices.shape[0]
    nfeatures = len(MEASURE_INDEX)

    datasets = np.zeros((nsamples, nfeatures), dtype=np.float64)
    for i, mat in enumerate(matrices):
        datasets[i] = compute_graph_measures(mat)

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
        "--percentile",
        type=float,
        default=0.95,
        help="Remove values that are in the percentile",
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
            percentile=args.percentile,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

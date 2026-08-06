# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import numpy as np

from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array1D

log = module_logger("lib.graph_measures")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Compute common graph measures for fMRI matrices.

Example:

    > {SCRIPT_PATH.name} filename.npz
"""

NAME_TO_LABEL = {
    "avg_strength": r"$\langle A \rangle$",
    # "avg_clustering": r"$\langle C \rangle$",
    "transitivity": "T",
    "global_efficiency": "GE",
    # "local_efficiency": "LE",
    "avg_betweenness": r"$\langle BC \rangle$",
    "avg_eigenvector": r"$\langle EC \rangle$",
    "assortativity": "AC",
    # "modularity": "M",
    # "algebraic_connectivity": "AC",
    # "dmd_mode_1": r"$|b_1|$",
    "dmd_mode_2": r"$|b_2|$",
    "dmd_mode_3": r"$|b_3|$",
    "dmd_mode_4": r"$|b_4|$",
    # "dmd_eig_1": r"$|\lambda_1|$",
    "dmd_eig_2": r"$|\lambda_2|$",
    "dmd_eig_3": r"$|\lambda_3|$",
    "dmd_eig_4": r"$|\lambda_4|$",
}


def dmd_optimal_mode_strengths(
    mat: np.ndarray,
    *,
    n_modes: int = 4,
    maxit: int = 64,
    escape_radius: float = 100.0,
    resolution: int = 32,
) -> dict[str, np.ndarray]:
    """Compute optimal DMD mode amplitudes and eigenvalue magnitudes.

    Evolves the quadratic iteration z <- (A z)^2 + c over a grid of initial
    conditions (matching dmd-reconstruction.py), builds the X1/X2 snapshot
    pairs, fits a DMD model, then solves for optimal mode amplitudes by least
    squares over all snapshots (Jovanovic et al. 2014).

    Returns a dict with keys ``"modes"`` (optimal amplitude magnitudes |b_k|)
    and ``"eigs"`` (eigenvalue magnitudes |lambda_k|), both sorted by
    descending |eigenvalue| and of length *n_modes*.
    """
    from commonlib import generate_dataset

    d = mat.shape[0]
    bbox = (-0.025, 0.025, -0.025, 0.025)

    Z, c = generate_dataset(
        mat,
        bbox=bbox,
        resolution=resolution,
        escape_radius=escape_radius,
        maxit=maxit,
    )

    # mask out points that escaped
    mask = np.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, axis=-1) < escape_radius**2

    # build snapshot matrix: append c as extra variable, keep only converged points
    C = np.tile(c[None, :, :, None], (maxit, 1, 1, 1))
    X = np.concatenate([Z, C], axis=3)
    X = X[:, mask, :]
    X = X.reshape(maxit, -1)

    X1 = X[:-1].reshape(-1, d + 1).T
    X2 = X[1:].reshape(-1, d + 1).T

    from nneuroutil.dmd import build_dmd

    model = build_dmd(X1, X2, eps=1.0e-6)

    # eigendecomposition of the reduced propagator; sort by descending |eigenvalue|
    eigenvalues, eigenvectors = np.linalg.eig(model.Ahat)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # project all X1 snapshots into the reduced space: shape (rank, n_pairs)
    # X1 has shape (d+1, n_pairs), U has shape (d+1, rank)
    Xhat = np.conj(model.U).T @ X1

    # optimal amplitude via Jovanovic normal equations:
    #   P = (Phi^H Phi) o conj(V V^H),  q = diag(V Xhat^H Phi),  b = P^{-1} q
    #
    # The Vandermonde V[k, j] = lambda_k^{t(j)} where t(j) is the time index of
    # snapshot pair j. Pairs are stacked as (maxit-1) blocks of n_converged points,
    # so t(j) = j // n_converged.
    n_converged = int(mask.sum())
    t = np.repeat(np.arange(maxit - 1), n_converged)  # shape (n_pairs,)
    n_modes_avail = len(eigenvalues)
    Vdmd = eigenvalues[:, None] ** t[None, :]  # (n_modes_avail, n_pairs)

    PhiH_Phi = eigenvectors.conj().T @ eigenvectors
    Vdmd_VdmdH = Vdmd @ Vdmd.conj().T
    P = PhiH_Phi * Vdmd_VdmdH.conj()
    q = np.diag(Vdmd @ Xhat.conj().T @ eigenvectors)

    b = np.linalg.solve(P, q)

    n_out = min(n_modes, n_modes_avail)

    def _pad(arr: np.ndarray) -> np.ndarray:
        if n_out < n_modes:
            arr = np.pad(arr, (0, n_modes - n_out))
        return arr.astype(mat.dtype)

    return {
        "modes": _pad(np.abs(b[:n_out])),
        "eigs": _pad(np.abs(eigenvalues[:n_out])),
    }


def compute_class_graph_measures(
    filename: pathlib.Path,
    *,
    datakey: str = "matrices",
    tol: float = 0.05,
) -> dict[str, Array1D]:
    # {{{ vallidate inputs

    if not filename.exists():
        raise FileNotFoundError(f"file does not exist: '{filename}'.")

    try:
        data = np.load(filename)
    except Exception:
        raise RuntimeError(f"failed to load data from file: '{filename}'.") from None

    if datakey not in data:
        raise ValueError(f"file does not contain '{datakey}' dataset: '{filename}'.")

    matrices = data[datakey]
    if matrices.ndim != 3:
        raise ValueError(f"expected a 3 dimensional array: {matrices.shape}.")
        return 1

    log.info("Graph measures: '%s'.", filename)

    # }}}

    # {{{ compute graph measures for each matrix

    import networkx as nx

    nsamples, rx, ry = matrices.shape
    assert rx == ry

    avg_strength = np.empty(nsamples, dtype=matrices.dtype)
    avg_clustering = np.empty(nsamples, dtype=matrices.dtype)
    transitivity = np.empty(nsamples, dtype=matrices.dtype)
    global_efficiency = np.empty(nsamples, dtype=matrices.dtype)
    local_efficiency = np.empty(nsamples, dtype=matrices.dtype)
    avg_betweenness = np.empty(nsamples, dtype=matrices.dtype)
    avg_eigenvector = np.empty(nsamples, dtype=matrices.dtype)
    assortativity = np.empty(nsamples, dtype=matrices.dtype)
    modularity = np.empty(nsamples, dtype=matrices.dtype)
    algebraic_connectivity = np.empty(nsamples, dtype=matrices.dtype)
    dmd_modes = np.empty((nsamples, 4), dtype=matrices.dtype)
    dmd_eigs = np.empty((nsamples, 4), dtype=matrices.dtype)

    for i in range(nsamples):
        # NOTE: this is a choice, maybe not the best one, but we need an unsigned
        # matrix for most of these measures.
        mat = np.abs(matrices[i])
        log.info("%s: matrix %d / %d", filename, i + 1, nsamples)

        # DMD evolves the quadratic iteration z <- (A z)^2 + c driven by this
        # matrix; the snapshots are generated internally by generate_dataset
        dmd_result = dmd_optimal_mode_strengths(matrices[i], n_modes=4)
        dmd_modes[i] = dmd_result["modes"]
        dmd_eigs[i] = dmd_result["eigs"]

        # clean up the matrices a bit to avoid noise
        np.fill_diagonal(mat, 0.0)
        cutoff = np.quantile(mat[mat > 0.0], 1 - tol)
        mat = np.where(mat > cutoff, mat, 0.0)

        # construct graph
        G = nx.from_numpy_array(mat)
        for u, v, d in G.edges(data=True):
            G[u][v]["distance"] = 1.0 - d["weight"]

        # 1. average strength
        strengths = np.array([d for _, d in G.degree(weight="weight")])
        avg_strength[i] = np.mean(strengths)

        # 2. weighted clustering coefficient (Onnela et al.; weight = strength)
        avg_clustering[i] = nx.average_clustering(G, weight="weight")

        # 3. transitivity (structural only — nx.transitivity has no weight param)
        transitivity[i] = nx.transitivity(G)

        # 4. global efficiency (nx ignores weights; hop-count paths only)
        global_efficiency[i] = nx.global_efficiency(G)

        # 5. local efficiency (nx ignores weights; hop-count paths only)
        local_efficiency[i] = nx.local_efficiency(G)

        # 6. betweenness centrality (weight = distance/cost, so use "distance" attr)
        bc = nx.betweenness_centrality(G, weight="distance", normalized=True)
        avg_betweenness[i] = np.mean(list(bc.values()))

        # 7. eigenvector centrality (weight = strength)
        ec = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        avg_eigenvector[i] = np.mean(list(ec.values()))

        # 8. degree assortativity (weight = strength, i.e. weighted degree)
        assortativity[i] = nx.degree_assortativity_coefficient(G, weight="weight")

        # 9. modularity (detect communities with greedy algorithm, then score)
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        modularity[i] = nx.community.modularity(G, communities, weight="weight")

        # 10. algebraic connectivity — second eigenvalue of weighted Laplacian
        algebraic_connectivity[i] = nx.algebraic_connectivity(G, weight="weight")

    return {
        "avg_strength": avg_strength,
        "avg_clustering": avg_clustering,
        "transitivity": transitivity,
        "global_efficiency": global_efficiency,
        "local_efficiency": local_efficiency,
        "avg_betweenness": avg_betweenness,
        "avg_eigenvector": avg_eigenvector,
        "assortativity": assortativity,
        "modularity": modularity,
        # "algebraic_connectivity": algebraic_connectivity,
        "dmd_mode_1": dmd_modes[:, 0],
        "dmd_mode_2": dmd_modes[:, 1],
        "dmd_mode_3": dmd_modes[:, 2],
        "dmd_mode_4": dmd_modes[:, 3],
        "dmd_eig_1": dmd_eigs[:, 0],
        "dmd_eig_2": dmd_eigs[:, 1],
        "dmd_eig_3": dmd_eigs[:, 2],
        "dmd_eig_4": dmd_eigs[:, 3],
    }


def main(
    filenames: list[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    tol: float = 0.05,
    overwrite: bool = False,
) -> int:
    if len(filenames) <= 1:
        log.error("At least two files are required")
        return 1

    if outfile is None:
        outfile = filenames[0].parent / "graph-measures.npz"

    basename = outfile.with_suffix("")

    # {{{ compute measures

    if not overwrite and outfile.exists():
        data = np.load(outfile, allow_pickle=True)
        datasets = data["datasets"][()]
    else:
        datasets = {}
        for filename in filenames:
            # try:
            datasets[filename.stem] = compute_class_graph_measures(
                filename,
                datakey=datakey,
                tol=tol,
            )
            # except Exception as exc:
            #     log.error("Failed to compute graph measures: '%s'.", filename)
            #     log.error("Error: %s", exc)
            #     return 1

        np.savez(outfile, datasets=datasets)

    # }}}

    # {{{ build stacked arrays and class labels

    available = set(next(iter(datasets.values())).keys())
    measure_names = [m for m in NAME_TO_LABEL if m in available]
    class_names = list(datasets.keys())
    nmeasures = len(measure_names)

    # stack all classes into (nsamples_total,) arrays; labels are integer class indices
    stacked: dict[str, np.ndarray] = {
        m: np.concatenate([datasets[c][m] for c in class_names]) for m in measure_names
    }
    labels = np.concatenate([
        np.full(len(datasets[c][measure_names[0]]), i)
        for i, c in enumerate(class_names)
    ])

    # feature matrix shape (nsamples_total, nmeasures)
    X = np.column_stack([stacked[m] for m in measure_names])

    from scipy.stats import entropy as scipy_entropy
    from sklearn.feature_selection import mutual_info_classif

    mi = np.mean(
        [mutual_info_classif(X, labels, random_state=i) for i in range(32)], axis=0
    )

    # normalise by label entropy so NMI ∈ [0, 1]: fraction of label uncertainty resolved
    _, class_counts = np.unique(labels, return_counts=True)
    h_labels = scipy_entropy(class_counts)  # in nats, same units as mutual_info_classif
    nmi = mi / h_labels

    from scipy.stats import spearmanr

    corr_result = spearmanr(X)
    corr = np.array(corr_result.statistic)

    # eta squared: SS_between / SS_total per measure (one-way ANOVA effect size)
    eta2 = np.empty(nmeasures)
    for j, m in enumerate(measure_names):
        x = stacked[m]
        grand_mean = np.mean(x)
        ss_total = np.sum((x - grand_mean) ** 2)
        ss_between = sum(
            len(datasets[c][m]) * (np.mean(datasets[c][m]) - grand_mean) ** 2
            for c in class_names
        )
        eta2[j] = ss_between / ss_total if ss_total > 0.0 else 0.0

    # }}}

    # {{{ mutual information bar plot

    from commonlib import set_matplotlib_style  # ruff:ignore[unused-import]
    from nneuroutil.visualization import figure, make_colorbar_axes

    pretty_names = [NAME_TO_LABEL[name] for name in measure_names]
    with figure(basename.with_stem(f"{basename.stem}-mi")) as fig:
        ax = fig.gca()
        ax.grid(visible=False, which="both")

        ax.bar(np.arange(nmeasures), nmi, width=0.9)
        ax.set_xticks(np.arange(nmeasures))
        ax.set_xticklabels(pretty_names, rotation=45, ha="right", fontsize=16)
        ax.set_ylabel("NMI")

    # }}}

    # {{{ eta squared bar plot

    with figure(basename.with_stem(f"{basename.stem}-eta2")) as fig:
        ax = fig.gca()
        ax.grid(visible=False, which="both")

        ax.bar(np.arange(nmeasures), eta2, width=0.9)
        ax.set_xticks(np.arange(nmeasures))
        ax.set_xticklabels(pretty_names, rotation=45, ha="right", fontsize=16)
        ax.set_ylabel(r"$\eta^2$")

    # }}}

    # {{{ spearman correlation matrix

    with figure(basename.with_stem(f"{basename.stem}-spearman")) as fig:
        ax = fig.gca()
        ax.grid(visible=False, which="both")

        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="seismic")
        ax.set_xticks(np.arange(nmeasures))
        ax.set_yticks(np.arange(nmeasures))
        ax.set_xticklabels(pretty_names, rotation=45, ha="right", fontsize=16)
        ax.set_yticklabels(pretty_names, fontsize=16)

        cax = make_colorbar_axes(ax)
        fig.colorbar(im, cax=cax)

    # }}}

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
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
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
            args.filenames,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

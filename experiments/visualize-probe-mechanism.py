# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import math
import pathlib
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple, get_args

import numpy as np
import torch

import commonlib  # ruff: ignore[unused-import]
from classifylib import (
    ClassificationMethod,
    ClassificationResult,
    FeatureDataset,
    RegressionResult,
    classify_fingerprinting,
    classify_task_decoding,
    flatten_feature_datasets,
    load_feature_datasets,
    make_subject_index,
    run_ridge_cv,
    select_feature_group,
)
from nneuroutil.helpers import FuzzyChoiceAction
from nneuroutil.interop.torch import available_device_names, torch_default_device
from nneuroutil.typing import Array1D, Array3D
from nneuroutil.visualization import figure

log = logging.getLogger("lib.visualize_probe_mechanism")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Analyze the probe-subspace mechanism for Autoencoder-DMD (Family I) on
functional connectomes.

Evaluates two questions:
1. Which block of Family I carries 7-way task and fingerprinting signal.
2. How the Family I blocks relate to the probe-subspace vectors (A1, (A1)^2,
   A(A1)^2) vs. eigenvectors orthogonal to 1, and to scalar moments of A1.

Example:

    > {SCRIPT_PATH.name} --device cuda \\
        data/dataset-features-ae-dmd.npz \\
        data/dataset-features-matrix.npz
"""


# {{{ helpers

BLOCK_SHORT_NAMES: dict[str, str] = {
    "Leading eigenvalue moduli": "Eigenvalues",
    "Control sensitivity (log)": "Control",
    "Attractor node magnitudes": "Magnitudes",
    "Attractor energy": "Energy",
    "Full AE-DMD": "Full",
}

DEGREE_TARGETS: dict[str, str] = {
    "||A1||_2": r"$\norm{\matr{A}\vect{1}}_2$",
    "std(A1)": r"$\mathrm{std}(\matr{A}\vect{1})$",
    "1^T A 1": r"$\vect{1}^T \matr{A} \vect{1}$",
}


def block_selections(ds: FeatureDataset) -> list[tuple[str, int | None]]:
    """Return the named feature blocks used by the mechanism analysis."""
    return [
        *((descr, i) for i, descr in enumerate(ds.descrs)),
        ("Full AE-DMD", None),
    ]


def iter_blocks(ds: FeatureDataset) -> Iterator[tuple[str, FeatureDataset]]:
    """Yield each named feature block and its dataset view."""
    for name, idx in block_selections(ds):
        yield name, (ds if idx is None else select_feature_group(ds, idx))


def reconstruct_matrices(ds: FeatureDataset) -> Array3D[np.floating[Any]]:
    """Reconstruct symmetric zero-diagonal matrices from lower-triangular features."""
    # NOTE: it's easier to take the matrix feature file instead of all the
    # separate files, so we do a little dance here to reconstruct the matrices

    x, _, _ = flatten_feature_datasets(ds)
    m = x.shape[1]
    n = (1 + math.isqrt(1 + 8 * m)) // 2
    if n * (n - 1) // 2 != m:
        raise ValueError(f"feature count {m} is not a triangular number n*(n-1)/2")

    rows, cols = np.tril_indices(n, k=-1)
    mats = np.zeros((x.shape[0], n, n), dtype=np.float64)
    mats[:, rows, cols] = x
    mats[:, cols, rows] = x

    return mats


# }}}


# {{{ block-wise classification


class BlockScores(NamedTuple):
    name: str
    nfeatures: int
    task: tuple[ClassificationResult, ...]
    fp: tuple[ClassificationResult, ...]


def evaluate_blocks(
    ds_ae: FeatureDataset,
    *,
    methods: Sequence[ClassificationMethod] = ("mlp",),
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[BlockScores]:
    """Evaluate task decoding and fingerprinting per Family I block."""
    if not methods:
        raise ValueError("at least one classification method is required")

    results: list[BlockScores] = []
    for name, sub in iter_blocks(ds_ae):
        nfeatures = int(sub.x[0].shape[1])
        x_task, y_task, subject_ids = flatten_feature_datasets(sub)
        x_fp = sub.to_tensor()

        log.info("Evaluating block '%s' (%d features)...", name, nfeatures)
        tasks = []
        fps = []
        for method in methods:
            result, _ = classify_task_decoding(
                x_task,
                y_task,
                subject_ids,
                label=name,
                method=method,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                device=device,
            )
            tasks.append(result)

            result, _ = classify_fingerprinting(
                x_fp,
                label=name,
                method=method,
                seed=seed,
                device=device,
            )
            fps.append(result)

        results.append(
            BlockScores(
                name=name,
                nfeatures=nfeatures,
                task=tuple(tasks),
                fp=tuple(fps),
            )
        )

    return results


# }}}


# {{{ compute_node_correlations


class NodeCorrelations(NamedTuple):
    r_v32: Array1D[np.floating[Any]]
    r_v33: Array1D[np.floating[Any]]
    r_v34: Array1D[np.floating[Any]]
    r_orth_med: Array1D[np.floating[Any]]
    r_orth_max: Array1D[np.floating[Any]]


def pearsonr(u: Array1D[np.floating[Any]], v: Array1D[np.floating[Any]]) -> float:
    # NOTE: scipy.stats.pearsonr is a bit slow here, so we implemented it ourselves
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)

    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    if denom < 1.0e-15:
        return 0.0

    return np.dot(u_c, v_c) / denom


def compute_node_correlations(
    ds_ae: FeatureDataset,
    mats: Array3D[np.floating[Any]],
) -> NodeCorrelations:
    """Compute per-connectome correlations between |z_N| and P(A) vs u_k."""
    idx = ds_ae.descrs.index("Attractor node magnitudes")

    ds_mag = select_feature_group(ds_ae, idx)
    mags, _, _ = flatten_feature_datasets(ds_mag)
    if mats.shape[:2] != mags.shape[:2]:
        raise ValueError(
            f"shape mismatch between matrices {mats.shape} and magnitudes {mags.shape}"
        )

    total, n = mags.shape
    ones = np.ones(n, dtype=mats.dtype)
    ones_unit = ones / np.sqrt(n)

    r_v32 = np.empty(total, dtype=np.float64)
    r_v33 = np.empty(total, dtype=np.float64)
    r_v34 = np.empty(total, dtype=np.float64)
    r_orth_med = np.empty(total, dtype=np.float64)
    r_orth_max = np.empty(total, dtype=np.float64)

    for i, (mat, mag) in enumerate(zip(mats, mags, strict=True)):
        d = mat @ ones

        # NOTE: these are the first 3 vectors from P(A) in Proposition 3
        v32 = d * d
        ad2 = mat @ v32
        v33 = 2.0 * d * ad2
        v34 = ad2 * ad2

        r_v32[i] = pearsonr(mag, v32)
        r_v33[i] = pearsonr(mag, np.abs(v33))
        r_v34[i] = pearsonr(mag, v34)

        _, eigvecs = np.linalg.eigh(mat)
        proj = np.abs(eigvecs.T @ ones_unit)
        orth_idx = np.flatnonzero(proj < 0.05)
        if orth_idx.size == 0:
            raise ValueError(
                f"no eigenvector of connectome {i} is orthogonal to 1: "
                f"min |u_k^T 1| / sqrt(d) = {np.min(proj):.4f}"
            )

        # NOTE: compute the absolute correlation with the eigenvectors of *mat*
        # that are orthogonal to 1 (anti-correlated is fine here, still a line)
        orth_rs = np.array([
            abs(pearsonr(mag, np.abs(eigvecs[:, k]))) for k in orth_idx
        ])
        r_orth_med[i] = np.median(orth_rs)
        r_orth_max[i] = np.max(orth_rs)

    return NodeCorrelations(
        r_v32=r_v32,
        r_v33=r_v33,
        r_v34=r_v34,
        r_orth_med=r_orth_med,
        r_orth_max=r_orth_max,
    )


# }}}


# {{{ compute_degree_moment_r2


def compute_degree_moment_r2(
    ds_ae: FeatureDataset,
    mats: Array3D[np.floating[Any]],
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> list[RegressionResult]:
    """Predict functional scalar moments of A1 from Family I blocks via RidgeCV."""
    d = np.sum(mats, axis=2)
    targets = {
        "||A1||_2": np.linalg.norm(d, axis=1),
        "std(A1)": np.std(d, axis=1),
        "1^T A 1": np.sum(d, axis=1),
    }

    _, _, subject_ids = flatten_feature_datasets(ds_ae)
    subject_index = make_subject_index(subject_ids)

    results: list[RegressionResult] = []
    for bname, sub in iter_blocks(ds_ae):
        x, _, _ = flatten_feature_datasets(sub)
        for tname in DEGREE_TARGETS:
            res = run_ridge_cv(
                x,
                targets[tname],
                subject_index,
                name=bname,
                target_name=tname,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
            )
            results.append(res)

    return results


# }}}


# {{{ reporting & plotting


def _select_result(
    results: Sequence[ClassificationResult],
    method: ClassificationMethod,
) -> ClassificationResult:
    for res in results:
        if res.method == method:
            return res

    raise ValueError(f"method {method!r} was not evaluated")


def _fmt_score(res: ClassificationResult) -> str:
    return (
        f"${res.mean_bacc:.3f} \\pm {res.std_bacc:.3f}$ & "
        f"$[{res.ci_lo:.3f}, {res.ci_hi:.3f}]$"
    )


def _med_iqr(arr: Array1D[np.floating[Any]]) -> tuple[float, float, float]:
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return float(q50), float(q25), float(q75)


def log_summary(
    blocks: Sequence[BlockScores],
    corrs: NodeCorrelations,
    r2: Sequence[RegressionResult],
    *,
    method: ClassificationMethod = "mlp",
) -> None:
    method_label = method.upper()
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"\textbf{AE-DMD block} & \textbf{Size}"
            rf" & \textbf{{Task ({method_label})}} & \textbf{{95\% CI}}"
            rf" & \textbf{{FP ({method_label})}} & \textbf{{95\% CI}} \\"
        ),
        r"\midrule",
    ]
    for b in blocks:
        task = _select_result(b.task, method)
        fp = _select_result(b.fp, method)
        lines.append(
            f"{b.name} & {b.nfeatures:,} & {_fmt_score(task)} & {_fmt_score(fp)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    log.info("Family I block-wise table:\n%s", "\n".join(lines))

    log.info(
        "Functional R^200 Pearson r with |z_N| (median [IQR]): "
        "|v_3,2|=%.3f [%.3f, %.3f], |v_3,3|=%.3f [%.3f, %.3f], "
        "|v_3,4|=%.3f [%.3f, %.3f], orth med=%.3f [%.3f, %.3f], "
        "orth max=%.3f [%.3f, %.3f]",
        *_med_iqr(corrs.r_v32),
        *_med_iqr(corrs.r_v33),
        *_med_iqr(corrs.r_v34),
        *_med_iqr(corrs.r_orth_med),
        *_med_iqr(corrs.r_orth_max),
    )

    size_by_block = {b.name: b.nfeatures for b in blocks}
    by_block: dict[str, dict[str, RegressionResult]] = {}
    for entry in r2:
        by_block.setdefault(entry.name, {})[entry.target_name] = entry

    targets = list(DEGREE_TARGETS)
    r2_lines = [
        r"\begin{tabular}{@{}lr" + "r" * len(targets) + "@{}}",
        r"\toprule",
        " & ".join(
            [r"\textbf{AE-DMD block}", r"\textbf{Size}"]
            + [rf"\textbf{{{DEGREE_TARGETS[t]}}}" for t in targets]
        )
        + r" \\",
        r"\midrule",
    ]
    block_names = list(by_block)
    for i, bname in enumerate(block_names):
        entries = by_block[bname]
        vals = " & ".join(f"${entries[t].r2_mean:.3f}$" for t in targets)
        cis = " & ".join(
            f"$[{entries[t].r2_ci_lo:.3f}, {entries[t].r2_ci_hi:.3f}]$" for t in targets
        )
        r2_lines.append(f"{bname} & {size_by_block[bname]:,} & {vals} \\\\")
        r2_lines.append(f" & & {cis} \\\\")
        if i + 1 < len(block_names):
            r2_lines.append(r"\addlinespace[2pt]")
    r2_lines.extend([r"\bottomrule", r"\end{tabular}"])
    log.info("Degree-moment Ridge R^2 table:\n%s", "\n".join(r2_lines))


def plot_mechanism(
    outfile: pathlib.Path,
    blocks: Sequence[BlockScores],
    corrs: NodeCorrelations,
    *,
    method: ClassificationMethod = "mlp",
    overwrite: bool = False,
) -> None:
    """Plot Family I block-wise decoding and R^200 probe-subspace correlations."""
    labels = [BLOCK_SHORT_NAMES.get(b.name, b.name) for b in blocks]
    xpos = np.arange(len(blocks))
    width = 0.36

    task = [_select_result(b.task, method) for b in blocks]
    fp = [_select_result(b.fp, method) for b in blocks]

    task_mean = np.array([res.mean_bacc for res in task])
    task_err = np.maximum(
        np.array([
            [res.mean_bacc - res.ci_lo for res in task],
            [res.ci_hi - res.mean_bacc for res in task],
        ]),
        0.0,
    )
    fp_mean = np.array([res.mean_bacc for res in fp])
    fp_err = np.maximum(
        np.array([
            [res.mean_bacc - res.ci_lo for res in fp],
            [res.ci_hi - res.mean_bacc for res in fp],
        ]),
        0.0,
    )

    with figure(
        outfile.with_stem(f"{outfile.stem}-blocks"),
        overwrite=overwrite,
    ) as fig:
        (ax,) = fig.axes
        ax.bar(
            xpos - width / 2,
            task_mean,
            width,
            yerr=task_err,
            capsize=3,
            label="Task decoding",
        )
        ax.bar(
            xpos + width / 2,
            fp_mean,
            width,
            yerr=fp_err,
            capsize=3,
            label="Fingerprinting",
        )
        ax.axhline(task[0].chance, color="C0", ls="--", lw=1.2, alpha=0.8)
        ax.axhline(fp[0].chance, color="C1", ls=":", lw=1.2, alpha=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0.0, 0.7)
        ax.grid(visible=False, which="both")
        ax.legend(fontsize=16)

    box_data = [
        corrs.r_v32,
        corrs.r_v33,
        corrs.r_v34,
        corrs.r_orth_med,
        corrs.r_orth_max,
    ]
    box_labels = [
        r"$|\mathbf{v}_{3,2}|$",
        r"$|\mathbf{v}_{3,3}|$",
        r"$|\mathbf{v}_{3,4}|$",
        r"$\mathbf{u}_k$ (median)",
        r"$\mathbf{u}_k$ (max)",
    ]

    with figure(
        outfile.with_stem(f"{outfile.stem}-alignment"),
        overwrite=overwrite,
    ) as fig:
        (ax,) = fig.axes
        bp = ax.boxplot(
            box_data,
            tick_labels=box_labels,
            patch_artist=True,
            showfliers=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("C0" if i < 3 else "C3")
            patch.set_alpha(0.65)
        ax.axhline(0.0, color="k", ls="--", lw=1.0, alpha=0.6)
        ax.set_ylabel(r"Pearson $r$ with $|\hat{z}_i|$")
        ax.set_ylim(-0.15, 1.0)
        ax.tick_params(axis="x", labelrotation=20)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
        ax.grid(visible=False, axis="x", which="both")


# }}}


def main(
    ae_file: pathlib.Path,
    matrix_file: pathlib.Path,
    *,
    method: ClassificationMethod = "mlp",
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    device = torch.get_default_device()
    log.info("Device: %s", device)

    if outfile is None:
        outfile = ae_file.with_stem(f"{ae_file.stem}-mechanism").with_suffix("")

    try:
        ds_ae = load_feature_datasets(ae_file, binary=False)
    except Exception as exc:
        log.error("Failed to load '%s': %s", ae_file, exc)
        return 1

    try:
        ds_mat = load_feature_datasets(matrix_file, binary=False)
    except Exception as exc:
        log.error("Failed to load '%s': %s", matrix_file, exc)
        return 1

    if ds_ae.class_names != ds_mat.class_names:
        log.error(
            "Feature classes in '%s' and '%s' do not match: %s vs %s",
            ae_file,
            matrix_file,
            ds_ae.class_names,
            ds_mat.class_names,
        )
        return 1

    mats = reconstruct_matrices(ds_mat)
    blocks = evaluate_blocks(
        ds_ae,
        methods=(method,),
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        device=device,
    )

    corrs = compute_node_correlations(ds_ae, mats)
    r2 = compute_degree_moment_r2(
        ds_ae, mats, n_splits=n_splits, n_repeats=n_repeats, seed=seed
    )

    log_summary(blocks, corrs, r2, method=method)
    plot_mechanism(outfile, blocks, corrs, method=method, overwrite=overwrite)

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
    parser.add_argument(
        "ae_file",
        type=pathlib.Path,
        help="Merged Autoencoder-DMD feature file",
    )
    parser.add_argument(
        "matrix_file",
        type=pathlib.Path,
        help="Merged connectome matrix feature file",
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default device used by the framework",
    )
    parser.add_argument(
        "--method",
        choices=get_args(ClassificationMethod),
        default="mlp",
        help="Classifier used in the block table and panel (a)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of subject-grouped CV folds",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of repeated CV runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splits, bootstraps, and MLP initialization",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output plots",
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
    logging.getLogger("lib.torchlib").setLevel(logging.WARNING)

    with torch_default_device(args.device):
        torch.set_float32_matmul_precision("high")
        raise SystemExit(
            main(
                args.ae_file,
                args.matrix_file,
                method=args.method,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                seed=args.seed,
                outfile=args.outfile,
                overwrite=args.force,
            )
        )

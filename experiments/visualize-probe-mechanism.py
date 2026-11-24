# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import math
import pathlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args

import numpy as np
import torch

import commonlib  # ruff: ignore[unused-import]
from classifylib import (
    FeatureDataset,
    bootstrap_balanced_accuracy_ci,
    bootstrap_identification_ci,
    flatten_feature_datasets,
    load_feature_datasets,
    make_subject_index,
    run_leave_one_condition_out_cv,
    run_ridge_cv,
    run_subject_grouped_cv,
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

ClassificationMethod = Literal["lda", "mlp"]

_BLOCK_SHORT_NAMES: dict[str, str] = {
    "Leading eigenvalue moduli": "Eigenvalues",
    "Control sensitivity (log)": "Control",
    "Attractor node magnitudes": "Magnitudes",
    "Attractor energy": "Energy",
    "Full AE-DMD": "Full",
}


def _block_selections(ds: FeatureDataset) -> list[tuple[str, int | None]]:
    """Return the named feature blocks used by the mechanism analysis."""
    return [
        *((descr, i) for i, descr in enumerate(ds.descrs)),
        ("Full AE-DMD", None),
    ]


def _reconstruct_matrices(ds: FeatureDataset) -> Array3D[np.floating[Any]]:
    """Reconstruct symmetric zero-diagonal matrices from lower-triangular features."""
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


# {{{ block-wise classification


@dataclass(frozen=True)
class BlockScores:
    name: str
    nfeatures: int
    task_lda: tuple[float, float, float, float]
    task_mlp: tuple[float, float, float, float]
    fp_lda: tuple[float, float, float, float]
    fp_mlp: tuple[float, float, float, float]


def _eval_task_block(
    ds: FeatureDataset,
    *,
    method: ClassificationMethod,
    n_splits: int,
    n_repeats: int,
    seed: int,
    device: torch.device | str,
) -> tuple[float, float, float, float]:
    x, y, subject_ids = flatten_feature_datasets(ds)
    oof_preds, fold_bacc, _ = run_subject_grouped_cv(
        x,
        y,
        subject_ids,
        nclasses=len(ds.class_names),
        method=method,
        zscore_features=True,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        device=device,
    )
    ci_lo, ci_hi = bootstrap_balanced_accuracy_ci(y, oof_preds, subject_ids, seed=seed)
    return (
        float(np.mean(fold_bacc)),
        float(np.std(fold_bacc)),
        float(ci_lo),
        float(ci_hi),
    )


def _eval_fp_block(
    ds: FeatureDataset,
    *,
    method: ClassificationMethod,
    seed: int,
    device: torch.device | str,
) -> tuple[float, float, float, float]:
    x = ds.to_tensor()
    oof_preds, fold_bacc = run_leave_one_condition_out_cv(
        x,
        method=method,
        zscore_features=True,
        seed=seed,
        device=device,
    )
    ci_lo, ci_hi = bootstrap_identification_ci(oof_preds, seed=seed)
    return (
        float(np.mean(fold_bacc)),
        float(np.std(fold_bacc)),
        float(ci_lo),
        float(ci_hi),
    )


def evaluate_blocks(
    ds_ae: FeatureDataset,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[BlockScores]:
    """Evaluate 7-way task and fingerprinting per Family I block."""
    selections = _block_selections(ds_ae)

    results: list[BlockScores] = []
    for name, idx in selections:
        sub = ds_ae if idx is None else select_feature_group(ds_ae, idx)
        nfeat = int(sub.x[0].shape[1])

        log.info("Evaluating block '%s' (%d features)...", name, nfeat)
        results.append(
            BlockScores(
                name=name,
                nfeatures=nfeat,
                task_lda=_eval_task_block(
                    sub,
                    method="lda",
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    seed=seed,
                    device=device,
                ),
                task_mlp=_eval_task_block(
                    sub,
                    method="mlp",
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    seed=seed,
                    device=device,
                ),
                fp_lda=_eval_fp_block(sub, method="lda", seed=seed, device=device),
                fp_mlp=_eval_fp_block(sub, method="mlp", seed=seed, device=device),
            )
        )

    return results


# }}}


# {{{ probe-subspace alignment in R^200 & scalar degree moments


@dataclass(frozen=True)
class NodeCorrelations:
    r_d: Array1D[np.floating[Any]]
    r_d2: Array1D[np.floating[Any]]
    r_ad2: Array1D[np.floating[Any]]
    r_orth_med: Array1D[np.floating[Any]]
    r_orth_max: Array1D[np.floating[Any]]


@dataclass(frozen=True)
class DegreeMomentR2:
    block: str
    target: str
    r2_mean: float
    r2_ci_lo: float
    r2_ci_hi: float


def _pearsonr(u: Array1D[np.floating[Any]], v: Array1D[np.floating[Any]]) -> float:
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    denom = float(np.linalg.norm(u_c) * np.linalg.norm(v_c))
    if denom < 1.0e-15:
        return 0.0
    return float(np.dot(u_c, v_c) / denom)


def compute_node_correlations(
    ds_ae: FeatureDataset,
    mats: Array3D[np.floating[Any]],
) -> NodeCorrelations:
    """Compute per-connectome R^200 correlations between |z_N| and P(A) vs u_k."""
    idx = ds_ae.descrs.index("Attractor node magnitudes")
    ds_mag = select_feature_group(ds_ae, idx)
    mags, _, _ = flatten_feature_datasets(ds_mag)
    if mats.shape[0] != mags.shape[0] or mats.shape[1] != mags.shape[1]:
        raise ValueError(
            f"shape mismatch between matrices {mats.shape} and magnitudes {mags.shape}"
        )

    total, n = mags.shape
    ones = np.ones(n, dtype=np.float64)
    ones_unit = ones / np.sqrt(n)

    r_d = np.empty(total, dtype=np.float64)
    r_d2 = np.empty(total, dtype=np.float64)
    r_ad2 = np.empty(total, dtype=np.float64)
    r_orth_med = np.empty(total, dtype=np.float64)
    r_orth_max = np.empty(total, dtype=np.float64)

    for idx, (mat, mag) in enumerate(zip(mats, mags, strict=True)):
        d = mat @ ones
        d2 = d * d
        ad2 = mat @ d2

        r_d[idx] = _pearsonr(mag, np.abs(d))
        r_d2[idx] = _pearsonr(mag, d2)
        r_ad2[idx] = _pearsonr(mag, np.abs(ad2))

        _, eigvecs = np.linalg.eigh(mat)
        proj = np.abs(eigvecs.T @ ones_unit)
        orth_mask = proj < 0.05
        if not np.any(orth_mask):
            orth_mask = proj <= np.median(proj)

        orth_rs = np.array([
            abs(_pearsonr(mag, np.abs(eigvecs[:, k])))
            for k in np.flatnonzero(orth_mask)
        ])
        r_orth_med[idx] = float(np.median(orth_rs))
        r_orth_max[idx] = float(np.max(orth_rs))

    return NodeCorrelations(
        r_d=r_d,
        r_d2=r_d2,
        r_ad2=r_ad2,
        r_orth_med=r_orth_med,
        r_orth_max=r_orth_max,
    )


def compute_degree_moment_r2(
    ds_ae: FeatureDataset,
    mats: Array3D[np.floating[Any]],
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> list[DegreeMomentR2]:
    """Predict functional scalar moments of A1 from Family I blocks via RidgeCV."""
    d = np.sum(mats, axis=2)
    targets = {
        "||A1||_2": np.linalg.norm(d, axis=1),
        "std(A1)": np.std(d, axis=1),
        "1^T A 1": np.sum(d, axis=1),
    }

    _, _, subject_ids = flatten_feature_datasets(ds_ae)
    subject_index = make_subject_index(subject_ids)
    nsamples = subject_ids.size  # ty: ignore[missing-attribute]
    is_struct = np.zeros(nsamples, dtype=np.bool_)

    blocks = _block_selections(ds_ae)

    results: list[DegreeMomentR2] = []
    for bname, idx in blocks:
        sub = ds_ae if idx is None else select_feature_group(ds_ae, idx)
        x, _, _ = flatten_feature_datasets(sub)
        for tname, y in targets.items():
            res = run_ridge_cv(
                x,
                y,
                subject_index,
                is_struct,
                name=bname,
                target_name=tname,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
            )
            results.append(
                DegreeMomentR2(
                    block=bname,
                    target=tname,
                    r2_mean=float(res.r2_mean),
                    r2_ci_lo=float(res.r2_ci_lo),
                    r2_ci_hi=float(res.r2_ci_hi),
                )
            )

    return results


# }}}


# {{{ reporting & plotting


def _fmt_score(vals: tuple[float, float, float, float]) -> str:
    mean, std, lo, hi = vals
    return f"${mean:.3f} \\pm {std:.3f}$ & $[{lo:.3f}, {hi:.3f}]$"


def _med_iqr(arr: Array1D[np.floating[Any]]) -> tuple[float, float, float]:
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return float(q50), float(q25), float(q75)


def log_summary(
    blocks: Sequence[BlockScores],
    corrs: NodeCorrelations,
    r2: Sequence[DegreeMomentR2],
) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"\textbf{AE-DMD block} & \textbf{Size}"
            r" & \textbf{Task (MLP)} & \textbf{95\% CI}"
            r" & \textbf{FP (MLP)} & \textbf{95\% CI} \\"
        ),
        r"\midrule",
    ]
    for b in blocks:
        lines.append(
            f"{b.name} & {b.nfeatures} & "
            f"{_fmt_score(b.task_mlp)} & {_fmt_score(b.fp_mlp)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    log.info("Family I block-wise table:\n%s", "\n".join(lines))

    log.info(
        "Functional R^200 Pearson r with |z_N| (median [IQR]): "
        "|A1|=%.3f [%.3f, %.3f], (A1)^2=%.3f [%.3f, %.3f], "
        "|A(A1)^2|=%.3f [%.3f, %.3f], orth med=%.3f [%.3f, %.3f], "
        "orth max=%.3f [%.3f, %.3f]",
        *_med_iqr(corrs.r_d),
        *_med_iqr(corrs.r_d2),
        *_med_iqr(corrs.r_ad2),
        *_med_iqr(corrs.r_orth_med),
        *_med_iqr(corrs.r_orth_max),
    )

    size_by_block = {b.name: b.nfeatures for b in blocks}
    targets = ["||A1||_2", "std(A1)", "1^T A 1"]
    by_block: dict[str, dict[str, DegreeMomentR2]] = {}
    for entry in r2:
        by_block.setdefault(entry.block, {})[entry.target] = entry

    r2_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        (
            r"\textbf{AE-DMD block} & \textbf{Size}"
            r" & \textbf{$\norm{\matr{A}\vect{1}}_2$}"
            r" & \textbf{$\mathrm{std}(\matr{A}\vect{1})$}"
            r" & \textbf{$\vect{1}^\top \matr{A} \vect{1}$} \\"
        ),
        r"\midrule",
    ]
    for bname, entries in by_block.items():
        vals = " & ".join(f"${entries[t].r2_mean:.3f}$" for t in targets)
        cis = " & ".join(
            f"$[{entries[t].r2_ci_lo:.3f}, {entries[t].r2_ci_hi:.3f}]$" for t in targets
        )
        r2_lines.append(f"{bname} & {size_by_block[bname]} & {vals} \\\\")
        r2_lines.append(f" & & {cis} \\\\")
        if bname != list(by_block)[-1]:
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
    labels = [_BLOCK_SHORT_NAMES.get(b.name, b.name) for b in blocks]
    xpos = np.arange(len(blocks))
    width = 0.36

    task_vals = [getattr(b, f"task_{method}") for b in blocks]
    fp_vals = [getattr(b, f"fp_{method}") for b in blocks]

    task_mean = np.array([v[0] for v in task_vals])
    task_err = np.maximum(
        np.array([
            [v[0] - v[2] for v in task_vals],
            [v[3] - v[0] for v in task_vals],
        ]),
        0.0,
    )
    fp_mean = np.array([v[0] for v in fp_vals])
    fp_err = np.maximum(
        np.array([
            [v[0] - v[2] for v in fp_vals],
            [v[3] - v[0] for v in fp_vals],
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
        ax.axhline(1.0 / 7.0, color="C0", ls="--", lw=1.2, alpha=0.8)
        ax.axhline(1.0 / 48.0, color="C1", ls=":", lw=1.2, alpha=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0.0, 0.7)
        ax.grid(visible=False, which="both")
        ax.legend(fontsize=16)

    box_data = [
        corrs.r_d,
        corrs.r_d2,
        corrs.r_ad2,
        corrs.r_orth_med,
        corrs.r_orth_max,
    ]
    box_labels = [
        r"$|\mathbf{A}\mathbf{1}|$",
        r"$(\mathbf{A}\mathbf{1})^2$",
        r"$|\mathbf{A}(\mathbf{A}\mathbf{1})^2|$",
        r"$\mathbf{u}_k \perp \mathbf{1}$ (med)",
        r"$\mathbf{u}_k \perp \mathbf{1}$ (max)",
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
        ax.set_xticklabels(box_labels, rotation=20, ha="right")
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
    cachefile: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not ae_file.exists():
        log.error("AE-DMD feature file does not exist: '%s'.", ae_file)
        return 1
    if not matrix_file.exists():
        log.error("Matrix feature file does not exist: '%s'.", matrix_file)
        return 1

    device = torch.get_default_device()
    log.info("Device: %s", device)

    if cachefile is not None and cachefile.exists() and not overwrite:
        log.info("Loading cached Stage 3 results from '%s'.", cachefile)
        cached = np.load(cachefile, allow_pickle=True)
        blocks = [BlockScores(**row) for row in cached["blocks"][()]]
        corrs = NodeCorrelations(
            r_d=cached["r_d"],
            r_d2=cached["r_d2"],
            r_ad2=cached["r_ad2"],
            r_orth_med=cached["r_orth_med"],
            r_orth_max=cached["r_orth_max"],
        )
        r2 = [DegreeMomentR2(**row) for row in cached["r2"][()]]
    else:
        ds_ae = load_feature_datasets(ae_file, binary=False)
        ds_mat = load_feature_datasets(matrix_file, binary=False)
        if ds_ae.class_names != ds_mat.class_names:
            raise ValueError(
                f"class mismatch: {ds_ae.class_names} vs {ds_mat.class_names}"
            )
        mats = _reconstruct_matrices(ds_mat)

        blocks = evaluate_blocks(
            ds_ae,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            device=device,
        )
        corrs = compute_node_correlations(ds_ae, mats)
        r2 = compute_degree_moment_r2(
            ds_ae, mats, n_splits=n_splits, n_repeats=n_repeats, seed=seed
        )
        if cachefile is not None:
            np.savez(
                cachefile,
                blocks=np.array([asdict(b) for b in blocks], dtype=object),
                r2=np.array([asdict(entry) for entry in r2], dtype=object),
                **asdict(corrs),
            )
            log.info("Saved Stage 3 cache to '%s'.", cachefile)

    log_summary(blocks, corrs, r2)

    if outfile is None:
        outfile = ae_file.with_stem(f"{ae_file.stem}-mechanism").with_suffix("")
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
        help="Classifier shown in panel (a)",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=pathlib.Path, default=None)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
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
                cachefile=args.cache,
                outfile=args.outfile,
                overwrite=args.force,
            )
        )

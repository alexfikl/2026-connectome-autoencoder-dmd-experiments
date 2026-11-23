# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
import torch

from commonlib import flatten_feature_datasets, load_feature_datasets
from nneuroutil.helpers import FuzzyChoiceAction
from nneuroutil.interop.torch import available_device_names, torch_default_device
from nneuroutil.typing import Array1D, Array2D
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.classify_task_decoding")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Classify tasks using features computed by make-features.py

Examples:

    > {SCRIPT_PATH.name} run --method lda dataset-features-matrix.npz
    > {SCRIPT_PATH.name} summary --binary data/dataset-features-*.npz
"""

ClassificationMethod = Literal["lda", "svm", "mlp"]


class ClassificationResult(NamedTuple):
    label: str
    nfeatures: int
    nclasses: int
    method: ClassificationMethod
    mean_bacc: float
    std_bacc: float
    ci_lo: float
    ci_hi: float
    mean_f1: float
    p_val: float | None


def fit_and_predict(
    x_train_list: list[torch.Tensor],
    y_train_list: list[int],
    x_test: torch.Tensor,
    method: ClassificationMethod,
    *,
    seed: int,
) -> torch.Tensor:
    """Fit classifier on training tensors and return predicted labels."""
    if method == "lda":
        from nneuroutil.classifier import classify_linear_discriminant_analysis

        clf = classify_linear_discriminant_analysis(x_train_list, y_train_list)
        return clf.predict(x_test, metric="l2")  # ty: ignore[invalid-explicit-return]

    if method == "svm":
        from nneuroutil.classifier import classify_support_vector_machine

        clf = classify_support_vector_machine(x_train_list, y_train_list)
        return clf.predict(x_test)  # ty: ignore[invalid-explicit-return]

    if method == "mlp":
        from torchlib import classify_multilayer_perceptron

        torch.manual_seed(seed)
        clf = classify_multilayer_perceptron(
            x_train_list, y_train_list, learning_rate_init=1.0e-2, max_iter=400
        )
        return clf.predict(x_test)

    raise ValueError(f"unknown method: {method!r}")


def make_subject_rows(
    groups: Array1D[np.integer[Any]],
) -> list[Array1D[np.integer[Any]]]:
    """Return the row indices of each subject in ``groups``."""
    return [np.flatnonzero(groups == s) for s in np.unique(groups)]


def bootstrap_subject_ci(
    y_true: Array1D[np.integer[Any]],
    oof_preds: Array2D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap CI over subjects for balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score

    subj_rows = make_subject_rows(groups)
    n_subjects = len(subj_rows)
    n_repeats = oof_preds.shape[0]

    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_subjects, size=n_subjects)
        idx = np.concatenate([subj_rows[s] for s in sampled])
        vals[b] = balanced_accuracy_score(
            np.tile(y_true[idx], n_repeats), np.ravel(oof_preds[:, idx])
        )

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


# {{{ plot_confusion


def plot_confusion(
    basename: pathlib.Path,
    cm: Array2D[np.integer[Any]],
    class_names: tuple[str, ...],
    *,
    method: str,
    balanced_acc: float,
    overwrite: bool = False,
) -> None:
    """Plot the aggregated confusion matrix of the out-of-fold predictions."""
    nclasses = len(class_names)
    vmax = max(np.max(cm), 1)

    with figure(
        basename.with_stem(f"{basename.stem}-confusion-{method}"),
        figsize=(8, 7),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.grid(visible=False)

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(nclasses))
        ax.set_yticks(range(nclasses))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize="medium")
        ax.set_yticklabels(class_names, fontsize="medium")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(rf"{method.upper()} (bal. acc. ${balanced_acc:.3f}$)")

        for i, j in np.ndindex(cm.shape):
            if cm[i, j] == 0:
                continue

            color = "white" if cm[i, j] > 0.5 * vmax else "black"
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=color,
                fontsize="small",
            )

        fig.colorbar(im, cax=make_colorbar_axes(ax), label="Count")


# }}}


def run_cross_validation(
    x: Array2D[np.floating[Any]],
    y: Array1D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
    *,
    nclasses: int,
    method: ClassificationMethod,
    zscore_features: bool,
    n_splits: int,
    n_repeats: int,
    seed: int,
    device: torch.device | str,
) -> tuple[Array2D[np.integer[Any]], list[float], list[float]]:
    """Run repeated subject-grouped K-fold CV on ``(nsamples, nfeatures)``."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    unique_subjects = np.unique(groups)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    nsamples = x.shape[0]
    oof_preds = np.empty((n_repeats, nsamples), dtype=np.int64)
    fold_bacc: list[float] = []
    fold_f1: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(unique_subjects)):
        rep = fold_idx // n_splits
        train_mask = np.isin(groups, unique_subjects[train_idx])
        test_mask = np.isin(groups, unique_subjects[test_idx])

        x_tr_np = x[train_mask]
        y_tr_np = y[train_mask]
        x_te_np = x[test_mask]
        y_te_np = y[test_mask]

        if zscore_features:
            scaler = StandardScaler()
            x_tr_np = scaler.fit_transform(x_tr_np)
            x_te_np = scaler.transform(x_te_np)

        x_train_list = [
            torch.from_numpy(x_tr_np[y_tr_np == c]).to(device) for c in range(nclasses)
        ]
        y_train_list = list(range(nclasses))
        x_test_t = torch.from_numpy(x_te_np).to(device)

        y_pred_t = fit_and_predict(
            x_train_list,
            y_train_list,
            x_test_t,
            method,
            seed=seed + fold_idx,
        )
        y_pred = y_pred_t.detach().cpu().numpy()

        oof_preds[rep, test_mask] = y_pred
        fold_bacc.append(balanced_accuracy_score(y_te_np, y_pred))
        fold_f1.append(f1_score(y_te_np, y_pred, average="macro"))

    return oof_preds, fold_bacc, fold_f1


def evaluate_dataset(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod,
    binary: bool,
    zscore_features: bool,
    n_splits: int,
    n_repeats: int,
    n_permutations: int,
    seed: int,
    device: torch.device | str,
) -> tuple[
    ClassificationResult,
    Array1D[np.integer[Any]],
    Array2D[np.integer[Any]],
    tuple[str, ...],
]:
    ds = load_feature_datasets(filename, binary=binary)
    x, y, groups = flatten_feature_datasets(ds)
    class_names = ds.class_names
    nclasses = len(class_names)
    unique_subjects = np.unique(groups)

    log.info(
        "Evaluating '%s' (%s, %s, z=%s): %d samples (%d subjects), %d features.",
        filename.stem,
        "binary" if binary else "multiclass",
        method.upper(),
        zscore_features,
        x.shape[0],
        unique_subjects.size,
        x.shape[1],
    )

    oof_preds, fold_bacc, fold_f1 = run_cross_validation(
        x,
        y,
        groups,
        nclasses=nclasses,
        method=method,
        zscore_features=zscore_features,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        device=device,
    )

    mean_bacc = np.mean(fold_bacc)
    std_bacc = np.std(fold_bacc)
    mean_f1 = np.mean(fold_f1)
    ci_lo, ci_hi = bootstrap_subject_ci(y, oof_preds, groups, seed=seed)

    p_val: float | None = None
    perm_str = ""
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        null_baccs = np.empty(n_permutations, dtype=np.float64)
        subj_rows = make_subject_rows(groups)
        for p in range(n_permutations):
            y_perm = np.copy(y)
            for rows in subj_rows:
                y_perm[rows] = rng.permutation(y_perm[rows])
            _, p_bacc, _ = run_cross_validation(
                x,
                y_perm,
                groups,
                nclasses=nclasses,
                method=method,
                zscore_features=zscore_features,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed + 1000 + p,
                device=device,
            )
            null_baccs[p] = np.mean(p_bacc)
        p_val = (np.sum(null_baccs >= mean_bacc) + 1) / (n_permutations + 1)
        perm_str = f" | perm p={p_val:.4g} (null={np.mean(null_baccs):.4f})"

    log.info(
        "Bal. Acc.: %.4f +/- %.4f (95%% CI [%.4f, %.4f]) | Macro-F1: %.4f%s",
        mean_bacc,
        std_bacc,
        ci_lo,
        ci_hi,
        mean_f1,
        perm_str,
    )

    result = ClassificationResult(
        label=ds.label,
        nfeatures=x.shape[1],
        nclasses=nclasses,
        method=method,
        mean_bacc=mean_bacc,
        std_bacc=std_bacc,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        mean_f1=mean_f1,
        p_val=p_val,
    )

    return result, y, oof_preds, class_names


def _format_score(res: ClassificationResult, *, underline: bool = False) -> str:
    score = f"${res.mean_bacc:.3f} \\pm {res.std_bacc:.3f}$"
    return f"\\underline{{{score}}}" if underline else score


def format_latex_table(
    rows: Sequence[tuple[ClassificationResult, ClassificationResult]],
) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Feature family} & \textbf{Dim.} & \textbf{LDA} & \textbf{MLP} \\",
        r"\midrule",
    ]
    if rows:
        # NOTE: underline the lowest score in each method column
        lda_min = min(res_lda.mean_bacc for res_lda, _ in rows)
        mlp_min = min(res_mlp.mean_bacc for _, res_mlp in rows)

        for res_lda, res_mlp in rows:
            lines.append(
                f"{res_lda.label} & {res_lda.nfeatures:,} & "
                f"{_format_score(res_lda, underline=res_lda.mean_bacc == lda_min)}"
                f" & {_format_score(res_mlp, underline=res_mlp.mean_bacc == mlp_min)}"
                r" \\"
            )

        chance = 1.0 / rows[0][0].nclasses
        lines.extend([
            r"\midrule",
            f"Chance & -- & ${chance:.3f}$ & ${chance:.3f}$ \\\\",
        ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return "\n".join(lines)


def main(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod = "svm",
    binary: bool = False,
    zscore_features: bool = True,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_permutations: int = 0,
    seed: int = 42,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    device = torch.get_default_device()
    log.info("Device: %s", device)

    try:
        res, y, oof_preds, class_names = evaluate_dataset(
            filename,
            method=method,
            binary=binary,
            zscore_features=zscore_features,
            n_splits=n_splits,
            n_repeats=n_repeats,
            n_permutations=n_permutations,
            seed=seed,
            device=device,
        )
    except Exception as exc:
        log.error("Failed to evaluate '%s': %s", filename, exc)
        return 1

    y_rep = np.tile(y, n_repeats)
    cm = confusion_matrix(y_rep, np.ravel(oof_preds), labels=list(range(res.nclasses)))
    overall_bacc = balanced_accuracy_score(y_rep, np.ravel(oof_preds))

    suffix = "-binary" if binary else "-multiclass"
    if zscore_features:
        suffix += "-zscore"
    basename = (
        filename.with_stem(f"{filename.stem}{suffix}").with_suffix("")
        if outfile is None
        else outfile.with_suffix("")
    )
    plot_confusion(
        basename,
        cm,
        class_names,
        method=method,
        balanced_acc=overall_bacc,
        overwrite=overwrite,
    )

    return 0


def summary(
    filenames: Sequence[pathlib.Path],
    *,
    binary: bool = False,
    zscore_features: bool = True,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_permutations: int = 0,
    seed: int = 42,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    device = torch.get_default_device()
    log.info("Device: %s", device)

    rows: list[tuple[ClassificationResult, ClassificationResult]] = []
    for filename in filenames:
        try:
            res_lda, _, _, _ = evaluate_dataset(
                filename,
                method="lda",
                binary=binary,
                zscore_features=zscore_features,
                n_splits=n_splits,
                n_repeats=n_repeats,
                n_permutations=n_permutations,
                seed=seed,
                device=device,
            )
            res_mlp, _, _, _ = evaluate_dataset(
                filename,
                method="mlp",
                binary=binary,
                zscore_features=zscore_features,
                n_splits=n_splits,
                n_repeats=n_repeats,
                n_permutations=n_permutations,
                seed=seed,
                device=device,
            )
        except Exception as exc:
            log.error("Failed to evaluate '%s': %s", filename, exc)
            return 1

        rows.append((res_lda, res_mlp))

    table = format_latex_table(rows)
    if outfile is None:
        print(table)
    else:
        if outfile.exists() and not overwrite:
            log.error("Output file '%s' exists (use --force to overwrite).", outfile)
            return 1

        outfile.write_text(f"{table}\n", encoding="utf-8")
        log.info("Wrote summary table to '%s'.", outfile)

    return 0


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default device used by the framework",
    )
    common.add_argument(
        "--binary",
        action="store_true",
        help="Group classes by scan type (Functional vs. Structural)",
    )
    common.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of subject-grouped CV folds",
    )
    common.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of repeated CV runs",
    )
    common.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Number of subject-level label permutations (0 to skip)",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splits and permutations",
    )
    common.add_argument(
        "--no-zscore",
        action="store_false",
        dest="zscore",
        help="Disable z-scoring each feature to zero mean and unit variance",
    )
    common.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files",
    )
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        formatter_class=HelpFormatter,
        help="Evaluate a single feature dataset and plot its confusion matrix",
    )
    run_parser.add_argument("filename", type=pathlib.Path)
    run_parser.add_argument(
        "--method",
        choices=get_args(ClassificationMethod),
        default="lda",
        help="The method used to do the classification",
    )
    run_parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
    )

    summary_parser = subparsers.add_parser(
        "summary",
        parents=[common],
        formatter_class=HelpFormatter,
        help="Evaluate multiple feature datasets (LDA and MLP) and emit a LaTeX table",
    )
    summary_parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    summary_parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Optional output .tex file (defaults to stdout)",
    )

    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)
    logging.getLogger("lib.torchlib").setLevel(logging.WARNING)

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        if args.command == "run":
            raise SystemExit(
                main(
                    args.filename,
                    outfile=args.outfile,
                    method=args.method,
                    binary=args.binary,
                    zscore_features=args.zscore,
                    n_splits=args.n_splits,
                    n_repeats=args.n_repeats,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    overwrite=args.force,
                )
            )
        elif args.command == "summary":
            raise SystemExit(
                summary(
                    args.filenames,
                    binary=args.binary,
                    zscore_features=args.zscore,
                    n_splits=args.n_splits,
                    n_repeats=args.n_repeats,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise ValueError(f"unsupported command: {args.command}")

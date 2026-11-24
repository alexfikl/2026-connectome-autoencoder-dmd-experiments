# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
import torch

from commonlib import FeatureDataset, load_feature_datasets
from nneuroutil.helpers import FuzzyChoiceAction
from nneuroutil.interop.torch import available_device_names, torch_default_device
from nneuroutil.typing import Array2D, Array3D
from nneuroutil.visualization import figure

log = logging.getLogger("lib.classify_subject_fingerprinting")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Evaluate leave-one-condition-out subject fingerprinting from feature datasets.

For each feature file, the features are reshaped to ``(ntasks, nsubjects,
nfeatures)``, assuming matched subjects in the same order in every class. In
each fold, one task is held out and the remaining ``ntasks - 1`` tasks are used
to z-score the features and train the classifier (G-SVD LDA or MLP, following
Hannum et al., 2023).

Examples:

    > {SCRIPT_PATH.name} run --method lda data/dataset-features-matrix.npz
    > {SCRIPT_PATH.name} summary --device cuda data/dataset-features-*.npz
"""

ClassificationMethod = Literal["lda", "svm", "mlp"]


class ClassificationResult(NamedTuple):
    label: str
    nfeatures: int
    nsubjects: int
    method: ClassificationMethod
    mean_bacc: float
    std_bacc: float
    ci_lo: float
    ci_hi: float
    p_val: float | None


def to_tensor(ds: FeatureDataset) -> Array3D[np.floating[Any]]:
    """Stack matched-subject features to ``(ntasks, nsubjects, nfeatures)``."""
    shapes = {np.asarray(x).shape for x in ds.x}
    if len(shapes) != 1:
        raise ValueError("expected all classes to have the same shape")

    return np.stack([np.asarray(x) for x in ds.x], axis=0)


def fit_and_predict(
    x_train_list: list[torch.Tensor],
    y_train_list: list[int],
    x_test: torch.Tensor,
    method: ClassificationMethod,
    *,
    seed: int,
) -> torch.Tensor:
    """Fit classifier on training tensors and return predicted subject labels."""
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

        # NOTE: the features are stored as float64, but the MLP is significantly
        # faster in single precision and the resulting accuracy is unchanged
        torch.manual_seed(seed)
        clf = classify_multilayer_perceptron(
            [xi.float() for xi in x_train_list],
            y_train_list,
            alpha=1.0,
            learning_rate_init=1.0e-2,
            max_iter=200,
        )
        return clf.predict(x_test)

    raise ValueError(f"unknown method: {method!r}")


def bootstrap_subject_ci(
    oof_preds: Array2D[np.integer[Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap CI over subjects for identification accuracy."""
    _ntasks, nsubjects = oof_preds.shape
    targets = np.arange(nsubjects)[None, :]
    correct = oof_preds == targets

    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, nsubjects, size=nsubjects)
        vals[b] = np.mean(correct[:, sampled])  # ty: ignore[not-subscriptable]

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def run_cross_validation(
    x: Array3D[np.floating[Any]],
    *,
    method: ClassificationMethod,
    zscore_features: bool,
    seed: int,
    device: torch.device | str,
) -> tuple[Array2D[np.integer[Any]], list[float]]:
    """Run leave-one-condition-out CV on ``(ntasks, nsubjects, nfeatures)``."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    ntasks, nsubjects, nfeatures = x.shape
    y_train_list = list(range(nsubjects))
    y_test = np.arange(nsubjects, dtype=np.int64)

    oof_preds = np.empty((ntasks, nsubjects), dtype=np.int64)
    fold_bacc: list[float] = []

    for t in range(ntasks):
        x_tr_np = np.delete(x, t, axis=0)
        x_te_np = x[t]

        if zscore_features:
            scaler = StandardScaler()
            tr_flat = scaler.fit_transform(np.reshape(x_tr_np, (-1, nfeatures)))
            x_te_np = scaler.transform(x_te_np)
            x_tr_np = np.reshape(tr_flat, (ntasks - 1, nsubjects, nfeatures))

        x_train_list = [
            torch.from_numpy(x_tr_np[:, s, :]).to(device) for s in range(nsubjects)
        ]
        x_test_t = torch.from_numpy(x_te_np).to(device)

        y_pred_t = fit_and_predict(
            x_train_list,
            y_train_list,
            x_test_t,
            method,
            seed=seed + t,
        )
        y_pred = y_pred_t.detach().cpu().numpy()

        oof_preds[t] = y_pred
        fold_bacc.append(float(balanced_accuracy_score(y_test, y_pred)))

    return oof_preds, fold_bacc


def evaluate_dataset(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod,
    zscore_features: bool,
    n_permutations: int,
    seed: int,
    device: torch.device | str,
) -> tuple[
    ClassificationResult,
    list[float],
    tuple[str, ...],
]:
    ds = load_feature_datasets(filename)
    x = to_tensor(ds)
    ntasks, nsubjects, nfeatures = x.shape

    log.info(
        "Evaluating '%s' (%s, z=%s): %d tasks x %d subjects, %d features.",
        filename.stem,
        method.upper(),
        zscore_features,
        ntasks,
        nsubjects,
        nfeatures,
    )

    oof_preds, fold_bacc = run_cross_validation(
        x,
        method=method,
        zscore_features=zscore_features,
        seed=seed,
        device=device,
    )

    mean_bacc = float(np.mean(fold_bacc))
    std_bacc = float(np.std(fold_bacc))
    ci_lo, ci_hi = bootstrap_subject_ci(oof_preds, seed=seed)

    p_val: float | None = None
    perm_str = ""
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        null_baccs = np.empty(n_permutations, dtype=np.float64)
        for p in range(n_permutations):
            x_perm = np.copy(x)
            for t in range(ntasks):
                x_perm[t] = x_perm[t][rng.permutation(nsubjects)]

            _, p_bacc = run_cross_validation(
                x_perm,
                method=method,
                zscore_features=zscore_features,
                seed=seed + 1000 + p,
                device=device,
            )
            null_baccs[p] = np.mean(p_bacc)

        p_val = float((np.sum(null_baccs >= mean_bacc) + 1) / (n_permutations + 1))
        perm_str = f" | perm p={p_val:.4g} (null={np.mean(null_baccs):.4f})"

    log.info(
        "Bal. Acc.: %.4f +/- %.4f (95%% CI [%.4f, %.4f])%s",
        mean_bacc,
        std_bacc,
        ci_lo,
        ci_hi,
        perm_str,
    )

    result = ClassificationResult(
        label=ds.label,
        nfeatures=nfeatures,
        nsubjects=nsubjects,
        method=method,
        mean_bacc=mean_bacc,
        std_bacc=std_bacc,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        p_val=p_val,
    )
    return result, fold_bacc, ds.class_names


# {{{ plots


def plot_task_accuracy(
    basename: pathlib.Path,
    fold_bacc: Sequence[float],
    class_names: tuple[str, ...],
    *,
    method: str,
    nsubjects: int,
    overwrite: bool = False,
) -> None:
    """Plot leave-one-condition-out fingerprinting accuracy per held-out task."""
    ntasks = len(class_names)
    xpos = np.arange(ntasks)
    chance = 1.0 / nsubjects
    mean_bacc = float(np.mean(fold_bacc))

    with figure(
        basename.with_stem(f"{basename.stem}-fingerprinting-{method}"),
        figsize=(8, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.bar(xpos, fold_bacc, color="darkcyan", width=0.55)
        ax.axhline(
            chance,
            color="crimson",
            ls="--",
            lw=1.5,
            label=rf"Chance ({chance:.3f})",
        )
        ax.axhline(
            mean_bacc,
            color="black",
            ls=":",
            lw=1.5,
            label=rf"Mean ({mean_bacc:.3f})",
        )
        ax.set_xticks(xpos)
        ax.set_xticklabels(list(class_names), rotation=45, ha="right")
        ax.set_xlabel("Held-out condition")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim((0.0, 1.05))
        ax.set_title(rf"Subject fingerprinting ({method.upper()})")
        ax.legend(fontsize="small")


# }}}


# {{{ table


def _format_score(res: ClassificationResult, *, underline: bool = False) -> str:
    score = f"${res.mean_bacc:.3f} \\pm {res.std_bacc:.3f}$"
    return f"\\underline{{{score}}}" if underline else score


def _format_ci(res: ClassificationResult) -> str:
    return f"$[{res.ci_lo:.3f}, {res.ci_hi:.3f}]$"


def format_latex_table(
    rows: Sequence[tuple[ClassificationResult, ClassificationResult]],
) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"\textbf{Feature family} & \textbf{Size}"
            r" & \textbf{LDA} & \textbf{95\% CI}"
            r" & \textbf{MLP} & \textbf{95\% CI} \\"
        ),
        r"\midrule",
    ]
    if rows:
        lda_min = min(res_lda.mean_bacc for res_lda, _ in rows)
        mlp_min = min(res_mlp.mean_bacc for _, res_mlp in rows)

        for res_lda, res_mlp in rows:
            underline_lda = res_lda.mean_bacc == lda_min
            underline_mlp = res_mlp.mean_bacc == mlp_min
            lda_score = _format_score(res_lda, underline=underline_lda)
            mlp_score = _format_score(res_mlp, underline=underline_mlp)

            lines.append(
                f"{res_lda.label} & {res_lda.nfeatures:,} & {lda_score} & "
                f"{_format_ci(res_lda)} & {mlp_score} & {_format_ci(res_mlp)}"
                r" \\"
            )

        chance = 1.0 / rows[0][0].nsubjects
        lines.extend([
            r"\midrule",
            f"Chance & -- & ${chance:.3f}$ & -- & ${chance:.3f}$ & -- \\\\",
        ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return "\n".join(lines)


# }}}


def main(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod = "lda",
    zscore_features: bool = True,
    n_permutations: int = 0,
    seed: int = 42,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    device = torch.get_default_device()
    log.info("Device: %s", device)

    try:
        res, fold_bacc, class_names = evaluate_dataset(
            filename,
            method=method,
            zscore_features=zscore_features,
            n_permutations=n_permutations,
            seed=seed,
            device=device,
        )
    except Exception as exc:
        log.error("Failed to evaluate '%s': %s", filename, exc)
        return 1

    suffix = "-zscore" if zscore_features else ""
    basename = (
        filename.with_stem(f"{filename.stem}{suffix}").with_suffix("")
        if outfile is None
        else outfile.with_suffix("")
    )
    plot_task_accuracy(
        basename,
        fold_bacc,
        class_names,
        method=method,
        nsubjects=res.nsubjects,
        overwrite=overwrite,
    )

    return 0


def summary(
    filenames: Sequence[pathlib.Path],
    *,
    zscore_features: bool = True,
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
            res_lda, _, _ = evaluate_dataset(
                filename,
                method="lda",
                zscore_features=zscore_features,
                n_permutations=n_permutations,
                seed=seed,
                device=device,
            )
            res_mlp, _, _ = evaluate_dataset(
                filename,
                method="mlp",
                zscore_features=zscore_features,
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
        "--n-permutations",
        type=int,
        default=0,
        help="Number of label permutations (0 to skip)",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutations and MLP initialization",
    )
    common.add_argument(
        "--no-zscore",
        action="store_false",
        dest="zscore",
        help="Disable z-scoring each feature on the training conditions",
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
        help="Evaluate a single feature dataset and plot per-task accuracy",
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
        torch.set_float32_matmul_precision("high")

        if args.command == "run":
            raise SystemExit(
                main(
                    args.filename,
                    outfile=args.outfile,
                    method=args.method,
                    zscore_features=args.zscore,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    overwrite=args.force,
                )
            )
        elif args.command == "summary":
            raise SystemExit(
                summary(
                    args.filenames,
                    zscore_features=args.zscore,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise ValueError(f"unsupported command: {args.command}")

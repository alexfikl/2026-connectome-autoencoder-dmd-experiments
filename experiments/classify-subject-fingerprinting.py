# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import get_args

import numpy as np
import torch

import commonlib  # ruff: ignore[unused-import]
from classifylib import (
    ClassificationMethod,
    ClassificationResult,
    bootstrap_identification_ci,
    format_latex_table,
    load_feature_datasets,
    run_leave_one_condition_out_cv,
    save_summary,
    select_feature_group,
)

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


def evaluate_dataset(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod,
    zscore_features: bool,
    n_permutations: int,
    seed: int,
    device: torch.device | str,
    group: int | str | None = None,
) -> tuple[
    ClassificationResult,
    list[float],
    tuple[str, ...],
]:
    ds = load_feature_datasets(filename)
    if group is not None:
        ds = select_feature_group(ds, group)
    x = ds.to_tensor()
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

    oof_preds, fold_bacc = run_leave_one_condition_out_cv(
        x,
        method=method,
        zscore_features=zscore_features,
        seed=seed,
        device=device,
    )

    mean_bacc = float(np.mean(fold_bacc))
    std_bacc = float(np.std(fold_bacc))
    ci_lo, ci_hi = bootstrap_identification_ci(oof_preds, seed=seed)

    p_val: float | None = None
    perm_str = ""
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        null_baccs = np.empty(n_permutations, dtype=np.float64)
        for p in range(n_permutations):
            x_perm = np.copy(x)
            for t in range(ntasks):
                x_perm[t] = x_perm[t][rng.permutation(nsubjects)]

            _, p_bacc = run_leave_one_condition_out_cv(
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
        nclasses=nsubjects,
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

    from nneuroutil.visualization import figure

    with figure(
        basename.with_stem(f"{basename.stem}-fingerprinting-{method}"),
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


def main(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod = "lda",
    zscore_features: bool = True,
    n_permutations: int = 0,
    seed: int = 42,
    group: int | str | None = None,
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
            group=group,
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
        nsubjects=res.nclasses,
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

    print(format_latex_table(rows))

    if outfile is not None and rows:
        if outfile.exists() and not overwrite:
            log.error("Output file '%s' exists (use --force to overwrite).", outfile)
            return 1

        save_summary(outfile, rows)
        log.info("Wrote summary arrays to '%s'.", outfile)

    return 0


if __name__ == "__main__":
    import argparse

    from nneuroutil.helpers import FuzzyChoiceAction
    from nneuroutil.interop.torch import available_device_names, torch_default_device

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
        "--group",
        type=str,
        default=None,
        help="Evaluate a single feature group (0-based index or description substring)",
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
        help="Optional output .npz file saving summary arrays for plotting",
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
                    group=args.group,
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

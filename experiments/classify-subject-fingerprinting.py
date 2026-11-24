# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, get_args

import numpy as np
import torch

import commonlib  # ruff: ignore[unused-import]
from classifylib import (
    ClassificationMethod,
    ClassificationResult,
    classify_fingerprinting,
    format_latex_table,
    load_feature_datasets,
    save_summary,
    select_feature_group,
)
from nneuroutil.typing import Array1D

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


# {{{ plots


def plot_task_accuracy(
    basename: pathlib.Path,
    fold_bacc: Array1D[np.floating[Any]],
    class_names: tuple[str, ...],
    *,
    nsubjects: int,
    method: ClassificationMethod = "mlp",
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
        ax.legend()


# }}}


# {{{ main


def main(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod = "mlp",
    zscore_features: bool = True,
    seed: int = 42,
    group: int | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    device = torch.get_default_device()
    log.info("Device: %s", device)

    suffix = "-zscore" if zscore_features else ""
    basename = (
        filename.with_stem(f"{filename.stem}{suffix}").with_suffix("")
        if outfile is None
        else outfile.with_suffix("")
    )

    try:
        ds = load_feature_datasets(filename)
    except Exception as exc:
        log.error("Failed to load '%s': %s", filename, exc)
        return 1

    if group is not None:
        try:
            ds = select_feature_group(ds, group)
        except Exception as exc:
            log.error("Failed to select group %d from '%s': %s", group, filename, exc)
            return 1

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
    result, cv = classify_fingerprinting(
        x,
        label=ds.label,
        method=method,
        zscore_features=zscore_features,
        seed=seed,
        device=device,
    )
    log.info(
        "Bal. Acc.: %.4f +/- %.4f (95%% CI [%.4f, %.4f])",
        result.mean_bacc,
        result.std_bacc,
        result.ci_lo,
        result.ci_hi,
    )

    fold_bacc = cv.fold_bacc
    class_names = ds.class_names

    plot_task_accuracy(
        basename,
        fold_bacc,
        class_names,
        nsubjects=result.nclasses,
        method=method,
        overwrite=overwrite,
    )

    return 0


def summary(
    filenames: Sequence[pathlib.Path],
    *,
    zscore_features: bool = True,
    seed: int = 42,
    outfile: pathlib.Path | None = None,
    methods: tuple[ClassificationMethod, ...] = ("lda", "mlp"),
    overwrite: bool = False,
) -> int:
    device = torch.get_default_device()
    log.info("Device: %s", device)

    rows: list[tuple[ClassificationResult, ...]] = []
    for filename in filenames:
        ds = load_feature_datasets(filename)
        x = ds.to_tensor()
        ntasks, nsubjects, nfeatures = x.shape

        results = []
        for method in methods:
            log.info(
                "Evaluating '%s' (%s, z=%s): %d tasks x %d subjects, %d features.",
                filename.stem,
                method.upper(),
                zscore_features,
                ntasks,
                nsubjects,
                nfeatures,
            )
            result, _ = classify_fingerprinting(
                x,
                label=ds.label,
                method=method,
                zscore_features=zscore_features,
                seed=seed,
                device=device,
            )
            log.info(
                "Bal. Acc.: %.4f +/- %.4f (95%% CI [%.4f, %.4f])",
                result.mean_bacc,
                result.std_bacc,
                result.ci_lo,
                result.ci_hi,
            )
            results.append(result)

        rows.append(tuple(results))

    print(format_latex_table(rows))

    if outfile is not None and rows:
        if outfile.exists() and not overwrite:
            log.error("Output file '%s' exists (use --force to overwrite).", outfile)
            return 1

        save_summary(outfile, rows)
        log.info("Wrote summary arrays to '%s'.", outfile)

    return 0


# }}}


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
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstraps and MLP initialization",
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
        default="mlp",
        help="The method used to do the classification",
    )
    run_parser.add_argument(
        "--group",
        type=int,
        default=None,
        help="Evaluate a single feature group by its 0-based index",
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
        help="Evaluate multiple feature datasets and emit a LaTeX table",
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
                    seed=args.seed,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise ValueError(f"unsupported command: {args.command}")

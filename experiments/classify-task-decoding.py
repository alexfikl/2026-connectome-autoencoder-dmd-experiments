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
    classify_task_decoding,
    flatten_feature_datasets,
    format_latex_table,
    load_feature_datasets,
    save_summary,
    select_feature_group,
)
from nneuroutil.helpers import FuzzyChoiceAction
from nneuroutil.interop.torch import available_device_names, torch_default_device
from nneuroutil.typing import Array2D
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.classify_task_decoding")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Classify tasks using features computed by make-features.py

Examples:

    > {SCRIPT_PATH.name} run --method lda dataset-features-matrix.npz
    > {SCRIPT_PATH.name} summary --binary data/dataset-features-*.npz
"""


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
    """Plot the row-normalized confusion matrix (%) of out-of-fold predictions."""
    nclasses = len(class_names)
    row_sums = np.sum(cm, axis=1, keepdims=True)
    cm_pct = 100.0 * np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0
    )

    with figure(
        basename.with_stem(f"{basename.stem}-confusion-{method}"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.grid(visible=False)

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=100.0)
        ax.set_xticks(range(nclasses))
        ax.set_yticks(range(nclasses))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=16)
        ax.set_yticklabels(class_names, fontsize=16)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i, j in np.ndindex(cm_pct.shape):
            if cm[i, j] == 0:
                continue

            color = "white" if cm_pct[i, j] > 50.0 else "black"
            ax.text(
                j,
                i,
                f"{cm_pct[i, j]:.1f}",
                ha="center",
                va="center",
                color=color,
            )

        fig.colorbar(im, cax=make_colorbar_axes(ax), label="Percentage (\\%)")


# }}}


# {{{ main


def main(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod = "mlp",
    binary: bool = False,
    zscore_features: bool = True,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    group: int | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    device = torch.get_default_device()
    log.info("Device: %s", device)

    suffix = "-binary" if binary else "-multiclass"
    if zscore_features:
        suffix += "-zscore"
    basename = (
        filename.with_stem(f"{filename.stem}{suffix}").with_suffix("")
        if outfile is None
        else outfile.with_suffix("")
    )

    try:
        ds = load_feature_datasets(filename, binary=binary)
    except Exception as exc:
        log.error("Failed to load '%s': %s", filename, exc)
        return 1

    if group is not None:
        try:
            ds = select_feature_group(ds, group)
        except Exception as exc:
            log.error("Failed to select group %d from '%s': %s", group, filename, exc)
            return 1

    x, y, subject_ids = flatten_feature_datasets(ds)
    class_names = ds.class_names
    unique_subjects = np.unique(subject_ids)

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

    result, cv = classify_task_decoding(
        x,
        y,
        subject_ids,
        label=ds.label,
        method=method,
        zscore_features=zscore_features,
        n_splits=n_splits,
        n_repeats=n_repeats,
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

    y_rep = np.tile(y, n_repeats)
    cm = confusion_matrix(
        y_rep, np.ravel(cv.oof_preds), labels=list(range(result.nclasses))
    )
    overall_bacc = balanced_accuracy_score(y_rep, np.ravel(cv.oof_preds))

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
    seed: int = 42,
    outfile: pathlib.Path | None = None,
    methods: tuple[ClassificationMethod, ...] = ("lda", "mlp"),
    overwrite: bool = False,
) -> int:
    device = torch.get_default_device()
    log.info("Device: %s", device)

    rows: list[tuple[ClassificationResult, ...]] = []
    for filename in filenames:
        ds = load_feature_datasets(filename, binary=binary)
        x, y, subject_ids = flatten_feature_datasets(ds)
        unique_subjects = np.unique(subject_ids)

        results = []
        for method in methods:
            log.info(
                "Evaluating '%s' (%s, %s, z=%s): %d samples (%d subjects), "
                "%d features.",
                filename.stem,
                "binary" if binary else "multiclass",
                method.upper(),
                zscore_features,
                x.shape[0],
                unique_subjects.size,
                x.shape[1],
            )
            result, _ = classify_task_decoding(
                x,
                y,
                subject_ids,
                label=ds.label,
                method=method,
                zscore_features=zscore_features,
                n_splits=n_splits,
                n_repeats=n_repeats,
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splits, bootstraps, and MLP initialization",
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
                    seed=args.seed,
                    group=args.group,
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
                    seed=args.seed,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise ValueError(f"unsupported command: {args.command}")

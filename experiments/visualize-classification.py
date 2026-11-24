# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, Literal, NamedTuple, get_args

import numpy as np

import commonlib  # ruff: ignore[unused-import]
from nneuroutil.typing import Array1D, Array2D
from nneuroutil.visualization import figure

log = logging.getLogger("lib.visualize_classification")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Plot task-decoding versus subject-fingerprinting accuracy across feature families.

Reads the ``.npz`` summary files produced by ``classify-task-decoding.py summary -o``
and ``classify-subject-fingerprinting.py summary -o``.

Example:

    > {SCRIPT_PATH.name} \\
        --task data/stage2-task-summary.npz \\
        --fingerprint data/stage2-fingerprint-summary.npz
"""

ClassificationMethod = Literal["lda", "mlp"]

_LABEL_TO_SHORT: dict[str, str] = {
    "matrix scale": "Scale",
    "raw matrix": "A",
    "eigenspectrum": "B",
    "global graph": "C",
    "local graph": "D",
    "raw fractal": "E",
    "fourier": "F",
    "landmarks": "G",
    "geometry": "G",
    "dmdc": "H",
    "ae-dmd": "I",
}


def _short_label(label: str) -> str:
    key = label.strip().lower()
    if key in _LABEL_TO_SHORT:
        return _LABEL_TO_SHORT[key]
    if len(key) >= 2 and key[0] in "abcdefghi" and key[1] == ".":
        return key[0].upper()
    return label


def _family_group(short: str) -> str:
    if short == "Scale":
        return "Scale baseline"
    if short in {"A", "D"}:
        return "Node-indexed (A, D)"
    if short in {"B", "C"}:
        return "Node-invariant (B, C)"
    if short in {"E", "F", "G"}:
        return "Equi-M geometry (E--G)"
    return "Dynamical (H, I)"


# {{{ data


class ClassificationSummary(NamedTuple):
    labels: tuple[str, ...]
    groups: tuple[str, ...]
    task_mean: Array1D[np.floating[Any]]
    task_err: Array2D[np.floating[Any]]
    task_chance: float
    fp_mean: Array1D[np.floating[Any]]
    fp_err: Array2D[np.floating[Any]]
    fp_chance: float


def load_summary(
    task_npz: pathlib.Path,
    fp_npz: pathlib.Path,
    *,
    method: ClassificationMethod = "mlp",
) -> ClassificationSummary:
    """Load matched task-decoding and fingerprinting summaries from ``.npz`` files."""
    with np.load(task_npz, allow_pickle=False) as dt:
        task_labels = [str(x) for x in dt["labels"]]
        task_chance = float(dt["chance"])
        task_mean = np.asarray(dt[f"{method}_mean"], dtype=np.float64)
        task_lo = np.asarray(dt[f"{method}_ci_lo"], dtype=np.float64)
        task_hi = np.asarray(dt[f"{method}_ci_hi"], dtype=np.float64)

    with np.load(fp_npz, allow_pickle=False) as df:
        fp_labels = [str(x) for x in df["labels"]]
        fp_chance = float(df["chance"])
        fp_mean_all = np.asarray(df[f"{method}_mean"], dtype=np.float64)
        fp_lo_all = np.asarray(df[f"{method}_ci_lo"], dtype=np.float64)
        fp_hi_all = np.asarray(df[f"{method}_ci_hi"], dtype=np.float64)

    fp_index = {lbl: idx for idx, lbl in enumerate(fp_labels)}
    missing = [lbl for lbl in task_labels if lbl not in fp_index]
    if missing:
        raise ValueError(f"Labels missing from '{fp_npz}': {missing}")

    order = [fp_index[lbl] for lbl in task_labels]
    fp_mean = fp_mean_all[order]
    fp_lo = fp_lo_all[order]
    fp_hi = fp_hi_all[order]

    shorts = tuple(_short_label(lbl) for lbl in task_labels)
    groups = tuple(_family_group(s) for s in shorts)

    task_err = np.maximum(
        np.vstack([task_mean - task_lo, task_hi - task_mean]),
        0.0,
    )
    fp_err = np.maximum(
        np.vstack([fp_mean - fp_lo, fp_hi - fp_mean]),
        0.0,
    )

    return ClassificationSummary(
        labels=shorts,
        groups=groups,
        task_mean=task_mean,
        task_err=task_err,
        task_chance=task_chance,
        fp_mean=fp_mean,
        fp_err=fp_err,
        fp_chance=fp_chance,
    )


# }}}


# {{{ plot


def plot_classification(
    outfile: pathlib.Path,
    data: ClassificationSummary,
    *,
    method: str,
    overwrite: bool = False,
) -> None:
    """Plot task decoding versus fingerprinting balanced accuracy."""
    with figure(
        outfile,
        overwrite=overwrite,
    ) as fig:
        (ax,) = fig.axes

        ax.axvline(
            data.task_chance,
            color="k",
            ls="--",
            lw=1.2,
            label="Chance",
        )
        ax.axhline(data.fp_chance, color="k", ls="--", lw=1.2)

        group_arr = np.asarray(data.groups)
        for group in dict.fromkeys(data.groups):
            mask = group_arr == group
            ax.errorbar(
                data.task_mean[mask],
                data.fp_mean[mask],
                xerr=data.task_err[:, mask],
                yerr=data.fp_err[:, mask],
                fmt="o",
                capsize=3,
                linestyle="none",
                label=group,
            )

        for lbl, x, y in zip(data.labels, data.task_mean, data.fp_mean, strict=True):
            if lbl == "Scale":
                continue

            ax.annotate(
                rf"\textbf{{{lbl}}}",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=20,
            )

        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Task decoding")
        ax.set_ylabel("Subject fingerprinting")
        ax.legend(fontsize=16)


# }}}


# {{{ main


def main(
    task_npz: pathlib.Path,
    fp_npz: pathlib.Path,
    *,
    method: ClassificationMethod = "mlp",
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    try:
        data = load_summary(task_npz, fp_npz, method=method)
    except Exception as exc:
        log.error("Failed to load summary files: %s", exc)
        return 1

    if outfile is None:
        outfile = task_npz.parent / f"classification-scatter-{method}"

    plot_classification(
        outfile.with_suffix(""),
        data,
        method=method,
        overwrite=overwrite,
    )
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
    parser.add_argument(
        "--task",
        type=pathlib.Path,
        required=True,
        help="Summary .npz from classify-task-decoding.py summary -o",
    )
    parser.add_argument(
        "--fingerprint",
        type=pathlib.Path,
        required=True,
        help="Summary .npz from classify-subject-fingerprinting.py summary -o",
    )
    parser.add_argument(
        "--method",
        choices=get_args(ClassificationMethod),
        default="mlp",
        help="Classifier to plot",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Prefix of the output figure path",
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
            args.task,
            args.fingerprint,
            method=args.method,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )

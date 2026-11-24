# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import torch

from nneuroutil.typing import Array1D, Array2D

log = logging.getLogger("lib.classifylib")

ClassificationMethod = Literal["lda", "svm", "mlp"]


class ClassificationResult(NamedTuple):
    label: str
    nfeatures: int
    # NOTE: number of classes (or subjects, for fingerprinting)
    nclasses: int
    method: ClassificationMethod
    mean_bacc: float
    std_bacc: float
    ci_lo: float
    ci_hi: float
    p_val: float | None = None
    mean_f1: float | None = None

    @property
    def chance(self) -> float:
        return 1.0 / self.nclasses


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

        # NOTE: the features are stored as float64, but the MLP is significantly
        # faster in single precision and the resulting accuracy is unchanged
        torch.manual_seed(seed)
        clf = classify_multilayer_perceptron(
            [xi.float() for xi in x_train_list],
            y_train_list,
            alpha=1.0,
            learning_rate_init=1.0e-2,
            max_iter=400,
        )
        return clf.predict(x_test)

    raise ValueError(f"unknown method: {method!r}")


def make_subject_rows(
    groups: Array1D[np.integer[Any]],
) -> list[Array1D[np.integer[Any]]]:
    """Return the row indices of each subject in ``groups``."""
    return [np.flatnonzero(groups == s) for s in np.unique(groups)]


def _bootstrap_subject_ci(
    metric: Callable[[Array1D[np.integer[Any]]], float],
    n_subjects: int,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a 95% bootstrap CI by resampling subjects.

    :arg metric: callable that takes the sampled subject indices and returns
        the score of interest.
    """
    rng = np.random.default_rng(seed)

    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_subjects, size=n_subjects)
        vals[b] = metric(sampled)

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def bootstrap_balanced_accuracy_ci(
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
    n_repeats = oof_preds.shape[0]

    def metric(sampled: Array1D[np.integer[Any]]) -> float:
        idx = np.concatenate([subj_rows[s] for s in sampled])
        return float(
            balanced_accuracy_score(
                np.tile(y_true[idx], n_repeats), np.ravel(oof_preds[:, idx])
            )
        )

    return _bootstrap_subject_ci(
        metric, len(subj_rows), n_bootstrap=n_bootstrap, seed=seed
    )


def bootstrap_identification_ci(
    oof_preds: Array2D[np.integer[Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap CI over subjects for identification accuracy."""
    _ntasks, nsubjects = oof_preds.shape
    correct = oof_preds == np.arange(nsubjects)[None, :]

    def metric(sampled: Array1D[np.integer[Any]]) -> float:
        return float(np.mean(np.take(correct, sampled, axis=1)))

    return _bootstrap_subject_ci(metric, nsubjects, n_bootstrap=n_bootstrap, seed=seed)


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
        # NOTE: underline the lowest score in each method column
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

        chance = rows[0][0].chance
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


# {{{ summary


def save_summary(
    outfile: pathlib.Path,
    rows: Sequence[tuple[ClassificationResult, ClassificationResult]],
) -> None:
    """Save summary arrays consumed by ``visualize-classification.py``."""
    np.savez(
        outfile,
        labels=np.array([r[0].label for r in rows]),
        nfeatures=np.array([r[0].nfeatures for r in rows], dtype=np.int64),
        chance=np.float64(rows[0][0].chance),
        lda_mean=np.array([r[0].mean_bacc for r in rows], dtype=np.float64),
        lda_std=np.array([r[0].std_bacc for r in rows], dtype=np.float64),
        lda_ci_lo=np.array([r[0].ci_lo for r in rows], dtype=np.float64),
        lda_ci_hi=np.array([r[0].ci_hi for r in rows], dtype=np.float64),
        mlp_mean=np.array([r[1].mean_bacc for r in rows], dtype=np.float64),
        mlp_std=np.array([r[1].std_bacc for r in rows], dtype=np.float64),
        mlp_ci_lo=np.array([r[1].ci_lo for r in rows], dtype=np.float64),
        mlp_ci_hi=np.array([r[1].ci_hi for r in rows], dtype=np.float64),
    )


# }}}

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch

from nneuroutil.typing import Array1D, Array2D, Array3D

log = logging.getLogger("lib.classifylib")


SCAN_TYPE_LABELS = {"fmri": "Functional", "mri": "Structural"}

TASK_LABELS = {
    "emotion": "Emotion",
    "gambling": "Gambling",
    "language": "Language",
    "relational": "Relational",
    "rest": "Rest",
    "social": "Social",
    "wm": "WM",
}

ClassificationMethod = Literal["lda", "svm", "mlp"]


# {{{ load_feature_datasets


class FeatureDataset(NamedTuple):
    label: str
    """Label of the feature family, e.g. ``DMDc``."""

    x: tuple[Array2D[np.floating[Any]], ...]
    """One feature matrix of shape ``(nsamples, nfeatures)`` per class."""
    class_names: tuple[str, ...]
    """Class name for each entry of :attr:`x`."""

    starts: Array1D[np.integer[Any]]
    """Feature group boundaries of length ``ngroups + 1``."""
    descrs: tuple[str, ...]
    """Description of each feature group of length ``ngroups``."""
    subject_ids: tuple[Array1D[np.integer[Any]], ...]
    """Subject id for each row of :attr:`x`."""

    def to_tensor(self) -> Array3D[np.floating[Any]]:
        """Stack matched-subject features to ``(ntasks, nsubjects, nfeatures)``."""
        shapes = {x.shape for x in self.x}
        if len(shapes) != 1:
            raise ValueError("expected all classes to have the same shape")

        return np.stack(self.x, axis=0)


def _clean_class_name(name: str) -> str:
    """Reduce a raw class key such as ``task-emotion-fractal`` to a label."""
    if "_" in name:
        name, _ = name.split("_", maxsplit=1)
    else:
        parts = name.split("-")
        name = parts[0] if len(parts) == 2 else parts[1]

    return upper if (upper := name.upper()) == "WM" else name.title()


def load_feature_datasets(
    filename: pathlib.Path,
    *,
    binary: bool = False,
    datakey: str = "datasets",
) -> FeatureDataset:
    """Load a feature file written by ``make-features.py``.

    With ``binary=False`` only the functional classes are returned, one array
    per task. With ``binary=True`` classes are grouped by scan type and
    concatenated, e.g. all functional scans and the structural scans.

    :arg datakey: name of the variable holding the class dictionary.
    """
    if not filename.exists():
        raise FileNotFoundError(f"file does not exist: '{filename}'")

    try:
        data = np.load(filename, allow_pickle=True)
    except Exception:
        raise RuntimeError(f"failed to load data from file: '{filename}'") from None

    if datakey not in data:
        raise ValueError(f"file does not contain '{datakey}' dataset: '{filename}'")

    datasets = data[datakey]
    if datasets.ndim == 0:
        datasets = datasets[()]

    if not isinstance(datasets, dict):
        raise ValueError(f"expected a dictionary in '{datakey}': '{filename}'")

    if len(datasets) <= 1:
        raise ValueError(f"dataset does not have sufficient classes: {len(datasets)}")

    descrs = data["descrs"]
    if descrs.ndim == 0:
        descrs = descrs[()]
    descrs = tuple(str(d) for d in descrs)

    if "label" not in data:
        raise ValueError(f"file does not contain 'label' dataset: '{filename}'")

    label = str(data["label"])

    classes: list[str] = []
    arrays: list[Array2D[np.floating[Any]]] = []
    for name, x in datasets.items():
        if x.ndim != 2:
            raise ValueError(f"expected a 2d array for class '{name}': {x.shape}")

        if arrays and x.shape[1] != arrays[-1].shape[1]:
            raise ValueError(f"class '{name}' has inconsistent feature size: {x.shape}")

        if not np.all(np.isfinite(x)):
            raise ValueError(f"class '{name}' contains NaN or inf values")

        classes.append(name)
        arrays.append(x)

    scan_types = data["scan_types"]
    if scan_types.ndim == 0:
        scan_types = scan_types[()]
    scan_types = tuple(str(scan_types[name]) for name in classes)

    unique_scan_types = sorted(set(scan_types))
    if unique_scan_types != ["fmri", "mri"]:
        raise ValueError(f"invalid scan types: {unique_scan_types}")

    if binary:
        selections = [
            [i for i, name in enumerate(scan_types) if name == scan_type]
            for scan_type in unique_scan_types
        ]
        class_names = [SCAN_TYPE_LABELS[scan_type] for scan_type in unique_scan_types]
    else:
        selections = [[i] for i, name in enumerate(scan_types) if name == "fmri"]
        class_names = [_clean_class_name(classes[indices[0]]) for indices in selections]

    xs: list[Array2D[np.floating[Any]]] = []
    subject_ids: list[Array1D[np.integer[Any]]] = []
    for indices in selections:
        xs.append(np.concatenate([arrays[i] for i in indices], axis=0))
        subject_ids.append(
            np.concatenate([
                np.arange(arrays[i].shape[0], dtype=np.int64) for i in indices
            ])
        )

    return FeatureDataset(
        label=label,
        x=tuple(xs),
        class_names=tuple(class_names),
        starts=data["starts"],
        descrs=tuple(descrs),
        subject_ids=tuple(subject_ids),
    )


# }}}


# {{{ flatten_feature_datasets


def flatten_feature_datasets(
    ds: FeatureDataset,
) -> tuple[
    Array2D[np.floating[Any]],
    Array1D[np.integer[Any]],
    Array1D[np.integer[Any]],
]:
    """Concatenate the per-class arrays into ``(x, labels, subject_ids)``."""
    x = np.concatenate(ds.x, axis=0)
    labels = np.repeat(
        np.arange(len(ds.x), dtype=np.int64),
        [xi.shape[0] for xi in ds.x],
    )
    subject_ids = np.concatenate(ds.subject_ids, axis=0)

    return x, labels, subject_ids


# }}}


# {{{ select_feature_group


def select_feature_group(
    ds: FeatureDataset,
    index: int,
) -> FeatureDataset:
    """Return a view of *ds* restricted to a single feature group.

    :arg index: 0-based group index into ``ds.descrs``.
    """
    ngroups = len(ds.descrs)
    if index < 0:
        index += ngroups
    if not (0 <= index < ngroups):
        raise ValueError(f"feature group index {index!r} out of range [0, {ngroups})")

    start = ds.starts[index]
    end = ds.starts[index + 1]
    descr = ds.descrs[index]

    return FeatureDataset(
        label=f"{ds.label} ({descr})",
        x=tuple(xi[:, start:end] for xi in ds.x),
        class_names=ds.class_names,
        starts=np.array([0, end - start], dtype=ds.starts.dtype),
        descrs=(descr,),
        subject_ids=ds.subject_ids,
    )


# }}}


# {{{ fit_and_predict


class ClassificationResult(NamedTuple):
    """Outcome of one repeated cross-validated classification run."""

    label: str
    """Feature family label, e.g. ``DMDc``."""
    nfeatures: int
    """Number of features seen by the classifier."""
    nclasses: int
    """Number of classes (i.e. tasks or subjects)."""
    method: ClassificationMethod
    """Classifier that produced the predictions."""

    mean_bacc: float
    """Mean balanced accuracy over the CV folds."""
    std_bacc: float
    """Standard deviation of balanced accuracy over the CV folds."""

    ci_lo: float
    """Lower bound of the bootstrap confidence interval for balanced accuracy."""
    ci_hi: float
    """Upper bound of the bootstrap confidence interval for balanced accuracy."""

    @property
    def chance(self) -> float:
        return 1.0 / self.nclasses


def fit_and_predict(
    x_train: list[Array2D[np.floating[Any]]],
    y_train: list[int],
    x_test: Array2D[np.floating[Any]],
    *,
    method: ClassificationMethod = "mlp",
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> Array1D[np.integer[Any]]:
    """Fit classifier on arrays and return predicted labels."""
    x_train = [torch.from_numpy(xi).to(device) for xi in x_train]
    x_test = torch.from_numpy(x_test).to(device)

    if seed is not None:
        torch.manual_seed(seed)

    if method == "lda":
        from nneuroutil.classifier import classify_linear_discriminant_analysis

        clf = classify_linear_discriminant_analysis(x_train, y_train)
        result = clf.predict(x_test, metric="l2")
    elif method == "svm":
        from nneuroutil.classifier import classify_support_vector_machine

        clf = classify_support_vector_machine(x_train, y_train)
        result = clf.predict(x_test)
    elif method == "mlp":
        from torchlib import classify_multilayer_perceptron

        # NOTE: the features are stored as float64, but the MLP is significantly
        # faster in single precision and the resulting accuracy is unchanged
        clf = classify_multilayer_perceptron(
            [xi.float() for xi in x_train],  # ty: ignore[missing-attribute]
            y_train,
            alpha=1.0,
            learning_rate_init=1.0e-2,
            max_iter=400,
        )
        result = clf.predict(x_test)
    else:
        raise ValueError(f"unknown method: {method!r}")

    result = result.detach().cpu().numpy()  # ty: ignore[missing-union-attribute]
    return cast(Array1D[np.integer[Any]], result)


# }}}


# {{{ run_subject_grouped_cv


class GroupedCVResult(NamedTuple):
    oof_preds: Array2D[np.integer[Any]]
    """Out-of-fold predictions of shape ``(n_repeats, nsamples)``."""
    fold_bacc: Array1D[np.floating[Any]]
    """Balanced accuracy of each fold, of length ``n_splits * n_repeats``."""


def run_subject_grouped_cv(
    x: Array2D[np.floating[Any]],
    y: Array1D[np.integer[Any]],
    subject_ids: Array1D[np.integer[Any]],
    *,
    nclasses: int,
    method: ClassificationMethod = "mlp",
    zscore_features: bool = True,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> GroupedCVResult:
    """Run repeated subject-grouped K-fold CV on ``(nsamples, nfeatures)``."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    unique_subjects = np.unique(subject_ids)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    nsamples = x.shape[0]
    nfolds = n_splits * n_repeats
    oof_preds = np.empty((n_repeats, nsamples), dtype=np.int64)
    fold_bacc = np.empty(nfolds, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(unique_subjects)):
        rep = fold_idx // n_splits
        train_mask = np.isin(subject_ids, unique_subjects[train_idx])
        test_mask = np.isin(subject_ids, unique_subjects[test_idx])

        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[test_mask]
        y_test = y[test_mask]

        if zscore_features:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        y_pred = fit_and_predict(
            [x_train[y_train == c] for c in range(nclasses)],
            list(range(nclasses)),
            x_test,
            method=method,
            seed=(seed + fold_idx) if seed is not None else seed,
            device=device,
        )

        oof_preds[rep, test_mask] = y_pred
        fold_bacc[fold_idx] = balanced_accuracy_score(y_test, y_pred)

    return GroupedCVResult(oof_preds=oof_preds, fold_bacc=fold_bacc)


# }}}


# {{{ run_leave_one_condition_out_cv


def run_leave_one_condition_out_cv(
    x: Array3D[np.floating[Any]],
    *,
    method: ClassificationMethod = "mlp",
    zscore_features: bool = True,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> GroupedCVResult:
    """Run leave-one-condition-out CV on ``(ntasks, nsubjects, nfeatures)``."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    ntasks, nsubjects, nfeatures = x.shape
    y_train = list(range(nsubjects))
    y_test = np.arange(nsubjects, dtype=np.int64)

    oof_preds = np.empty((ntasks, nsubjects), dtype=np.int64)
    fold_bacc = np.empty(ntasks, dtype=np.float64)

    for t in range(ntasks):
        x_train = np.delete(x, t, axis=0)
        x_test = x[t]

        if zscore_features:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(np.reshape(x_train, (-1, nfeatures)))
            x_train = np.reshape(x_train, (ntasks - 1, nsubjects, nfeatures))
            x_test = scaler.transform(x_test)

        y_pred = fit_and_predict(
            [x_train[:, s, :] for s in range(nsubjects)],
            y_train,
            x_test,
            method=method,
            seed=(seed + t) if seed is not None else seed,
            device=device,
        )

        oof_preds[t] = y_pred
        fold_bacc[t] = balanced_accuracy_score(y_test, y_pred)

    return GroupedCVResult(oof_preds=oof_preds, fold_bacc=fold_bacc)


# }}}


# {{{ make_subject_index


class SubjectIndex(NamedTuple):
    unique: Array1D[np.integer[Any]]
    idx: Array1D[np.integer[Any]]
    rows: list[Array1D[np.integer[Any]]]


def make_subject_index(subject_ids: Array1D[np.integer[Any]]) -> SubjectIndex:
    """Map each sample to its subject and collect the rows of each subject."""
    unique, idx = np.unique(subject_ids, return_inverse=True)
    return SubjectIndex(
        unique=unique,
        idx=idx,
        rows=[np.flatnonzero(idx == s) for s in range(unique.size)],
    )


# }}}


# {{{ bootstrap


def bootstrap_balanced_accuracy_ci(
    y_true: Array1D[np.integer[Any]],
    oof_preds: Array2D[np.integer[Any]],
    subject_ids: Array1D[np.integer[Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap CI over subjects for balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score

    subj_rows = make_subject_index(subject_ids).rows
    n_repeats = oof_preds.shape[0]
    rng = np.random.default_rng(seed)

    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, len(subj_rows), size=len(subj_rows))
        idx = np.concatenate([subj_rows[s] for s in sampled])
        vals[b] = balanced_accuracy_score(
            np.tile(y_true[idx], n_repeats), np.ravel(oof_preds[:, idx])
        )

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def bootstrap_identification_ci(
    oof_preds: Array2D[np.integer[Any]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap CI over subjects for identification accuracy."""
    _ntasks, nsubjects = oof_preds.shape
    correct = oof_preds == np.arange(nsubjects)[None, :]
    rng = np.random.default_rng(seed)

    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, nsubjects, size=nsubjects)
        vals[b] = np.mean(np.take(correct, sampled, axis=1))

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


# }}}


# {{{ regression


class RegressionResult(NamedTuple):
    name: str
    target_name: str
    r2_mean: float
    r2_ci_lo: float
    r2_ci_hi: float
    y_true: Array1D[np.floating[Any]]
    y_pred: Array1D[np.floating[Any]]
    is_structural: Array1D[np.bool]


def run_ridge_cv(
    x: Array2D[np.floating[Any]],
    y: Array1D[np.floating[Any]],
    subject_index: SubjectIndex,
    is_structural: Array1D[np.bool],
    *,
    name: str,
    target_name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> RegressionResult:
    """Run repeated subject-grouped K-fold Ridge regression."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    alphas = np.logspace(-3, 5, 25)

    nsamples = x.shape[0]
    oof_pred = np.empty((n_repeats, nsamples), dtype=y.dtype)
    fold_r2 = np.empty(n_splits * n_repeats, dtype=x.dtype)

    # go through all the folds: fit + compute the R2 scores
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(subject_index.unique)):
        rep = fold_idx // n_splits
        train_mask = np.isin(subject_index.idx, train_idx)
        test_mask = np.isin(subject_index.idx, test_idx)

        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x[train_mask])
        x_te_s = scaler.transform(x[test_mask])

        reg = RidgeCV(alphas=alphas, cv=min(5, train_idx.size))
        reg.fit(x_tr_s, y[train_mask])
        pred = reg.predict(x_te_s)

        oof_pred[rep, test_mask] = pred
        fold_r2[fold_idx] = r2_score(y[test_mask], pred)

    y_pred_mean = np.mean(oof_pred, axis=0)

    # compute confidence intervals
    rng = np.random.default_rng(seed)
    boot_r2 = np.empty(n_bootstrap, dtype=x.dtype)

    for b in range(n_bootstrap):
        sampled = rng.integers(0, len(subject_index.rows), size=len(subject_index.rows))
        idx = np.concatenate([subject_index.rows[s] for s in sampled])
        boot_r2[b] = r2_score(y[idx], y_pred_mean[idx])

    ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])

    return RegressionResult(
        name=name,
        target_name=target_name,
        r2_mean=np.mean(fold_r2),
        r2_ci_lo=ci_lo,
        r2_ci_hi=ci_hi,
        y_true=y,
        y_pred=y_pred_mean,
        is_structural=is_structural,
    )


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

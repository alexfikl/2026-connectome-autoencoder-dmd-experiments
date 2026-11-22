# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np

from commonlib import FeatureDataset, load_feature_datasets
from nneuroutil.typing import Array1D, Array2D, Array3D
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.classify_subject_fingerprinting")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Evaluate subject fingerprinting (identification) from feature datasets.

For each feature file, the features are reshaped to ``(ntasks, nsubjects,
nfeatures)``, assuming matched subjects in the same order in every class. We
then measure how well a held-out task can identify the correct subject:

1. Leave-one-task-out nearest-neighbour identification (rank-k accuracy).
2. Task-by-task identifiability matrix (reference task vs. test task).
3. Leave-one-task-out subject classification (balanced accuracy).
4. Split-half reliability (self vs. other subject similarity).
5. Permutation null for the rank-1 identification accuracy.

Example:

    > {SCRIPT_PATH.name} dataset-features-matrix.npz dataset-features-geometry.npz
"""

SimilarityMetric = Literal["correlation", "cosine", "euclidean"]
ClassificationMethod = Literal["lda", "ridge"]

EPS = 1.0e-12


# {{{ helpers


def clean_label(name: str) -> str:
    """Shorten a dataset name for clean plotting."""
    label = name.split("_", maxsplit=1)[0]
    return label.removeprefix("task-").removesuffix("-fractal")


def to_tensor(ds: FeatureDataset) -> Array3D[np.floating[Any]]:
    """Stack matched-subject features to ``(ntasks, nsubjects, nfeatures)``."""
    shapes = {np.asarray(x).shape for x in ds.x}
    if len(shapes) != 1:
        raise ValueError("expected all classes to have the same shape")

    return np.stack([np.asarray(x) for x in ds.x], axis=0)


def _normalize_rows(x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, EPS)


def similarity(
    a: Array2D[np.floating[Any]],
    b: Array2D[np.floating[Any]],
    metric: SimilarityMetric,
) -> Array2D[np.floating[Any]]:
    """Pairwise similarity of two ``(nsamples, nfeatures)`` blocks.

    Higher values mean more similar. For ``euclidean`` we return the negated
    distance so that the same ranking code can be reused.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if metric == "correlation":
        a = _normalize_rows(a - np.mean(a, axis=-1, keepdims=True))
        b = _normalize_rows(b - np.mean(b, axis=-1, keepdims=True))
        return a @ b.T  # ty: ignore[missing-attribute]

    if metric == "cosine":
        return _normalize_rows(a) @ _normalize_rows(b).T  # ty: ignore[missing-attribute]

    if metric == "euclidean":
        from scipy.spatial.distance import cdist

        return -cdist(a, b, metric="euclidean")

    raise ValueError(f"unknown similarity metric: {metric!r}")


def _ranks(sim: Array2D[np.floating[Any]]) -> Array1D[np.integer[Any]]:
    """Zero-based rank of the matched row/column in a square similarity matrix."""
    if sim.shape[0] != sim.shape[1]:
        raise ValueError(f"expected a square similarity matrix, got {sim.shape}")

    order = np.argsort(-sim, axis=1)
    target = np.arange(sim.shape[0])[:, None]
    return np.argmax(order == target, axis=1)


# }}}


# {{{ identification


class IdentificationResults(NamedTuple):
    ranks: Array2D[np.integer[Any]]
    """Zero-based rank of the correct subject for each (task, subject)."""
    rank_curve: Array1D[np.floating[Any]]
    """Fraction with rank ``< k`` for ``k = 1, ..., nsubjects``."""

    @property
    def rank1(self) -> float:
        return float(np.mean(self.ranks == 0))

    @property
    def rank5(self) -> float:
        return np.mean(self.ranks < 5)  # ty: ignore[unsupported-operator]

    @property
    def mean_rank(self) -> float:
        """One-based mean rank (chance is ``(nsubjects + 1) / 2``)."""
        return float(1.0 + np.mean(self.ranks))


def compute_leave_one_task_out(
    x: Array3D[np.floating[Any]], metric: SimilarityMetric
) -> IdentificationResults:
    """Identify subjects from a held-out task using all other tasks."""
    ntasks, nsubjects, _ = x.shape
    ranks = np.zeros((ntasks, nsubjects), dtype=int)

    for t in range(ntasks):
        reference = np.delete(x, t, axis=0).mean(axis=0)
        ranks[t] = _ranks(similarity(x[t], reference, metric))

    flat = ranks.ravel()
    rank_curve = np.array([np.mean(flat < k) for k in range(1, nsubjects + 1)])

    return IdentificationResults(ranks=ranks, rank_curve=rank_curve)


def compute_identifiability_matrix(
    x: Array3D[np.floating[Any]], metric: SimilarityMetric
) -> Array2D[np.floating[Any]]:
    """Rank-1 identification using one reference task against one test task."""
    ntasks = x.shape[0]
    identifiability = np.full((ntasks, ntasks), np.nan)

    for i in range(ntasks):
        for j in range(ntasks):
            if i == j:
                continue

            identifiability[i, j] = np.mean(_ranks(similarity(x[j], x[i], metric)) == 0)

    return identifiability


def compute_cross_validated_classification(
    x: Array3D[np.floating[Any]],
    *,
    method: ClassificationMethod = "lda",
) -> float:
    """Leave-one-task-out subject classification (balanced accuracy)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ntasks, nsubjects, nfeatures = x.shape
    x_flat = np.reshape(x, (ntasks * nsubjects, nfeatures))
    groups = np.repeat(np.arange(ntasks), nsubjects)
    labels = np.tile(np.arange(nsubjects), ntasks)

    if method == "lda":
        # NOTE: 'lsqr' with automatic shrinkage handles the nfeatures >> nsamples
        # regime without an explicit PCA pre-reduction
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif method == "ridge":
        clf = RidgeClassifier(alpha=1.0)
    else:
        raise ValueError(f"unknown classification method: {method!r}")

    model = make_pipeline(StandardScaler(), clf)

    scores = []
    for train, test in LeaveOneGroupOut().split(x_flat, labels, groups):
        model.fit(x_flat[train], labels[train])
        predictions = model.predict(x_flat[test])
        scores.append(balanced_accuracy_score(labels[test], predictions))

    return float(np.mean(scores))


def compute_split_half_reliability(
    x: Array3D[np.floating[Any]],
    metric: SimilarityMetric,
    *,
    random_state: int = 42,
) -> tuple[Array1D[np.floating[Any]], Array1D[np.floating[Any]]]:
    """Per-subject self vs. mean-other similarity across two task halves."""
    ntasks, nsubjects, _ = x.shape
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(ntasks)
    half = max(1, ntasks // 2)

    first = x[idx[:half]].mean(axis=0)
    second = x[idx[half:]].mean(axis=0)
    sim = similarity(first, second, metric)

    self = np.diag(sim).copy()
    others = (np.sum(sim, axis=1) - self) / (nsubjects - 1)

    return self, others


def compute_permutation_null(
    x: Array3D[np.floating[Any]],
    metric: SimilarityMetric,
    *,
    nperm: int = 200,
    random_state: int = 42,
) -> Array1D[np.floating[Any]]:
    """Null rank-1 identification accuracy with shuffled subject labels."""
    rng = np.random.default_rng(random_state)
    ntasks, nsubjects, _ = x.shape
    null = np.empty(nperm)

    for p in range(nperm):
        permuted = np.copy(x)
        for t in range(ntasks):
            permuted[t] = permuted[t][rng.permutation(nsubjects)]

        null[p] = compute_leave_one_task_out(permuted, metric).rank1

    return null


# }}}


# {{{ plots


def plot_rank_curve(
    basename: pathlib.Path,
    results: IdentificationResults,
    *,
    label: str = "",
    overwrite: bool = False,
) -> None:
    nsubjects = results.ranks.shape[1]
    ranks = np.arange(1, nsubjects + 1)

    with figure(
        basename.with_stem(f"{basename.stem}-rank-curve"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ax.plot(
            ranks,
            results.rank_curve,
            "o-",
            markersize=4,
            label=label or "features",
        )
        ax.plot(
            ranks,
            ranks / nsubjects,
            color="crimson",
            lw=1.0,
            ls="--",
            label="chance (random)",
        )
        ax.axhline(1.0, color="k", lw=1.0, ls=":")
        ax.set_xlabel("Rank $k$")
        ax.set_ylabel(r"Cumulative accuracy (rank $\leq k$)")
        ax.set_xlim((0, nsubjects + 1))
        ax.set_ylim((0.0, 1.05))
        ax.set_title(rf"Subject ID (rank-1 ${results.rank1:.3f}$)")
        ax.legend(loc="lower right", fontsize="small")


def plot_identifiability(
    basename: pathlib.Path,
    identifiability: Array2D[np.floating[Any]],
    label_names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> None:
    labels = list(label_names)
    ntasks = len(labels)

    with figure(
        basename.with_stem(f"{basename.stem}-identifiability"),
        figsize=(8, 7),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.grid(visible=False)

        im = ax.imshow(identifiability, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(ntasks))
        ax.set_yticks(range(ntasks))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax.set_yticklabels(labels, fontsize="medium")
        ax.set_xlabel("Test task")
        ax.set_ylabel("Reference task")
        ax.set_title("Rank-1 identification")

        for i in range(ntasks):
            for j in range(ntasks):
                if i == j:
                    continue

                val = identifiability[i, j]
                color = "white" if val < 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize="small",
                )

        fig.colorbar(im, cax=make_colorbar_axes(ax), label="Rank-1 accuracy")


def plot_permutation_null(
    basename: pathlib.Path,
    null: Array1D[np.floating[Any]],
    observed: float,
    *,
    overwrite: bool = False,
) -> None:
    with figure(
        basename.with_stem(f"{basename.stem}-null"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ax.hist(null, bins=20, color="cornflowerblue", alpha=0.8, edgecolor="none")
        ax.axvline(observed, color="crimson", lw=2, label=rf"observed ${observed:.3f}$")
        ax.set_xlabel("Rank-1 accuracy")
        ax.set_ylabel("Count")
        ax.set_title("Permutation null")
        ax.legend(fontsize="small")


def plot_reliability(
    basename: pathlib.Path,
    self_similarity: Array1D[np.floating[Any]],
    other_similarity: Array1D[np.floating[Any]],
    *,
    overwrite: bool = False,
) -> None:
    with figure(
        basename.with_stem(f"{basename.stem}-reliability"),
        ncols=2,
        figsize=(13, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        lo = min(np.min(self_similarity), np.min(other_similarity))
        hi = max(np.max(self_similarity), np.max(other_similarity))
        bins = np.linspace(lo, hi, 25)

        ax1.hist(self_similarity, bins=bins, alpha=0.7, label="same subject")  # ty: ignore[invalid-argument-type]
        ax1.hist(other_similarity, bins=bins, alpha=0.7, label="other subjects")  # ty: ignore[invalid-argument-type]
        ax1.set_xlabel("Similarity (split-half)")
        ax1.set_ylabel("Count")
        ax1.set_title("Split-half reliability")
        ax1.legend(fontsize="small")

        ax2.scatter(other_similarity, self_similarity, s=24, alpha=0.8)
        ax2.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
        ax2.set_xlabel("Other-subject similarity")
        ax2.set_ylabel("Same-subject similarity")
        ax2.set_title("Self vs. others")
        ax2.legend(fontsize="small")


def plot_comparison(
    basename: pathlib.Path,
    labels: Sequence[str],
    rank1: Sequence[float],
    null_std: Sequence[float],
    cv_accuracy: Sequence[float],
    *,
    chance: float,
    overwrite: bool = False,
) -> None:
    xpos = np.arange(len(labels))

    with figure(
        basename.with_stem(f"{basename.stem}-comparison"),
        ncols=2,
        figsize=(14, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        ax1.bar(xpos, rank1, color="steelblue", width=0.6)
        ax1.errorbar(xpos, rank1, yerr=null_std, fmt="none", ecolor="k", capsize=3)
        ax1.axhline(chance, color="crimson", ls="--", lw=1.2, label="chance")
        ax1.set_xticks(xpos)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Rank-1 accuracy")
        ax1.set_ylim((0.0, 1.05))
        ax1.set_title("Subject identification")
        ax1.legend(fontsize="small")

        ax2.bar(xpos, cv_accuracy, color="darkcyan", width=0.6)
        ax2.axhline(chance, color="crimson", ls="--", lw=1.2, label="chance")
        ax2.set_xticks(xpos)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("Balanced accuracy")
        ax2.set_ylim((0.0, 1.05))
        ax2.set_title("Leave-one-task-out classification")
        ax2.legend(fontsize="small")


# }}}


# {{{ main


def main(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    metric: SimilarityMetric = "correlation",
    method: ClassificationMethod = "lda",
    nperm: int = 200,
    random_state: int = 42,
    overwrite: bool = False,
) -> int:
    ret = 0

    summary_labels: list[str] = []
    summary_rank1: list[float] = []
    summary_null_std: list[float] = []
    summary_cv: list[float] = []
    chance = 0.0

    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            ret = 1
            continue

        try:
            ds = load_feature_datasets(filename)
        except Exception as exc:
            log.error("Failed to load features from '%s': %s", filename, exc)
            ret = 1
            continue

        try:
            x = to_tensor(ds)
        except ValueError as exc:
            log.error("Failed to reshape features from '%s': %s", filename, exc)
            ret = 1
            continue

        ntasks, nsubjects, nfeatures = x.shape
        chance = 1.0 / nsubjects
        log.info(
            "Loaded '%s': %d tasks, %d subjects, %d features.",
            filename.stem,
            ntasks,
            nsubjects,
            nfeatures,
        )

        if outfile is None:
            basename = filename.with_suffix("")
        elif len(filenames) == 1:
            basename = outfile.with_suffix("")
        else:
            basename = outfile.with_stem(f"{filename.stem}-{outfile.stem}")

        # 1. leave-one-task-out identification
        results = compute_leave_one_task_out(x, metric)
        log.info(
            "    Rank-1: %.3f | rank-5: %.3f | mean rank: %.2f (chance %.4f)",
            results.rank1,
            results.rank5,
            results.mean_rank,
            chance,
        )
        plot_rank_curve(
            basename, results, label=clean_label(filename.stem), overwrite=overwrite
        )

        # 2. identifiability matrix
        identifiability = compute_identifiability_matrix(x, metric)
        plot_identifiability(
            basename, identifiability, ds.class_names, overwrite=overwrite
        )

        # 3. leave-one-task-out classification
        cv_accuracy = compute_cross_validated_classification(x, method=method)
        log.info("    Balanced accuracy (%s): %.3f", method, cv_accuracy)

        # 4. split-half reliability
        self_similarity, other_similarity = compute_split_half_reliability(
            x, metric, random_state=random_state
        )
        log.info(
            "    Split-half similarity: self %.3f vs others %.3f",
            np.mean(self_similarity),
            np.mean(other_similarity),
        )
        plot_reliability(
            basename, self_similarity, other_similarity, overwrite=overwrite
        )

        # 5. permutation null
        null = compute_permutation_null(
            x, metric, nperm=nperm, random_state=random_state
        )
        log.info(
            "    Permutation null: %.3f +/- %.3f (observed %.3f)",
            np.mean(null),
            np.std(null),
            results.rank1,
        )
        plot_permutation_null(basename, null, results.rank1, overwrite=overwrite)

        summary_labels.append(clean_label(filename.stem))
        summary_rank1.append(results.rank1)
        summary_null_std.append(float(np.std(null)))
        summary_cv.append(cv_accuracy)

    if len(summary_labels) > 1:
        if outfile is not None:
            compare_basename = outfile.with_suffix("")
        else:
            compare_basename = filenames[0].parent / "dataset-features"

        plot_comparison(
            compare_basename,
            summary_labels,
            summary_rank1,
            summary_null_std,
            summary_cv,
            chance=chance,
            overwrite=overwrite,
        )

    return ret


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
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
    )
    parser.add_argument(
        "--metric",
        choices=("correlation", "cosine", "euclidean"),
        default="correlation",
        help="Similarity metric used for identification",
    )
    parser.add_argument(
        "--method",
        choices=("lda", "ridge"),
        default="lda",
        help="Classifier used for leave-one-task-out classification",
    )
    parser.add_argument(
        "--nperm",
        type=int,
        default=200,
        help="Number of permutation-null samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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
            args.filenames,
            outfile=args.outfile,
            metric=args.metric,
            method=args.method,
            nperm=args.nperm,
            random_state=args.seed,
            overwrite=args.force,
        )
    )

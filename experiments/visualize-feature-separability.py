# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from commonlib import FeatureDataset, load_feature_datasets
from nneuroutil.classifier import AnovaScores, compute_anova_scores
from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array1D, Array2D, Array3D
from nneuroutil.visualization import figure, make_colorbar_axes

log = module_logger("lib.visualize_feature_separability")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Diagnose feature separability and discriminative power for classification.

Evaluates high- and low-dimensional features for:
1. Representational dissimilarity (class centroid distance matrix).
2. Supervised linear discriminant analysis (LDA) spectrum and Fisher projection.
3. Feature effect size distributions (vectorized ANOVA eta^2 and F-statistic).
4. Cross-validated linear decoding performance and confusion matrix.
5. Task vs. subject variance ratio (if repeated-measures/matched subjects).

Example:

    > {SCRIPT_PATH.name} dataset-features-matrix.npz
"""

EPS = 1.0e-12


# {{{ statistics


def compute_centroid_rdm(
    x: Array3D[np.floating[Any]],
) -> Array2D[np.floating[Any]]:
    """Compute class centroid correlation distance matrix."""
    from scipy.spatial.distance import cdist

    centroids = np.mean(x, axis=0)

    # correlation distance: 1 - pearson_r(u, v)
    rdm = cdist(centroids, centroids, metric="correlation")
    rdm = np.nan_to_num(rdm, nan=0.0)
    np.fill_diagonal(rdm, 0.0)

    return rdm


class LdaResults(NamedTuple):
    coords: Array2D[np.floating[Any]]
    """Projected coordinates on canonical discriminant dimensions."""
    explained_variance_ratio: Array1D[np.floating[Any]]
    """Ratio of between-class variance explained by each canonical component."""


def compute_supervised_lda(
    x: Array3D[np.floating[Any]],
    labels: Array3D[np.integer[Any]],
    *,
    pca_max_components: int = 50,
    random_state: int = 42,
) -> LdaResults:
    """Compute supervised Linear Discriminant Analysis."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    nsamples, nclasses, nfeatures = x.shape
    x = np.reshape(x, (nsamples * nclasses, nfeatures))
    labels = np.reshape(labels, -1)

    max_components = min(nclasses - 1, nfeatures)

    # When features exceed samples, project with PCA first to avoid singular Sw
    if nfeatures > x.shape[0] - nclasses:
        pca_dim = min(pca_max_components, x.shape[0] - nclasses, nfeatures)
        pca = PCA(n_components=pca_dim, random_state=random_state)
        x_in = pca.fit_transform(x)
    else:
        x_in = x

    lda = LinearDiscriminantAnalysis(n_components=max_components)
    coords = lda.fit_transform(x_in, labels)
    evr = lda.explained_variance_ratio_

    return LdaResults(coords=coords, explained_variance_ratio=evr)


class DecodingResults(NamedTuple):
    confusion_matrix: Array2D[np.floating[Any]]
    """Normalized confusion matrix (true rows, predicted columns)."""
    per_class_accuracy: Array1D[np.floating[Any]]
    """Balanced per-class accuracy (recall)."""
    overall_accuracy: float
    """Mean balanced accuracy across classes."""


def compute_cross_validated_decoding(
    x: Array3D[np.floating[Any]],
    labels: Array3D[np.integer[Any]],
    *,
    cv_folds: int = 5,
    random_state: int = 42,
) -> DecodingResults:
    """Evaluate linear decodability using cross-validation."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    nsamples, nclasses, nfeatures = x.shape
    x = np.reshape(x, (nsamples * nclasses, nfeatures))
    labels = np.reshape(labels, -1)

    clf = RidgeClassifier(alpha=1.0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    preds = cross_val_predict(clf, x, labels, cv=cv)

    cm = confusion_matrix(labels, preds, normalize="true")
    per_class = np.diag(cm)
    overall = float(np.mean(per_class))

    return DecodingResults(
        confusion_matrix=cm,
        per_class_accuracy=per_class,
        overall_accuracy=overall,
    )


class VarianceDecomposition(NamedTuple):
    ms_task: Array1D[np.floating[Any]]
    """Mean squares for task condition."""
    ms_subject: Array1D[np.floating[Any]]
    """Mean squares for individual subjects."""
    eta2_task: Array1D[np.floating[Any]]
    """Task effect size."""
    eta2_subject: Array1D[np.floating[Any]]
    """Subject effect size."""
    ratio: Array1D[np.floating[Any]]
    """Ratio of task to subject mean squares."""


def compute_task_subject_variance(
    x: Array3D[np.floating[Any]],
) -> VarianceDecomposition | None:
    """Decompose feature variance into task vs. subject components."""
    nsubjects, nclasses, nfeatures = x.shape
    if nsubjects < 2 or nclasses < 2:
        return None

    grand_mean = np.mean(x, axis=(0, 1), keepdims=True)
    task_mean = np.mean(x, axis=0, keepdims=True)
    subject_mean = np.mean(x, axis=1, keepdims=True)

    ss_task = nsubjects * np.sum((task_mean - grand_mean) ** 2, axis=(0, 1))
    ss_subject = nclasses * np.sum((subject_mean - grand_mean) ** 2, axis=(0, 1))
    ss_total = np.sum((x - grand_mean) ** 2, axis=(0, 1))

    df_task = max(nclasses - 1, 1)
    df_subject = max(nsubjects - 1, 1)

    ms_task = ss_task / df_task
    ms_subject = ss_subject / df_subject

    eta2_task = np.divide(
        ss_task,
        ss_total,
        out=np.zeros(nfeatures, dtype=x.dtype),
        where=ss_total > EPS,
    )
    eta2_subject = np.divide(
        ss_subject,
        ss_total,
        out=np.zeros(nfeatures, dtype=x.dtype),
        where=ss_total > EPS,
    )

    ratio = np.divide(
        ms_task,
        ms_subject,
        out=np.ones(nfeatures, dtype=x.dtype),
        where=ms_subject > EPS,
    )

    return VarianceDecomposition(
        ms_task=ms_task,
        ms_subject=ms_subject,
        eta2_task=eta2_task,
        eta2_subject=eta2_subject,
        ratio=ratio,
    )


# }}}


# {{{ plots


def clean_label(name: str) -> str:
    """Shorten task name for clean plotting."""
    return name.split("_", maxsplit=1)[0]


def plot_rdm(
    basename: pathlib.Path,
    rdm: Array2D[np.floating[Any]],
    label_names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> None:
    """Plot Representational Dissimilarity Matrix."""
    labels = [clean_label(n) for n in label_names]
    nclasses = len(labels)

    with figure(
        basename.with_stem(f"{basename.stem}-rdm"),
        figsize=(8, 7),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.grid(visible=False)
        im = ax.imshow(rdm, cmap="viridis", vmin=0.0)

        ax.set_xticks(range(nclasses))
        ax.set_yticks(range(nclasses))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax.set_yticklabels(labels, fontsize="medium")

        for i in range(nclasses):
            for j in range(nclasses):
                val = rdm[i, j]
                text_color = "white" if val < 0.6 * np.max(rdm) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize="small",
                )

        fig.colorbar(
            im, cax=make_colorbar_axes(ax), label=r"\text{Correlation Distance}"
        )
        ax.set_title(r"\text{Centroid Dissimilarity (RDM)}")


def plot_supervised_lda(
    basename: pathlib.Path,
    lda: LdaResults,
    ds: FeatureDataset,
    *,
    overwrite: bool = False,
) -> None:
    """Plot LDA canonical spectrum and 2D Fisher projection."""
    with figure(
        basename.with_stem(f"{basename.stem}-supervised-lda"),
        ncols=2,
        figsize=(15, 6),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        # Panel 1: variance spectrum
        ncomp = np.size(lda.explained_variance_ratio)
        idx = np.arange(1, ncomp + 1)
        ax1.bar(idx, lda.explained_variance_ratio, color="steelblue", width=0.6)
        ax1.set_xlabel(r"\text{Canonical Component}")
        ax1.set_ylabel(r"\text{Between-Class Variance Ratio}")
        ax1.set_title(r"\text{Fisher Discriminant Spectrum}")
        ax1.set_xticks(idx)

        # Panel 2: 2D projection
        if lda.coords.shape[1] >= 2:
            nsubjects, nclasses, _ = ds.x.shape
            coords = np.reshape(lda.coords, (nsubjects, nclasses, lda.coords.shape[-1]))

            for i, name in enumerate(ds.class_names):
                label = clean_label(name)
                ax2.scatter(
                    coords[:, i, 0],
                    coords[:, i, 1],
                    s=24,
                    alpha=0.75,
                    label=label,
                )
            ax2.set_xlabel(r"\text{Canonical Component 1}")
            ax2.set_ylabel(r"\text{Canonical Component 2}")
            ax2.set_title(r"\text{Fisher Linear Discriminant Map}")
            ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
        else:
            ax2.text(0.5, 0.5, "Only 1 discriminant dimension", ha="center")


def plot_effect_size_dist(
    basename: pathlib.Path,
    scores: AnovaScores,
    *,
    overwrite: bool = False,
) -> None:
    """Plot feature effect size distribution and empirical CDF."""
    eta2 = scores.eta2
    large_mask = np.greater_equal(eta2, 0.14)
    med_mask = np.logical_and(np.greater_equal(eta2, 0.06), np.less(eta2, 0.14))
    pct_large = 100.0 * float(np.mean(large_mask))
    pct_medium = 100.0 * float(np.mean(med_mask))

    with figure(
        basename.with_stem(f"{basename.stem}-effect-sizes"),
        ncols=2,
        figsize=(14, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        # Panel 1: Histogram
        ax1.hist(
            eta2,
            bins=50,
            density=True,
            color="cornflowerblue",
            edgecolor="none",
            alpha=0.8,
        )
        ax1.axvline(
            0.01,
            color="gray",
            linestyle=":",
            lw=1.5,
            label=r"\text{Small (0.01)}",
        )
        ax1.axvline(
            0.06,
            color="goldenrod",
            linestyle="--",
            lw=1.5,
            label=r"\text{Medium (0.06)}",
        )
        ax1.axvline(
            0.14,
            color="crimson",
            linestyle="-.",
            lw=1.5,
            label=r"\text{Large (0.14)}",
        )
        ax1.set_xlabel(r"ANOVA $\eta^2$")
        ax1.set_ylabel(r"Density")
        title = (
            rf"Effect Size Distribution (Large: {pct_large:.1f}\%, "
            rf"Medium: {pct_medium:.1f}\%)"
        )
        ax1.set_title(title)
        ax1.legend(fontsize="small")

        # Panel 2: ECDF
        sorted_eta2 = np.sort(eta2)
        cdf = np.linspace(0.0, 1.0, np.size(sorted_eta2))
        ax2.plot(sorted_eta2, cdf, color="navy", lw=2)
        ax2.axvline(0.14, color="crimson", linestyle="-.", lw=1.5)
        ax2.set_xlabel(r"ANOVA $\eta^2$")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("Empirical CDF")


def plot_decoding_cv(
    basename: pathlib.Path,
    decoding: DecodingResults,
    label_names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> None:
    """Plot cross-validated decoding confusion matrix and per-class accuracy."""
    labels = [clean_label(n) for n in label_names]
    nclasses = len(labels)
    chance = 1.0 / nclasses

    with figure(
        basename.with_stem(f"{basename.stem}-decoding-cv"),
        ncols=2,
        figsize=(16, 6.5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        # Panel 1: Confusion Matrix
        ax1.grid(visible=False)
        im = ax1.imshow(
            decoding.confusion_matrix,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        ax1.set_xticks(range(nclasses))
        ax1.set_yticks(range(nclasses))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax1.set_yticklabels(labels, fontsize="medium")
        ax1.set_xlabel(r"\text{Predicted Task}")
        ax1.set_ylabel(r"\text{True Task}")
        acc_str = f"{decoding.overall_accuracy * 100:.1f}\\%"
        ax1.set_title(rf"\text{{Confusion Matrix (Mean Acc: {acc_str})}}")

        for i in range(nclasses):
            for j in range(nclasses):
                val = decoding.confusion_matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax1.text(
                    j,
                    i,
                    f"{val * 100:.0f}\\%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize="small",
                )

        fig.colorbar(im, cax=make_colorbar_axes(ax1), label=r"\text{Proportion}")

        # Panel 2: Per-class accuracy
        x_pos = np.arange(nclasses)
        ax2.bar(x_pos, decoding.per_class_accuracy, color="darkcyan", width=0.55)
        ax2.axhline(
            chance,
            color="crimson",
            linestyle="--",
            lw=1.5,
            label=rf"\text{{Chance ({chance * 100:.1f}\%)}}",
        )
        ax2.axhline(
            decoding.overall_accuracy,
            color="black",
            linestyle=":",
            lw=1.5,
            label=rf"\text{{Mean ({decoding.overall_accuracy * 100:.1f}\%)}}",
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax2.set_ylabel(r"\text{Decoding Accuracy}")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_title(r"\text{Per-Class Decoding Performance}")
        ax2.legend(fontsize="small")


def plot_task_vs_subject_ratio(
    basename: pathlib.Path,
    var_decomp: VarianceDecomposition,
    *,
    overwrite: bool = False,
) -> None:
    """Plot ratio and comparison of task variance vs subject variance."""
    log_ratio = np.log10(np.maximum(var_decomp.ratio, 1.0e-6))
    pct_task_dominant = 100.0 * float(np.mean(np.greater(log_ratio, 0.0)))

    with figure(
        basename.with_stem(f"{basename.stem}-task-vs-subject"),
        ncols=2,
        figsize=(14, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        # Panel 1: log ratio histogram
        ax1.hist(
            log_ratio,
            bins=50,
            density=True,
            color="darkorange",
            edgecolor="none",
            alpha=0.8,
        )
        ax1.axvline(
            0.0,
            color="crimson",
            linestyle="--",
            lw=1.5,
            label=r"$MS_{\text{task}} = MS_{\text{subject}}$",
        )
        ax1.set_xlabel(r"$\log_{10}(MS_{\text{task}} / MS_{\text{subject}})$")
        ax1.set_ylabel(r"Density")
        task_dom_str = f"{pct_task_dominant:.1f}\\%"
        ax1.set_title(rf"Task vs Subject Variance (Task Dominant: {task_dom_str})")
        ax1.legend(fontsize="small")

        # Panel 2: scatter eta2_subject vs eta2_task
        npoints = np.size(var_decomp.eta2_subject)
        # subsample if large for plotting performance
        if npoints > 5000:
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(npoints, size=5000, replace=False)
        else:
            sub_idx = np.arange(npoints)

        ax2.scatter(
            var_decomp.eta2_subject[sub_idx],
            var_decomp.eta2_task[sub_idx],
            s=8,
            alpha=0.4,
            color="navy",
            edgecolor="none",
        )
        max_val = max(
            float(np.max(var_decomp.eta2_subject)),
            float(np.max(var_decomp.eta2_task)),
        )
        ax2.plot([0, max_val], [0, max_val], "k--", lw=1.2, label=r"y = x")
        ax2.set_xlabel(r"$\eta^2_{\text{subject}}$")
        ax2.set_ylabel(r"$\eta^2_{\text{task}}$")
        ax2.set_title("Subject vs. Task Effect Size")
        ax2.legend(fontsize="small")


# }}}


# {{{ main


def main(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    cv_folds: int = 5,
    overwrite: bool = False,
) -> int:
    ret = 0

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

        if outfile is None:
            basename = filename.with_suffix("")
        elif len(filenames) == 1:
            basename = outfile.with_suffix("")
        else:
            basename = outfile.with_stem(f"{filename.stem}-{outfile.stem}")

        nsubjects, nclasses, nfeatures = ds.x.shape
        log.info(
            "Loaded '%s': %d samples, %d features, %d classes.",
            filename.stem,
            nsubjects * nclasses,
            nfeatures,
            nclasses,
        )

        var_decomp = compute_task_subject_variance(ds.x)

        # 1. RDM
        log.info("Computing Representational Dissimilarity Matrix...")
        rdm = compute_centroid_rdm(ds.x)
        plot_rdm(basename, rdm, ds.class_names, overwrite=overwrite)

        # 2. Supervised LDA
        log.info("Computing supervised Fisher / LDA projection...")
        lda = compute_supervised_lda(ds.x, ds.labels)
        plot_supervised_lda(basename, lda, ds, overwrite=overwrite)

        # 3. Effect sizes
        log.info("Computing vectorized ANOVA effect sizes...")
        features = [ds.x[:, i, :] for i in range(nclasses)]
        scores = compute_anova_scores(features, eps=EPS)
        plot_effect_size_dist(basename, scores, overwrite=overwrite)

        # 4. Cross-validated linear decoding
        log.info("Evaluating cross-validated decodability (%d-fold)...", cv_folds)
        decoding = compute_cross_validated_decoding(ds.x, ds.labels, cv_folds=cv_folds)
        log.info(
            "Cross-validated balanced accuracy: %.2f%%",
            decoding.overall_accuracy * 100,
        )
        plot_decoding_cv(basename, decoding, ds.class_names, overwrite=overwrite)

        # 5. Task vs Subject variance
        if var_decomp is not None:
            log.info("Plotting task vs subject variance decomposition...")
            plot_task_vs_subject_ratio(basename, var_decomp, overwrite=overwrite)

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
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for decoding",
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
            cv_folds=args.cv_folds,
            overwrite=args.force,
        )
    )

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from commonlib import FeatureDataset, load_feature_datasets
from nneuroutil.classifier import (
    AnovaScores,
    classify_linear_discriminant_analysis,
    compute_anova_scores,
)
from nneuroutil.typing import Array1D, Array2D, Array3D
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.visualize_feature_separability")

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


class PcaLdaResults(NamedTuple):
    coords: Array2D[np.floating[Any]]
    """Projected coordinates on canonical discriminant dimensions."""
    explained_variance_ratio: Array1D[np.floating[Any]]
    """Ratio of between-class variance explained by each canonical component."""


def compute_pca_lda(
    x: Array3D[np.floating[Any]],
    labels: Array3D[np.integer[Any]],
    *,
    pca_max_components: int = 50,
    standardize: bool = False,
    random_state: int = 42,
) -> PcaLdaResults:
    """Compute in-sample PCA+LDA (supervised Linear Discriminant Analysis)."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    nsamples, nclasses, nfeatures = x.shape
    x = np.reshape(x, (nsamples * nclasses, nfeatures))
    labels = np.reshape(labels, -1)

    max_components = min(nclasses - 1, nfeatures)

    estimators = []
    if standardize:
        estimators.append(StandardScaler())

    # When features exceed samples, project with PCA first to avoid singular Sw
    if nfeatures > x.shape[0] - nclasses:
        pca_dim = min(pca_max_components, x.shape[0] - nclasses, nfeatures)
        estimators.append(PCA(n_components=pca_dim, random_state=random_state))

    estimators.append(LinearDiscriminantAnalysis(n_components=max_components))
    lda = make_pipeline(*estimators)

    coords = lda.fit_transform(x, labels)
    evr = lda[-1].explained_variance_ratio_

    return PcaLdaResults(coords=coords, explained_variance_ratio=evr)


class GsvdLdaResults(NamedTuple):
    test_coords: Array3D[np.floating[Any]]
    """Held-out test subject coordinates of shape ``(n_test, nclasses, k - 1)``."""
    proj_centroids: Array2D[np.floating[Any]]
    """Projected training class centroids of shape ``(nclasses, k - 1)``."""
    explained_variance_ratio: Array1D[np.floating[Any]]
    """Fraction of projected between-class variance explained by each axis."""
    test_accuracy: float
    """Balanced accuracy on held-out test subjects."""


def compute_gsvd_lda(
    x: Array3D[np.floating[Any]],
    labels: Array3D[np.integer[Any]],
    *,
    standardize: bool = False,
    random_state: int = 42,
) -> GsvdLdaResults:
    """Fit G-SVD LDA on training subjects and project held-out test subjects."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    nsubjects, nclasses, nfeatures = x.shape
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(nsubjects)
    n_test = max(1, round(0.3 * nsubjects))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    x_train = x[train_idx]
    x_test = x[test_idx]

    if standardize:
        scaler = StandardScaler()
        scaler.fit(np.reshape(x_train, (train_idx.size * nclasses, nfeatures)))
        x_train = np.reshape(
            scaler.transform(np.reshape(x_train, (-1, nfeatures))),
            x_train.shape,
        )
        x_test = np.reshape(
            scaler.transform(np.reshape(x_test, (-1, nfeatures))),
            x_test.shape,
        )

    class_labels = [int(labels[0, i]) for i in range(nclasses)]
    lda = classify_linear_discriminant_analysis(
        [x_train[:, i, :] for i in range(nclasses)],
        class_labels,
        shrinkage=1.0,
    )

    x_test_flat = np.reshape(x_test, (n_test * nclasses, nfeatures))
    y_test_flat = np.reshape(labels[test_idx], -1)
    y_pred = lda.predict(x_test_flat, metric="l2")
    test_accuracy = float(balanced_accuracy_score(y_test_flat, y_pred))

    test_coords_flat = (x_test_flat - lda.centroids) @ lda.G
    test_coords = np.reshape(test_coords_flat, (n_test, nclasses, -1))

    between_var = np.sum(lda.proj_centroids**2, axis=0)
    total_var = float(np.sum(between_var))
    evr = between_var / total_var if total_var > 0.0 else between_var

    return GsvdLdaResults(
        test_coords=test_coords,
        proj_centroids=lda.proj_centroids,
        explained_variance_ratio=evr,
        test_accuracy=test_accuracy,
    )


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
    standardize: bool = False,
    random_state: int = 42,
) -> DecodingResults:
    """Evaluate linear decodability using cross-validation."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    nsamples, nclasses, nfeatures = x.shape
    x = np.reshape(x, (nsamples * nclasses, nfeatures))
    labels = np.reshape(labels, -1)

    clf = RidgeClassifier(alpha=1.0)
    if standardize:
        # NOTE: the scaler is fit on the training folds only
        clf = make_pipeline(StandardScaler(), clf)

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


def plot_pca_lda(
    basename: pathlib.Path,
    lda: PcaLdaResults,
    ds: FeatureDataset,
    *,
    overwrite: bool = False,
) -> None:
    """Plot in-sample PCA+LDA canonical spectrum and 2D Fisher projection."""
    with figure(
        basename.with_stem(f"{basename.stem}-lda-pca-spectrum"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ncomp = np.size(lda.explained_variance_ratio)
        idx = np.arange(1, ncomp + 1)
        ax.bar(idx, lda.explained_variance_ratio, color="steelblue", width=0.6)
        ax.set_xlabel(r"\text{Canonical Component}")
        ax.set_ylabel(r"\text{Between-Class Variance Ratio}")
        ax.set_title(r"\text{Fisher Discriminant Spectrum}")
        ax.set_xticks(idx)

    if lda.coords.shape[1] >= 2:
        with figure(
            basename.with_stem(f"{basename.stem}-lda-pca-map"),
            figsize=(8, 6),
            overwrite=overwrite,
        ) as fig:
            ax = fig.gca()
            nsubjects, nclasses, _ = ds.x.shape
            coords = np.reshape(lda.coords, (nsubjects, nclasses, lda.coords.shape[-1]))

            for i, name in enumerate(ds.class_names):
                label = clean_label(name)
                ax.scatter(
                    coords[:, i, 0],
                    coords[:, i, 1],
                    s=24,
                    alpha=0.75,
                    label=label,
                )
            ax.set_xlabel(r"\text{Canonical Component 1}")
            ax.set_ylabel(r"\text{Canonical Component 2}")
            ax.set_title(r"\text{Fisher Linear Discriminant Map}")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")


def plot_gsvd_lda(
    basename: pathlib.Path,
    lda: GsvdLdaResults,
    ds: FeatureDataset,
    *,
    overwrite: bool = False,
) -> None:
    """Plot held-out G-SVD LDA canonical spectrum and 2D test-subject projection."""
    with figure(
        basename.with_stem(f"{basename.stem}-lda-gsvd-spectrum"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ncomp = np.size(lda.explained_variance_ratio)
        idx = np.arange(1, ncomp + 1)
        ax.bar(idx, lda.explained_variance_ratio, color="steelblue", width=0.6)
        ax.set_xlabel(r"\text{Canonical Component}")
        ax.set_ylabel(r"\text{Between-Class Variance Ratio}")
        ax.set_title(r"\text{G-SVD Discriminant Spectrum}")
        ax.set_xticks(idx)

    if lda.test_coords.shape[2] >= 2:
        with figure(
            basename.with_stem(f"{basename.stem}-lda-gsvd-map"),
            figsize=(8, 6),
            overwrite=overwrite,
        ) as fig:
            ax = fig.gca()
            for i, name in enumerate(ds.class_names):
                label = clean_label(name)
                sc = ax.scatter(
                    lda.test_coords[:, i, 0],
                    lda.test_coords[:, i, 1],
                    s=28,
                    alpha=0.75,
                    label=label,
                )
                ax.scatter(
                    lda.proj_centroids[i, 0],
                    lda.proj_centroids[i, 1],
                    marker="x",
                    s=90,
                    linewidths=2.0,
                    color=sc.get_facecolor()[0],
                )
            ax.set_xlabel(r"\text{Canonical Component 1}")
            ax.set_ylabel(r"\text{Canonical Component 2}")
            acc_str = f"{lda.test_accuracy * 100:.1f}\\%"
            ax.set_title(rf"\text{{Held-Out G-SVD Fisher Map (Test Acc: {acc_str})}}")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")


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
        basename.with_stem(f"{basename.stem}-effect-sizes-hist"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ax.hist(
            eta2,
            bins=50,
            density=True,
            color="cornflowerblue",
            edgecolor="none",
            alpha=0.8,
        )
        ax.axvline(
            0.01,
            color="gray",
            linestyle=":",
            lw=1.5,
            label=r"\text{Small (0.01)}",
        )
        ax.axvline(
            0.06,
            color="goldenrod",
            linestyle="--",
            lw=1.5,
            label=r"\text{Medium (0.06)}",
        )
        ax.axvline(
            0.14,
            color="crimson",
            linestyle="-.",
            lw=1.5,
            label=r"\text{Large (0.14)}",
        )
        ax.set_xlabel(r"ANOVA $\eta^2$")
        ax.set_ylabel(r"Density")
        title = (
            rf"Effect Size Distribution (Large: {pct_large:.1f}\%, "
            rf"Medium: {pct_medium:.1f}\%)"
        )
        ax.set_title(title)
        ax.legend(fontsize="small")

    with figure(
        basename.with_stem(f"{basename.stem}-effect-sizes-ecdf"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        sorted_eta2 = np.sort(eta2)
        cdf = np.linspace(0.0, 1.0, np.size(sorted_eta2))
        ax.plot(sorted_eta2, cdf, color="navy", lw=2)
        ax.axvline(0.14, color="crimson", linestyle="-.", lw=1.5)
        ax.set_xlabel(r"ANOVA $\eta^2$")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Empirical CDF")


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
        basename.with_stem(f"{basename.stem}-decoding-cv-confusion"),
        figsize=(8, 7),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ax.grid(visible=False)
        im = ax.imshow(
            decoding.confusion_matrix,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xticks(range(nclasses))
        ax.set_yticks(range(nclasses))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax.set_yticklabels(labels, fontsize="medium")
        ax.set_xlabel(r"\text{Predicted Task}")
        ax.set_ylabel(r"\text{True Task}")
        acc_str = f"{decoding.overall_accuracy * 100:.1f}\\%"
        ax.set_title(rf"\text{{Confusion Matrix (Mean Acc: {acc_str})}}")

        for i in range(nclasses):
            for j in range(nclasses):
                val = decoding.confusion_matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val * 100:.0f}\\%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize="small",
                )

        fig.colorbar(im, cax=make_colorbar_axes(ax), label=r"\text{Proportion}")

    with figure(
        basename.with_stem(f"{basename.stem}-decoding-cv-accuracy"),
        figsize=(8, 6),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        x_pos = np.arange(nclasses)
        ax.bar(x_pos, decoding.per_class_accuracy, color="darkcyan", width=0.55)
        ax.axhline(
            chance,
            color="crimson",
            linestyle="--",
            lw=1.5,
            label=rf"\text{{Chance ({chance * 100:.1f}\%)}}",
        )
        ax.axhline(
            decoding.overall_accuracy,
            color="black",
            linestyle=":",
            lw=1.5,
            label=rf"\text{{Mean ({decoding.overall_accuracy * 100:.1f}\%)}}",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax.set_ylabel(r"\text{Decoding Accuracy}")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(r"\text{Per-Class Decoding Performance}")
        ax.legend(fontsize="small")


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
        basename.with_stem(f"{basename.stem}-task-vs-subject-ratio"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ax.hist(
            log_ratio,
            bins=50,
            density=True,
            color="darkorange",
            edgecolor="none",
            alpha=0.8,
        )
        ax.axvline(
            0.0,
            color="crimson",
            linestyle="--",
            lw=1.5,
            label=r"$MS_{\text{task}} = MS_{\text{subject}}$",
        )
        ax.set_xlabel(r"$\log_{10}(MS_{\text{task}} / MS_{\text{subject}})$")
        ax.set_ylabel(r"Density")
        task_dom_str = f"{pct_task_dominant:.1f}\\%"
        ax.set_title(rf"Task vs Subject Variance (Task Dominant: {task_dom_str})")
        ax.legend(fontsize="small")

    with figure(
        basename.with_stem(f"{basename.stem}-task-vs-subject-scatter"),
        figsize=(7, 5.5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        npoints = np.size(var_decomp.eta2_subject)
        # subsample if large for plotting performance
        if npoints > 5000:
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(npoints, size=5000, replace=False)
        else:
            sub_idx = np.arange(npoints)

        ax.scatter(
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
        ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label=r"y = x")
        ax.set_xlabel(r"$\eta^2_{\text{subject}}$")
        ax.set_ylabel(r"$\eta^2_{\text{task}}$")
        ax.set_title("Subject vs. Task Effect Size")
        ax.legend(fontsize="small")


# }}}


# {{{ main


def main(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    cv_folds: int = 5,
    standardize: bool = False,
    exclude: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> int:
    ret = 0

    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            ret = 1
            continue

        try:
            ds = load_feature_datasets(filename, exclude=exclude)
        except Exception as exc:
            log.error("Failed to load features from '%s': %s", filename, exc)
            ret = 1
            continue

        suffix = ""
        if outfile is None:
            basename = filename.with_stem(f"{filename.stem}{suffix}").with_suffix("")
        elif len(filenames) == 1:
            basename = outfile.with_stem(f"{outfile.stem}{suffix}").with_suffix("")
        else:
            basename = outfile.with_stem(f"{filename.stem}-{outfile.stem}{suffix}")

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

        # 2. In-sample PCA+LDA
        log.info("Computing in-sample PCA+LDA projection...")
        lda = compute_pca_lda(ds.x, ds.labels, standardize=standardize)
        plot_pca_lda(basename, lda, ds, overwrite=overwrite)

        # 2b. Held-out G-SVD LDA
        log.info("Computing held-out G-SVD LDA projection...")
        gsvd_lda = compute_gsvd_lda(ds.x, ds.labels, standardize=standardize)
        log.info(
            "Held-out G-SVD LDA balanced accuracy: %.2f%%",
            gsvd_lda.test_accuracy * 100,
        )
        plot_gsvd_lda(basename, gsvd_lda, ds, overwrite=overwrite)

        # 3. Effect sizes
        log.info("Computing vectorized ANOVA effect sizes...")
        features = [ds.x[:, i, :] for i in range(nclasses)]
        scores = compute_anova_scores(features, eps=EPS)
        plot_effect_size_dist(basename, scores, overwrite=overwrite)

        # 4. Cross-validated linear decoding
        log.info("Evaluating cross-validated decodability (%d-fold)...", cv_folds)
        decoding = compute_cross_validated_decoding(
            ds.x, ds.labels, cv_folds=cv_folds, standardize=standardize
        )
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
        "--exclude",
        action="append",
        default=None,
        help="Classes to exclude from the visualization",
    )
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
        "--standardize",
        action="store_true",
        help="Standardize each feature to zero mean and unit variance",
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
            standardize=args.standardize,
            exclude=args.exclude,
            overwrite=args.force,
        )
    )

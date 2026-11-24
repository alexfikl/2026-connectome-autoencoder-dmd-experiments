# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

import commonlib  # ruff: ignore[unused-import]
from classifylib import flatten_feature_datasets, load_feature_datasets
from nneuroutil.classifier import (
    AnovaScores,
    classify_linear_discriminant_analysis,
    compute_anova_scores,
)
from nneuroutil.typing import Array1D, Array2D
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.visualize_feature_separability")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Diagnose feature separability and discriminative power for classification.

Only the functional (fMRI) classes are used by default. Pass ``--binary`` to
pool all functional scans against the structural scans, using subject-grouped
cross-validation.

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
    x: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
) -> Array2D[np.floating[Any]]:
    """Compute class centroid correlation distance matrix."""
    from scipy.spatial.distance import cdist

    classes = np.unique(labels)
    centroids = np.stack([np.mean(x[labels == c], axis=0) for c in classes])

    # correlation distance: 1 - pearson_r(u, v)
    rdm = cdist(centroids, centroids, metric="correlation")
    rdm = np.nan_to_num(rdm, nan=0.0)
    np.fill_diagonal(rdm, 0.0)

    return rdm


class LdaResults(NamedTuple):
    test_coords: Array2D[np.floating[Any]]
    """Held-out test scan coordinates of shape ``(n_test, k - 1)``."""
    test_labels: Array1D[np.integer[Any]]
    """Class index for each held-out test scan."""
    proj_centroids: Array2D[np.floating[Any]]
    """Projected training class centroids of shape ``(nclasses, k - 1)``."""
    explained_variance_ratio: Array1D[np.floating[Any]]
    """Ratio of between-class variance explained by each canonical component."""
    test_accuracy: float
    """Balanced accuracy on held-out test scans."""


def _split_groups(
    groups: Array1D[np.integer[Any]],
    *,
    random_state: int = 42,
) -> tuple[Array1D[np.integer[Any]], Array1D[np.integer[Any]]]:
    from sklearn.model_selection import GroupShuffleSplit

    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, test_idx = next(split.split(groups, groups=groups))

    return train_idx, test_idx


def compute_pca_lda(
    x: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
    *,
    pca_max_components: int = 50,
    zscore: bool = True,
    random_state: int = 42,
) -> LdaResults:
    """Fit PCA+LDA on training subjects and project held-out test scans."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    nfeatures = x.shape[1]
    train_idx, test_idx = _split_groups(groups, random_state=random_state)

    x_train = x[train_idx]
    y_train = labels[train_idx]
    x_test = x[test_idx]
    y_test = labels[test_idx]

    classes = np.unique(labels)
    nclasses = classes.size
    max_components = min(nclasses - 1, nfeatures)

    estimators = []
    if zscore:
        estimators.append(StandardScaler())

    # When features exceed samples, project with PCA first to avoid singular Sw
    if nfeatures > x_train.shape[0] - nclasses:
        pca_dim = min(pca_max_components, x_train.shape[0] - nclasses, nfeatures)
        estimators.append(PCA(n_components=pca_dim, random_state=random_state))

    estimators.append(LinearDiscriminantAnalysis(n_components=max_components))
    lda = make_pipeline(*estimators)

    train_coords = lda.fit_transform(x_train, y_train)
    proj_centroids = np.stack([
        np.mean(train_coords[y_train == c], axis=0) for c in classes
    ])
    test_coords = lda.transform(x_test)
    evr = lda[-1].explained_variance_ratio_
    test_accuracy = balanced_accuracy_score(y_test, lda.predict(x_test))

    return LdaResults(
        test_coords=test_coords,
        test_labels=y_test,
        proj_centroids=proj_centroids,
        explained_variance_ratio=evr,
        test_accuracy=test_accuracy,
    )


def compute_gsvd_lda(
    x: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
    *,
    zscore: bool = True,
    random_state: int = 42,
) -> LdaResults:
    """Fit G-SVD LDA on training subjects and project held-out test scans."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    train_idx, test_idx = _split_groups(groups, random_state=random_state)

    x_train = x[train_idx]
    y_train = labels[train_idx]
    x_test = x[test_idx]
    y_test = labels[test_idx]

    if zscore:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    classes = np.unique(labels)
    class_labels = [int(c) for c in classes]
    lda = classify_linear_discriminant_analysis(
        [x_train[y_train == c] for c in classes],
        class_labels,
        shrinkage=1.0,
    )

    y_pred = lda.predict(x_test, metric="l2")
    test_accuracy = balanced_accuracy_score(y_test, y_pred)

    test_coords = (x_test - lda.centroids) @ lda.G

    between_var = np.sum(lda.proj_centroids**2, axis=0)
    total_var = np.sum(between_var)
    evr = between_var / total_var if total_var > 0.0 else between_var

    return LdaResults(
        test_coords=test_coords,
        test_labels=y_test,
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
    x: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
    *,
    cv_folds: int = 5,
    zscore: bool = True,
    random_state: int = 42,
) -> DecodingResults:
    """Evaluate linear decodability using subject-grouped cross-validation."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = RidgeClassifier(alpha=1.0)
    if zscore:
        # NOTE: the scaler is fit on the training folds only
        clf = make_pipeline(StandardScaler(), clf)

    cv = StratifiedGroupKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    preds = cross_val_predict(clf, x, labels, cv=cv, groups=groups)

    cm = confusion_matrix(labels, preds, normalize="true")
    per_class = np.diag(cm)
    overall = np.mean(per_class)

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
    x: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
) -> VarianceDecomposition | None:
    """Decompose feature variance into class vs. subject components."""
    classes, class_counts = np.unique(labels, return_counts=True)
    unique_groups, subject_counts = np.unique(groups, return_counts=True)
    nclasses = classes.size
    nsubjects = unique_groups.size
    if nsubjects < 2 or nclasses < 2:
        return None

    nfeatures = x.shape[1]
    grand_mean = np.mean(x, axis=0)
    class_means = np.stack([np.mean(x[labels == c], axis=0) for c in classes])
    subject_means = np.stack([np.mean(x[groups == g], axis=0) for g in unique_groups])

    ss_task = np.sum(class_counts[:, None] * (class_means - grand_mean) ** 2, axis=0)
    ss_subject = np.sum(
        subject_counts[:, None] * (subject_means - grand_mean) ** 2, axis=0
    )
    ss_total = np.sum((x - grand_mean) ** 2, axis=0)

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


def plot_rdm(
    basename: pathlib.Path,
    rdm: Array2D[np.floating[Any]],
    label_names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> None:
    """Plot Representational Dissimilarity Matrix."""
    labels = list(label_names)
    nclasses = len(labels)

    with figure(
        basename.with_stem(f"{basename.stem}-rdm"),
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

        fig.colorbar(im, cax=make_colorbar_axes(ax), label="Correlation Distance")
        ax.set_title("Centroid Dissimilarity (RDM)")


def plot_lda(
    basename: pathlib.Path,
    lda: LdaResults,
    class_names: tuple[str, ...],
    *,
    stem: str,
    name: str,
    overwrite: bool = False,
) -> None:
    """Plot a held-out LDA spectrum and, if possible, the Fisher map."""
    with figure(
        basename.with_stem(f"{basename.stem}-lda-{stem}-spectrum"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ncomp = np.size(lda.explained_variance_ratio)
        idx = np.arange(1, ncomp + 1)
        ax.bar(idx, lda.explained_variance_ratio, color="steelblue", width=0.6)
        ax.set_xlabel("Canonical Component")
        ax.set_ylabel("Between-Class Variance Ratio")
        ax.set_title(f"{name} Discriminant Spectrum")
        ax.set_xticks(idx)

    if lda.test_coords.shape[1] < 2:
        return

    with figure(
        basename.with_stem(f"{basename.stem}-lda-{stem}-map"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        for i, class_name in enumerate(class_names):
            mask = lda.test_labels == i
            sc = ax.scatter(
                lda.test_coords[mask, 0],
                lda.test_coords[mask, 1],
                s=28,
                alpha=0.75,
                label=class_name,
            )
            ax.scatter(
                lda.proj_centroids[i, 0],
                lda.proj_centroids[i, 1],
                marker="x",
                s=90,
                linewidths=2.0,
                color=sc.get_facecolor()[0],
            )

        ax.set_xlabel("Canonical Component 1")
        ax.set_ylabel("Canonical Component 2")
        acc = 100.0 * lda.test_accuracy
        ax.set_title(f"Held-Out {name} Fisher Map (Test Acc: {acc:.1f}\\%)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")


def plot_effect_size_dist(
    basename: pathlib.Path,
    scores: AnovaScores,
    *,
    overwrite: bool = False,
) -> None:
    """Plot feature effect size distribution and empirical CDF."""
    eta2 = scores.eta2
    pct_large = 100.0 * np.mean(eta2 >= 0.14)  # ty: ignore[unsupported-operator]
    pct_medium = 100.0 * np.mean((eta2 >= 0.06) & (eta2 < 0.14))  # ty: ignore[unsupported-operator]

    with figure(
        basename.with_stem(f"{basename.stem}-effect-sizes-hist"),
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
            label="Small (0.01)",
        )
        ax.axvline(
            0.06,
            color="goldenrod",
            linestyle="--",
            lw=1.5,
            label="Medium (0.06)",
        )
        ax.axvline(
            0.14,
            color="crimson",
            linestyle="-.",
            lw=1.5,
            label="Large (0.14)",
        )
        ax.set_xlabel(r"ANOVA $\eta^2$")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Effect Size Distribution (Large: {pct_large:.1f}\\%, "
            f"Medium: {pct_medium:.1f}\\%)"
        )
        ax.legend(fontsize="small")

    with figure(
        basename.with_stem(f"{basename.stem}-effect-sizes-ecdf"),
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
    labels = list(label_names)
    nclasses = len(labels)
    chance = 1.0 / nclasses

    with figure(
        basename.with_stem(f"{basename.stem}-decoding-cv-confusion"),
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
        ax.set_xlabel("Predicted Task")
        ax.set_ylabel("True Task")
        ax.set_title(
            f"Confusion Matrix (Mean Acc: {decoding.overall_accuracy * 100:.1f}\\%)"
        )

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

        fig.colorbar(im, cax=make_colorbar_axes(ax), label="Proportion")

    with figure(
        basename.with_stem(f"{basename.stem}-decoding-cv-accuracy"),
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
            label=rf"Chance ({chance * 100:.1f}\%)",
        )
        ax.axhline(
            decoding.overall_accuracy,
            color="black",
            linestyle=":",
            lw=1.5,
            label=rf"Mean ({decoding.overall_accuracy * 100:.1f}\%)",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="medium")
        ax.set_ylabel("Decoding Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Per-Class Decoding Performance")
        ax.legend(fontsize="small")


def plot_task_vs_subject_ratio(
    basename: pathlib.Path,
    var_decomp: VarianceDecomposition,
    *,
    overwrite: bool = False,
) -> None:
    """Plot ratio and comparison of task variance vs subject variance."""
    log_ratio = np.log10(np.maximum(var_decomp.ratio, 1.0e-6))
    pct_task_dominant = 100.0 * np.mean(log_ratio > 0.0)

    with figure(
        basename.with_stem(f"{basename.stem}-task-vs-subject-ratio"),
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
        ax.set_ylabel("Density")
        ax.set_title(
            f"Task vs Subject Variance (Task Dominant: {pct_task_dominant:.1f}\\%)"
        )
        ax.legend(fontsize="small")

    with figure(
        basename.with_stem(f"{basename.stem}-task-vs-subject-scatter"),
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
        max_val = max(np.max(var_decomp.eta2_subject), np.max(var_decomp.eta2_task))
        ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label="y = x")
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
    zscore: bool = True,
    binary: bool = False,
    overwrite: bool = False,
) -> int:
    ret = 0

    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            ret = 1
            continue

        try:
            ds = load_feature_datasets(filename, binary=binary)
            x, labels, groups = flatten_feature_datasets(ds)
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

        suffix = "-fmri-vs-mri" if binary else "-fmri-only"
        if zscore:
            suffix += "-zscore"
        basename = basename.with_stem(f"{basename.stem}{suffix}")

        log.info(
            "Loaded '%s': %d samples, %d features, %d classes.",
            filename.stem,
            x.shape[0],
            x.shape[1],
            len(ds.class_names),
        )

        var_decomp = None
        if binary:
            log.info("Skipping task vs subject variance for the unbalanced design.")
        else:
            var_decomp = compute_task_subject_variance(x, labels, groups)

        # 1. RDM
        log.info("Computing Representational Dissimilarity Matrix...")
        rdm = compute_centroid_rdm(x, labels)
        plot_rdm(basename, rdm, ds.class_names, overwrite=overwrite)

        # 2. Held-out PCA+LDA
        log.info("Computing held-out PCA+LDA projection...")
        lda = compute_pca_lda(x, labels, groups, zscore=zscore)
        log.info("Held-out PCA+LDA balanced accuracy: %.2f%%", lda.test_accuracy * 100)
        plot_lda(
            basename,
            lda,
            ds.class_names,
            stem="pca",
            name="PCA+LDA",
            overwrite=overwrite,
        )

        # 2b. Held-out G-SVD LDA
        log.info("Computing held-out G-SVD LDA projection...")
        gsvd_lda = compute_gsvd_lda(x, labels, groups, zscore=zscore)
        log.info(
            "Held-out G-SVD LDA balanced accuracy: %.2f%%",
            gsvd_lda.test_accuracy * 100,
        )
        plot_lda(
            basename,
            gsvd_lda,
            ds.class_names,
            stem="gsvd",
            name="G-SVD",
            overwrite=overwrite,
        )

        # 3. Effect sizes
        log.info("Computing vectorized ANOVA effect sizes...")
        features = [np.asarray(xi) for xi in ds.x]
        scores = compute_anova_scores(features, eps=EPS)
        plot_effect_size_dist(basename, scores, overwrite=overwrite)

        # 4. Cross-validated linear decoding
        log.info("Evaluating cross-validated decodability (%d-fold)...", cv_folds)
        decoding = compute_cross_validated_decoding(
            x,
            labels,
            groups,
            cv_folds=cv_folds,
            zscore=zscore,
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
        "--binary",
        action="store_true",
        help=(
            "Compare functional against structural scans; the default only "
            "compares the functional classes"
        ),
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
        "--no-zscore",
        action="store_false",
        dest="zscore",
        help="Disable z-scoring each feature to zero mean and unit variance",
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
            zscore=args.zscore,
            binary=args.binary,
            overwrite=args.force,
        )
    )

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np

from commonlib import FeatureDataset, load_feature_datasets
from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array1D, Array2D
from nneuroutil.visualization import figure, make_colorbar_axes

log = module_logger("lib.feature_discriminants")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Rank features computed by make-features.py by how discriminative they are.

Computes the mutual information (MI) and the ANOVA effect size (:math:`\\eta^2`)
between each feature and the class label, aggregates them over the feature
groups stored in the file and looks at the redundancy of the top features.

Example:

    > {SCRIPT_PATH.name} dataset-features-sgraph.npz
"""


# {{{ statistics


class Scores(NamedTuple):
    nmi: Array1D[np.floating[Any]]
    """Normalized Mutual Information."""
    eta2: Array1D[np.floating[Any]]
    """ANOVA effect size."""
    f: Array1D[np.floating[Any]]
    """ANOVA :math:`f` statistic."""


def label_entropy(labels: Array1D[np.integer[Any]]) -> float:
    """Entropy of the class labels in nats."""

    counts = np.bincount(labels)
    prob = counts / np.sum(counts)
    prob = prob[prob > 0.0]

    return -np.sum(prob * np.log(prob))


def compute_scores(
    x: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
    *,
    eps: float = 1.0e-12,
    mi_seeds: int = 32,
    random_state: int = 42,
) -> Scores:
    """Compute per-feature discriminant scores."""
    from sklearn.feature_selection import mutual_info_classif

    classes = np.unique(labels)

    mi = np.mean(
        [
            mutual_info_classif(x, labels, random_state=random_state + i)
            for i in range(mi_seeds)
        ],
        axis=0,
    )
    mi = np.nan_to_num(mi, nan=0.0)

    h_labels = label_entropy(labels)
    nmi = mi / h_labels if h_labels > 0.0 else mi

    from nneuroutil.classifier import compute_anova_scores

    # one-way ANOVA effect size (SS_between / SS_total) and F statistic
    features = [x[labels == c] for c in classes]
    anova = compute_anova_scores(features, eps=eps, xp=np)

    return Scores(nmi=nmi, eta2=anova.eta2, f=anova.f_stat)


def expand_groups(
    starts: Array1D[np.integer[Any]],
    descrs: tuple[str, ...],
    nfeatures: int,
) -> Array1D[np.integer[Any]]:
    """Expand the group boundaries into a per-feature group index."""
    starts = np.asarray(starts)
    ngroups = len(descrs)
    if starts.size != ngroups + 1:
        raise ValueError(
            f"expected {ngroups + 1} group boundaries, got {starts.size}: {descrs}."
        )
    if int(starts[0]) != 0 or int(starts[-1]) != nfeatures:
        raise ValueError(
            f"group boundaries do not span the features: {starts[0]} to {starts[-1]}"
            f" for {nfeatures} features."
        )

    groups = np.empty(nfeatures, dtype=np.int64)
    for i in range(ngroups):
        groups[starts[i] : starts[i + 1]] = i

    return groups


def group_reduce(
    values: Array1D[np.floating[Any]],
    groups: Array1D[np.integer[Any]],
    ngroups: int,
    *,
    how: Literal["mean", "max"] = "mean",
) -> Array1D[np.floating[Any]]:
    """Aggregate per-feature *values* over each feature group."""
    reduce_fn = np.max if how == "max" else np.mean
    return np.array([reduce_fn(values[groups == g]) for g in range(ngroups)])


def feature_labels(
    indices: Array1D[np.integer[Any]],
    groups: Array1D[np.integer[Any]],
    starts: Array1D[np.integer[Any]],
    descrs: tuple[str, ...],
) -> list[str]:
    """Build a label ``"<group> [<index within group>]"`` for each feature."""
    starts = np.asarray(starts)
    return [
        f"{descrs[int(groups[j])]} [{int(j - starts[int(groups[j])])}]" for j in indices
    ]


# }}}


# {{{ plots


def plot_top_features(
    basename: pathlib.Path,
    values: Array1D[np.floating[Any]],
    ds: FeatureDataset,
    groups: Array1D[np.integer[Any]],
    *,
    name: str,
    top_k: int = 30,
    overwrite: bool = False,
) -> None:
    import matplotlib as mp

    top_k = min(top_k, values.size)  # ty: ignore[unresolved-attribute]
    order = np.argsort(-values)[:top_k]
    labels = feature_labels(order, groups, ds.starts, ds.descrs)

    cmap = mp.colormaps["tab20"]
    colors = [cmap(int(groups[j]) % 20) for j in order]

    with figure(
        basename.with_stem(f"{basename.stem}-{name}-top"),
        figsize=(8, max(4, 0.3 * top_k)),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        y = np.arange(order.size)
        ax.barh(y, values[order], color=colors)
        ax.set_xlabel(name.upper())

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize="small")
        ax.invert_yaxis()

        ax.grid(visible=False, which="both")


def plot_groups(
    basename: pathlib.Path,
    values: Array1D[np.floating[Any]],
    groups: Array1D[np.integer[Any]],
    ds: FeatureDataset,
    *,
    name: str,
    overwrite: bool = False,
) -> None:
    ngroups = len(ds.descrs)
    mean = group_reduce(values, groups, ngroups, how="mean")
    maximum = group_reduce(values, groups, ngroups, how="max")

    x = np.arange(ngroups)
    width = 0.4
    with figure(
        basename.with_stem(f"{basename.stem}-group-{name}"),
        figsize=(max(8.0, 1.2 * ngroups), 5),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        ax.bar(x - width / 2, mean, width, label="mean")
        ax.bar(x + width / 2, maximum, width, label="max")

        ax.set_xticks(x)
        ax.set_xticklabels(ds.descrs, rotation=45, ha="right", fontsize="small")
        ax.set_ylabel(name.upper())

        ax.legend()


def plot_correlation(
    basename: pathlib.Path,
    x: Array2D[np.floating[Any]],
    values: Array1D[np.floating[Any]],
    ds: FeatureDataset,
    groups: Array1D[np.integer[Any]],
    *,
    top_k: int = 30,
    overwrite: bool = False,
) -> None:
    from scipy.stats import spearmanr

    order = np.argsort(-values)[: min(top_k, values.size)]  # ty: ignore[unresolved-attribute]
    if order.size < 2:
        return

    corr = np.asarray(spearmanr(x[:, order]).statistic)
    if corr.ndim != 2:
        return

    labels = feature_labels(order, groups, ds.starts, ds.descrs)
    with figure(
        basename.with_stem(f"{basename.stem}-correlation"),
        figsize=(9, 8),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="seismic")

        idx = np.arange(order.size)
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels(labels, rotation=90, fontsize="x-small")
        ax.set_yticklabels(labels, fontsize="x-small")

        ax.grid(visible=False, which="both")
        fig.colorbar(im, cax=make_colorbar_axes(ax), label="Spearman")


# }}}


# {{{ main


def main(
    filenames: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    top_k: int = 30,
    mi_seeds: int = 32,
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

        suffix = ""
        if outfile is None:
            basename = filename.with_stem(f"{filename.stem}{suffix}").with_suffix("")
        elif len(filenames) == 1:
            basename = outfile.with_stem(f"{outfile.stem}{suffix}").with_suffix("")
        else:
            basename = outfile.with_stem(f"{filename.stem}-{outfile.stem}{suffix}")

        nsubjects, nclasses, nfeatures = ds.x.shape
        x = np.reshape(ds.x, (nsubjects * nclasses, nfeatures))
        labels = np.reshape(ds.labels, -1)

        groups = expand_groups(ds.starts, ds.descrs, nfeatures)
        scores = compute_scores(x, labels, mi_seeds=mi_seeds)

        log.info(
            "Features '%s': %d samples, %d features, %d classes.",
            filename.stem,
            nsubjects * nclasses,
            nfeatures,
            nclasses,
        )

        # {{{ ranked table

        order = np.argsort(-scores.nmi)
        log.info("Top %d features by normalized mutual information:", top_k)
        for rank, j in enumerate(order[:top_k]):
            label = feature_labels(np.array([j]), groups, ds.starts, ds.descrs)[0]
            log.info(
                "[%3d] %-40s NMI %.5e eta2 %.5e F %.5e",
                rank + 1,
                label,
                scores.nmi[j],
                scores.eta2[j],
                scores.f[j],
            )

        log.info("Group importance (max):")
        for i, descr in enumerate(ds.descrs):
            mask = groups == i
            log.info(
                "[%3d] %-40s NMI %.5e eta2 %.5e",
                i + 1,
                descr,
                np.max(scores.nmi[mask]),
                np.max(scores.eta2[mask]),
            )

        # }}}

        for name in ("nmi", "eta2"):
            plot_top_features(
                basename,
                getattr(scores, name),
                ds,
                groups,
                name=name,
                top_k=top_k,
                overwrite=overwrite,
            )
            plot_groups(
                basename,
                getattr(scores, name),
                groups,
                ds,
                name=name,
                overwrite=overwrite,
            )

        plot_correlation(
            basename,
            x,
            scores.nmi,
            ds,
            groups,
            top_k=top_k,
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
        "--top-k",
        type=int,
        default=30,
        help="Number of highest scoring features to plot and tabulate",
    )
    parser.add_argument(
        "--mi-seeds",
        type=int,
        default=32,
        help="Number of random seeds used to average the mutual information",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
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
            top_k=args.top_k,
            mi_seeds=args.mi_seeds,
            overwrite=args.force,
        )
    )

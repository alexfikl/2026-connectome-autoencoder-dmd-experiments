# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, Literal

import numpy as np

import commonlib  # ruff: ignore[unused-import]
from commonlib import FeatureDataset, load_feature_datasets
from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array1D, Array2D
from nneuroutil.visualization import figure

log = module_logger("lib.visualize_features")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Visualize the features computed by make-features.py.

Each input file (e.g. ``dataset-features-sgraph.npz``) is embedded into two
dimensions and plotted colored by class. The explained variance of a PCA is
also plotted to give an idea of the intrinsic dimensionality of the features.

Example:

    > {SCRIPT_PATH.name} --method pca tsne dataset-features-sgraph.npz
"""

EmbeddingMethod = Literal["pca", "tsne", "mds", "isomap", "spectral"]

EPS = 1.0e-12


# {{{ embeddings


def standardize(
    x: Array2D[np.floating[Any]],
    *,
    eps: float = EPS,
) -> Array2D[np.floating[Any]]:
    """Center and scale each feature to unit variance."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def compute_embedding(
    x: Array2D[np.floating[Any]],
    method: EmbeddingMethod,
    *,
    pca_components: int = 50,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Array2D[np.floating[Any]]:
    """Embed *x* into two dimensions using the given *method*."""
    from sklearn.decomposition import PCA

    if method == "pca":
        return np.asarray(
            PCA(n_components=2, random_state=random_state).fit_transform(x)
        )

    # reduce to a manageable dimension for the manifold methods, mostly to
    # denoise the very high dimensional (e.g. matrix) features
    n_components = min(pca_components, x.shape[1], x.shape[0] - 1)
    if n_components > 2:
        x = PCA(n_components=n_components, random_state=random_state).fit_transform(x)

    if method == "tsne":
        from sklearn.manifold import TSNE

        # perplexity must be smaller than the number of samples
        perp = float(min(perplexity, (x.shape[0] - 1) / 3.0))
        model = TSNE(
            n_components=2,
            perplexity=max(perp, 1.0),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
    elif method == "mds":
        from sklearn.manifold import MDS

        model = MDS(
            n_components=2,
            n_init=1,
            max_iter=300,
            normalized_stress="auto",
            random_state=random_state,
        )
    elif method == "isomap":
        from sklearn.manifold import Isomap

        model = Isomap(n_components=2, n_neighbors=min(10, x.shape[0] - 1))
    elif method == "spectral":
        from sklearn.manifold import SpectralEmbedding

        model = SpectralEmbedding(
            n_components=2,
            n_neighbors=min(10, x.shape[0] - 1),
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown method: {method!r}")

    return np.asarray(model.fit_transform(x))


def compute_silhouette(
    coords: Array2D[np.floating[Any]],
    labels: Array1D[np.integer[Any]],
) -> float:
    """Compute the silhouette score of the embedding, or NaN if not possible."""
    from sklearn.metrics import silhouette_score

    nclasses = len(np.unique(labels))
    if nclasses < 2 or coords.shape[0] <= nclasses:
        return float("nan")

    return float(silhouette_score(coords, labels))


# }}}


# {{{ plots


def plot_pca_variance(
    basename: pathlib.Path,
    x: Array2D[np.floating[Any]],
    *,
    overwrite: bool = False,
) -> None:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(x.shape)).fit(x)
    evr = pca.explained_variance_ratio_
    idx = np.arange(1, evr.size + 1)

    with figure(
        basename.with_stem(f"{basename.stem}-pca-variance"),
        ncols=2,
        figsize=(12, 5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        ax1.plot(idx, evr, "o-", markersize=3)
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained variance ratio")
        ax1.set_title("Scree")

        ax2.plot(idx, np.cumsum(evr), "o-", markersize=3)
        ax2.axhline(0.9, color="k", lw=1, ls="--")
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Cumulative explained variance")
        ax2.set_title("Cumulative")


def plot_embedding(
    basename: pathlib.Path,
    method: EmbeddingMethod,
    coords: Array2D[np.floating[Any]],
    ds: FeatureDataset,
    *,
    overwrite: bool = False,
) -> None:
    silhouette = compute_silhouette(coords, ds.labels)

    with figure(
        basename.with_stem(f"{basename.stem}-embedding-{method}"),
        figsize=(7, 6),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()

        for i, name in enumerate(ds.label_names):
            mask = ds.labels == i
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=name)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(rf"{method.upper()} (silhouette ${silhouette:.3f}$)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")


# }}}


# {{{ main


def main(
    filenames: list[pathlib.Path],
    *,
    methods: tuple[EmbeddingMethod, ...] = ("pca", "tsne"),
    outfile: pathlib.Path | None = None,
    pca_components: int = 50,
    perplexity: float = 30.0,
    standardize_features: bool = True,
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
            basename = outfile.with_stem(f"{outfile.stem}-{filename.stem}")

        x = ds.x
        keep = np.var(x, axis=0) > EPS
        if not np.all(keep):
            log.info("Dropping %d constant features.", int(np.sum(~keep)))
            x = x[:, keep]

        log.info(
            "Features '%s': %d samples, %d features, %d classes.",
            filename.stem,
            x.shape[0],
            x.shape[1],
            len(ds.label_names),
        )

        if standardize_features:
            x = standardize(x)

        plot_pca_variance(basename, x, overwrite=overwrite)

        for method in methods:
            log.info("Computing '%s' embedding for '%s'.", method, filename.stem)
            coords = compute_embedding(
                x,
                method,
                pca_components=pca_components,
                perplexity=perplexity,
            )
            plot_embedding(basename, method, coords, ds, overwrite=overwrite)

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
        "--method",
        nargs="+",
        choices=("pca", "tsne", "mds", "isomap", "spectral"),
        default=("pca", "tsne"),
        help="Embedding method(s) to compute",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components used to pre-reduce the features",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity used by t-SNE",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize each feature to zero mean and unit variance",
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
            methods=tuple(args.method),
            outfile=args.outfile,
            pca_components=args.pca_components,
            perplexity=args.perplexity,
            standardize_features=args.standardize,
            overwrite=args.force,
        )
    )

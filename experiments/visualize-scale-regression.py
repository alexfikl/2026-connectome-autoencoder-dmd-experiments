# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from commonlib import load_feature_datasets
from nneuroutil.typing import Array1D, Array2D
from nneuroutil.visualization import figure

log = logging.getLogger("lib.visualize_scale_regression")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Evaluate out-of-fold Ridge regression predicting matrix scale from feature families.

Predicts global matrix scale (Frobenius norm and spectral radius from
``dataset-features-scale.npz``) using repeated subject-grouped 5-fold CV
with inner-fold Ridge regularization selection and training-fold z-scoring.

Example:

    > {SCRIPT_PATH.name} \\
        --scale-file data/dataset-features-scale.npz \\
        data/dataset-features-geometry.npz \\
        data/dataset-features-dmd.npz \\
        data/dataset-features-ae-dmd.npz
"""


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
    groups: Array1D[np.integer[Any]],
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

    unique_subjects = np.unique(groups)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    alphas = np.logspace(-3, 5, 25)

    nsamples = x.shape[0]
    oof_pred = np.empty((n_repeats, nsamples), dtype=y.dtype)
    fold_r2 = np.empty(n_splits * n_repeats, dtype=x.dtype)

    # go through all the folds: fit + compute the R2 scores
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(unique_subjects)):
        rep = fold_idx // n_splits
        train_mask = np.isin(groups, unique_subjects[train_idx])
        test_mask = np.isin(groups, unique_subjects[test_idx])

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
    subj_rows = [np.flatnonzero(groups == s) for s in unique_subjects]
    rng = np.random.default_rng(seed)
    boot_r2 = np.empty(n_bootstrap, dtype=x.dtype)

    for b in range(n_bootstrap):
        sampled = rng.integers(0, len(subj_rows), size=len(subj_rows))
        idx = np.concatenate([subj_rows[s] for s in sampled])
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


# {{{ plots


def plot_scale_regression(
    outfile: pathlib.Path,
    results_fro: list[RegressionResult],
    results_rho: list[RegressionResult],
    *,
    scatter_index: int = -1,
    overwrite: bool = False,
) -> None:
    """Plot the Ridge R^2 bars and the predicted vs. true scatter."""
    names = [r.name for r in results_fro]
    x_pos = np.arange(len(names))
    width = 0.36

    fro_means = np.array([r.r2_mean for r in results_fro])
    fro_err = np.array([
        [max(0.0, r.r2_mean - r.r2_ci_lo) for r in results_fro],
        [max(0.0, r.r2_ci_hi - r.r2_mean) for r in results_fro],
    ])

    rho_means = np.array([r.r2_mean for r in results_rho])
    rho_err = np.array([
        [max(0.0, r.r2_mean - r.r2_ci_lo) for r in results_rho],
        [max(0.0, r.r2_ci_hi - r.r2_mean) for r in results_rho],
    ])

    with figure(
        outfile.with_stem(f"{outfile.stem}-r2"),
        figsize=(6.5, 4.5),
        overwrite=overwrite,
    ) as fig:
        (ax,) = fig.axes
        ax.bar(
            x_pos - width / 2,
            fro_means,
            width,
            yerr=fro_err,
            capsize=3,
            label=r"$\|\mathbf{A}\|_F$",
        )
        ax.bar(
            x_pos + width / 2,
            rho_means,
            width,
            yerr=rho_err,
            capsize=3,
            label=r"$\rho(\mathbf{A})$",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names)
        ax.set_ylabel(r"Out-of-fold $R^2$")
        ax.set_ylim(min(0.0, np.min(fro_means) - 0.05), 1.05)
        ax.set_title(r"Global scale prediction ($R^2$)")
        ax.legend(fontsize="small")

    scatter_res = results_fro[scatter_index]
    with figure(
        outfile.with_stem(f"{outfile.stem}-scatter"),
        figsize=(5.0, 4.5),
        overwrite=overwrite,
    ) as fig:
        (ax,) = fig.axes
        struct = scatter_res.is_structural
        func = ~struct  # ty: ignore[unsupported-operator]
        ax.scatter(
            scatter_res.y_true[func],
            scatter_res.y_pred[func],
            s=18,
            alpha=0.7,
            label="Functional",
        )
        ax.scatter(
            scatter_res.y_true[struct],
            scatter_res.y_pred[struct],
            s=22,
            alpha=0.85,
            label="Structural",
        )

        lo = min(np.min(scatter_res.y_true), np.min(scatter_res.y_pred))
        hi = max(np.max(scatter_res.y_true), np.max(scatter_res.y_pred))
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(r"True $\|\mathbf{A}\|_F$")
        ax.set_ylabel(r"Predicted $\|\mathbf{A}\|_F$")
        ax.set_title(rf"{scatter_res.name} ($R^2 = {scatter_res.r2_mean:.3f}$)")
        ax.legend(fontsize="small")


# }}}


# {{{ main


def main(
    filenames: Sequence[pathlib.Path],
    *,
    scale_file: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    overwrite: bool = False,
) -> int:
    if not filenames:
        log.error("No predictor feature files given.")
        return 1

    if scale_file is None:
        scale_file = filenames[0].parent / "dataset-features-scale.npz"

    if not scale_file.exists():
        log.error("Scale feature file does not exist: '%s'.", scale_file)
        return 1

    ds_scale = load_feature_datasets(scale_file, binary=True)
    y_scale = np.concatenate(ds_scale.x, axis=0)
    y_fro = y_scale[:, 0]
    y_rho = y_scale[:, 1]
    is_structural = np.repeat(
        [name == "Structural" for name in ds_scale.class_names],
        [xi.shape[0] for xi in ds_scale.x],
    )

    results_fro: list[RegressionResult] = []
    results_rho: list[RegressionResult] = []

    for filename in filenames:
        if not filename.exists():
            log.error("Predictor feature file does not exist: '%s'.", filename)
            return 1

        ds = load_feature_datasets(filename, binary=True)
        if ds.class_names != ds_scale.class_names:
            log.error(
                "Class names in '%s' do not match '%s': %s vs %s",
                filename,
                scale_file,
                ds.class_names,
                ds_scale.class_names,
            )
            return 1

        x_all = np.concatenate(ds.x, axis=0)
        groups_all = np.concatenate(ds.groups, axis=0)
        name = ds.label

        res_fro = run_ridge_cv(
            x_all,
            y_fro,
            groups_all,
            is_structural,
            name=name,
            target_name="||A||_F",
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
        )
        res_rho = run_ridge_cv(
            x_all,
            y_rho,
            groups_all,
            is_structural,
            name=name,
            target_name="rho(A)",
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
        )

        log.info(
            "%-14s | ||A||_F R^2 = %.4f [%.4f, %.4f] | rho(A) R^2 = %.4f [%.4f, %.4f]",
            name,
            res_fro.r2_mean,
            res_fro.r2_ci_lo,
            res_fro.r2_ci_hi,
            res_rho.r2_mean,
            res_rho.r2_ci_lo,
            res_rho.r2_ci_hi,
        )

        results_fro.append(res_fro)
        results_rho.append(res_rho)

    if outfile is None:
        outfile = filenames[0].parent / "dataset-features-scale-regression"

    plot_scale_regression(
        outfile,
        results_fro,
        results_rho,
        scatter_index=-1,
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
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--scale-file",
        type=pathlib.Path,
        default=None,
        help="Path to dataset-features-scale.npz",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Prefix of the output figure paths",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of subject-grouped CV folds",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of repeated CV runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splits",
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
            scale_file=args.scale_file,
            outfile=args.outfile,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
            overwrite=args.force,
        )
    )

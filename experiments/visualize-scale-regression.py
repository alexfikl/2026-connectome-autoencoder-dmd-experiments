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


def _clean_family_name(path: pathlib.Path) -> str:
    stem = path.stem.removeprefix("dataset-features-")
    mapping = {
        "geometry": "Geometry",
        "dmd": "DMDc",
        "ae-dmd": "AE-DMD",
        "image": "Raw fractal",
        "global": "Global graph",
        "local": "Local graph",
        "eig": "Eigenspectrum",
        "matrix": "Raw matrix",
    }
    return mapping.get(stem, stem)


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
    oof_pred = np.empty((n_repeats, nsamples), dtype=np.float64)
    fold_r2 = np.empty(n_splits * n_repeats, dtype=np.float64)

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
        fold_r2[fold_idx] = float(r2_score(y[test_mask], pred))

    y_pred_mean = np.mean(oof_pred, axis=0)

    # Subject-level bootstrap 95% CI on out-of-fold R^2
    subj_rows = [np.flatnonzero(groups == s) for s in unique_subjects]
    rng = np.random.default_rng(seed)
    boot_r2 = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, len(subj_rows), size=len(subj_rows))
        idx = np.concatenate([subj_rows[s] for s in sampled])
        boot_r2[b] = float(r2_score(y[idx], y_pred_mean[idx]))
    ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])

    return RegressionResult(
        name=name,
        target_name=target_name,
        r2_mean=float(np.mean(fold_r2)),
        r2_ci_lo=float(ci_lo),
        r2_ci_hi=float(ci_hi),
        y_true=np.asarray(y, dtype=np.float64),
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
    """Plot 2-panel Stage 1 summary: (a) Ridge R^2 bars, (b) pred vs. true scatter."""
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

    scatter_res = results_fro[scatter_index]

    with figure(
        outfile,
        ncols=2,
        figsize=(11, 4.5),
        overwrite=overwrite,
    ) as fig:
        ax1, ax2 = fig.axes

        ax1.bar(
            x_pos - width / 2,
            fro_means,
            width,
            yerr=fro_err,
            capsize=3,
            label=r"$\|\mathbf{A}\|_F$",
        )
        ax1.bar(
            x_pos + width / 2,
            rho_means,
            width,
            yerr=rho_err,
            capsize=3,
            label=r"$\rho(\mathbf{A})$",
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names)
        ax1.set_ylabel(r"Out-of-fold $R^2$")
        ax1.set_ylim(min(0.0, float(np.min(fro_means)) - 0.05), 1.05)
        ax1.set_title(r"(a) Global scale prediction ($R^2$)")
        ax1.legend(fontsize="small")

        struct = scatter_res.is_structural
        func = ~struct  # ty: ignore[unsupported-operator]
        ax2.scatter(
            scatter_res.y_true[func],
            scatter_res.y_pred[func],
            s=18,
            alpha=0.7,
            label="Functional",
        )
        ax2.scatter(
            scatter_res.y_true[struct],
            scatter_res.y_pred[struct],
            s=22,
            alpha=0.85,
            label="Structural",
        )

        lo = float(min(np.min(scatter_res.y_true), np.min(scatter_res.y_pred)))
        hi = float(max(np.max(scatter_res.y_true), np.max(scatter_res.y_pred)))
        pad = 0.04 * (hi - lo)
        ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2)
        ax2.set_xlim(lo - pad, hi + pad)
        ax2.set_ylim(lo - pad, hi + pad)
        ax2.set_xlabel(r"True $\|\mathbf{A}\|_F$")
        ax2.set_ylabel(r"Predicted $\|\mathbf{A}\|_F$")
        ax2.set_title(rf"(b) {scatter_res.name} ($R^2 = {scatter_res.r2_mean:.3f}$)")
        ax2.legend(fontsize="small")


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
    is_structural = np.concatenate([
        np.full(xi.shape[0], name == "Structural", dtype=bool)
        for xi, name in zip(ds_scale.x, ds_scale.class_names, strict=True)
    ])

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
        name = _clean_family_name(filename)

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
        outfile = filenames[0].parent / "stage1-scale-regression.jpg"

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
        help="Output figure path",
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

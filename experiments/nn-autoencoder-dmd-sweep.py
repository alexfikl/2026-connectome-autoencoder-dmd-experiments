# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np
import torch

import dmdaelib
from nneuroutil.dmdc import DenseDMDc
from nneuroutil.helpers import FuzzyChoiceAction
from nneuroutil.interop.torch import available_device_names, torch_default_device
from nneuroutil.visualization import figure

log = logging.getLogger("lib.autoencoder_sweep")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Run or resume hyperparameter sweeps for the DMD Autoencoder and plot summary
diagnostics. Existing runs in the output cache are reused unless --force is given.

Examples:

    > {SCRIPT_PATH.name} --device cuda --study all data/Rest_LR_Task.npz
"""


def _mask_scores(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    eval_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from nneuroutil.scores import dice_score, jaccard_index

    if eval_mask is not None:
        pred = pred[eval_mask]
        true = true[eval_mask]

    iou = cast(torch.Tensor, jaccard_index(pred, true))
    dice = cast(torch.Tensor, dice_score(pred, true))

    return iou, dice


def _predict_metrics(
    model: dmdaelib.DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    config: dmdaelib.DMDConfig,
    *,
    zmask: torch.Tensor,
    eval_mask: torch.Tensor | None = None,
) -> tuple[dmdaelib.ErrorMetrics, torch.Tensor]:
    norm = dmdaelib.torch_predict(
        model,
        dmdc,
        bbox=config.bbox,
        resolution=config.resolution,
        maxit=config.maxit,
    )

    in_set = norm < 0.1 * config.escape_radius
    iou, dice = _mask_scores(in_set, zmask, eval_mask=eval_mask)
    metrics = dmdaelib.ErrorMetrics(
        iou=iou.detach().cpu(),
        dice=dice.detach().cpu(),
        a_radius=dmdaelib.spectral_radius(dmdc.A),  # ty: ignore[invalid-argument-type]
    )

    return metrics, in_set.detach().cpu()


# {{{ sweep

SWEEP_STUDIES: dict[str, dict[str, Sequence[int]]] = {
    "latent": {
        "npod": (24, 36, 48, 64),
        "latent_size": (48, 64, 96, 128, 192),
        "hidden_size": (256,),
        "c_features": (32,),
        "c_depth": (1,),
    },
    "control": {
        "npod": (48,),
        "latent_size": (128,),
        "hidden_size": (256,),
        "c_features": (8, 16, 32, 64, 96),
        "c_depth": (1, 2),
    },
    "width": {
        "npod": (48,),
        "latent_size": (128,),
        "hidden_size": (128, 256, 512),
        "c_features": (32,),
        "c_depth": (1,),
    },
}


def _print_sweep_summary(
    results: Mapping[str, Mapping[str, Any]], *, title: str
) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title, header_style="bold")
    table.add_column("npod", justify="right")
    table.add_column("latent", justify="right")
    table.add_column("hidden", justify="right")
    table.add_column("c_feat", justify="right")
    table.add_column("c_depth", justify="right")
    table.add_column("loss(train)", justify="right")
    table.add_column("loss(valid)", justify="right")
    table.add_column("loss(test)", justify="right")
    table.add_column("Dice(test)", justify="right")
    table.add_column("IoU(test)", justify="right")
    table.add_column("|lam|max", justify="right")

    # NOTE: order by the test loss, not by Dice/IoU
    records = sorted(results.values(), key=lambda r: r["metrics"]["loss_test"])

    for record in records:
        config = cast(Mapping[str, Any], record["config"])
        metrics = cast(Mapping[str, float], record["metrics"])
        table.add_row(
            str(config["npod"]),
            str(config["latent_size"]),
            str(config["hidden_size"]),
            str(config["c_features"]),
            str(config["c_depth"]),
            f"{metrics['loss_train']:.3e}",
            f"{metrics['loss_val']:.3e}",
            f"{metrics['loss_test']:.3e}",
            f"{metrics['dice']:.4f}",
            f"{metrics['iou']:.4f}",
            f"{metrics['dmdaelib.spectral_radius']:.3f}",
        )

    Console().print(table)


def _plot_sweep(
    filename: pathlib.Path,
    results: Mapping[str, Mapping[str, Any]],
    *,
    overwrite: bool = True,
) -> None:
    if not results:
        return

    stem = filename.with_suffix("")
    records = list(results.values())
    configs = [cast(Mapping[str, Any], record["config"]) for record in records]
    metrics = [cast(Mapping[str, float], record["metrics"]) for record in records]

    # {{{ summary

    param_names = [
        name
        for name in ("npod", "latent_size", "hidden_size", "c_features", "c_depth")
        if len({config[name] for config in configs}) > 1
    ]
    if not param_names:
        param_names = ["npod"]

    with figure(
        stem.with_stem(f"{stem.stem}-sweep-summary"),
        nrows=2,
        ncols=len(param_names),
        overwrite=overwrite,
    ) as fig:
        axes = np.reshape(fig.axes, (2, len(param_names)))
        quantities = (
            ("dice", "Dice (test)", False, (0.0, 1.0)),
            ("loss_test", "Test loss", True, None),
        )
        for col, name in enumerate(param_names):
            x = np.array([config[name] for config in configs])
            for row, (key, label, log_scale, ylim) in enumerate(quantities):
                ax = axes[row, col]
                y = np.array([metric[key] for metric in metrics], dtype=float)
                ax.scatter(x, y, s=18, alpha=0.7)
                ux = np.unique(x)
                uy = np.array([np.nanmean(y[x == u]) for u in ux])
                ax.plot(ux, uy, color="C1", marker="o")
                ax.set_xlabel(name)
                ax.set_title(label)
                if row == 0:
                    ax.set_ylabel(label)
                if log_scale:
                    ax.set_yscale("log")
                if ylim is not None:
                    ax.set_ylim(*ylim)

    # }}}

    # {{{ losses

    max_epochs = max(int(np.asarray(r["losses"]).shape[0]) for r in records)
    losses = np.full((len(records), max_epochs, 2, 4), np.nan)
    for i, record in enumerate(records):
        value = np.asarray(record["losses"], dtype=float)
        losses[i, : value.shape[0]] = value

    labels = ("Total", "Rec", "Lin", "Pre")
    with figure(
        stem.with_stem(f"{stem.stem}-sweep-losses"),
        ncols=len(labels),
        overwrite=overwrite,
    ) as fig:
        for k, (ax, label) in enumerate(zip(fig.axes, labels, strict=True)):
            for i in range(len(records)):
                ax.plot(losses[i, :, 0, k], color="C0", alpha=0.15, lw=0.6)
                ax.plot(
                    losses[i, :, 1, k],
                    color="C1",
                    alpha=0.15,
                    lw=0.6,
                    ls="--",
                )
            train = np.nanmedian(losses[:, :, 0, k], axis=0)
            valid = np.nanmedian(losses[:, :, 1, k], axis=0)
            ax.plot(train, color="C0", lw=1.5, label="Train")
            ax.plot(valid, color="C1", lw=1.5, ls="--", label="Valid.")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_title(label)
            ax.legend()

    # }}}

    # {{{ predictions

    valid = [record for record in records if "pred_mask" in record]
    if valid:
        bbox = cast(tuple[float, float, float, float], valid[0]["config"]["bbox"])
        ncols = min(6, len(valid) + 1)
        nrows = (len(valid) + 1 + ncols - 1) // ncols
        with figure(
            stem.with_stem(f"{stem.stem}-sweep-predict"),
            nrows=nrows,
            ncols=ncols,
            overwrite=overwrite,
        ) as fig:
            axes = fig.axes
            for ax in axes[len(valid) + 1 :]:
                ax.set_axis_off()

            ax = axes[0]
            ax.imshow(np.asarray(valid[0]["zmask"]), extent=bbox, cmap="binary")
            ax.set_title("True")
            ax.set_axis_off()

            for ax, record in zip(axes[1:], valid, strict=False):
                config = cast(Mapping[str, Any], record["config"])
                ax.imshow(np.asarray(record["pred_mask"]), extent=bbox, cmap="binary")
                label = " ".join(
                    f"{name}={config[name]}"
                    for name in ("npod", "latent_size", "c_features", "c_depth")
                )
                ax.set_title(label, fontsize="small")
                ax.set_axis_off()

    # }}}


def sweep(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    study: str = "all",
    overwrite: bool = False,
    **overrides: Any,
) -> int:
    """Train missing sweep configurations and plot all available results."""
    from commonlib import load_matrix_datasets

    device = torch.get_default_device()
    config = dmdaelib.DMDConfig.from_sources(overrides)

    if outfile is None:
        outfile = filename.parent / "sweeps" / f"{filename.stem}-m{config.dataidx}.pkl"
    else:
        outfile = outfile.with_suffix(".pkl")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    if outfile.exists():
        # NOTE: the results contain numpy arrays, so `weights_only=True` cannot
        # load them; this is a locally generated file
        results = cast(
            dict[str, dict[str, Any]],
            torch.load(outfile, map_location="cpu", weights_only=False),
        )
        log.info("Loaded %d sweep results from '%s'.", len(results), outfile)

    studies = SWEEP_STUDIES if study == "all" else {study: SWEEP_STUDIES[study]}

    run_configs: list[dmdaelib.DMDConfig] = []
    seen = set()
    for axes in studies.values():
        for values in itertools.product(
            axes["npod"],
            axes["latent_size"],
            axes["hidden_size"],
            axes["c_features"],
            axes["c_depth"],
        ):
            npod, latent_size, hidden_size, c_features, c_depth = values
            if latent_size < npod:
                continue
            effective_depth = 1 if c_features == 0 else c_depth
            run_config = replace(
                config,
                npod=npod,
                latent_size=latent_size,
                hidden_size=hidden_size,
                c_features=c_features,
                c_depth=effective_depth,
            )
            key = run_config.suffix
            if key in seen:
                continue
            seen.add(key)
            run_configs.append(run_config)

    if not run_configs:
        log.error("No valid parameter combinations to sweep.")
        return 1

    current_keys = {run_config.suffix for run_config in run_configs}

    missing = [
        run_config
        for run_config in run_configs
        if overwrite or run_config.suffix not in results
    ]
    log.info(
        "Sweep has %d configurations (%d missing, %d cached).",
        len(run_configs),
        len(missing),
        len(run_configs) - len(missing),
    )

    dataset: dmdaelib.DMDDataset | None = None
    if missing:
        try:
            matrices = cast(
                torch.Tensor,
                load_matrix_datasets(
                    filename, key=config.datakey, device=device, xp=torch
                ),
            )
            mat = matrices[config.dataidx].to(torch.complex64)
        except Exception as exc:
            log.error("Failed to load dataset '%s': %s", filename.name, exc)
            return 1

        # NOTE: the dataset is generated once and shared by all runs
        dataset = dmdaelib.prepare_dataset(config, mat)
        test_mask = dataset.test_mask
    else:
        # NOTE: no dataset is needed, but the cached metrics still have to be
        # recomputed on the test split
        test_mask = dmdaelib.make_test_mask(config, device)

    # NOTE: cached records may have been evaluated on the full grid
    for record in results.values():
        record_config = record.get("config", {})
        if (
            record_config.get("dataidx") != config.dataidx
            or record_config.get("resolution") != config.resolution
            or record_config.get("seed") != config.seed
            or tuple(record_config.get("split", ())) != tuple(config.split)
        ):
            continue

        pred = record.get("pred_mask")
        true = record.get("zmask")
        if pred is None or true is None:
            continue

        pred_t = torch.as_tensor(pred, dtype=torch.bool, device=device)
        true_t = torch.as_tensor(true, dtype=torch.bool, device=device)
        if pred_t.shape != test_mask.shape:
            continue

        iou, dice = _mask_scores(pred_t, true_t, eval_mask=test_mask)
        record["metrics"]["dice"] = float(dice)
        record["metrics"]["iou"] = float(iou)

    if dataset is not None:
        for i, run_config in enumerate(missing, start=1):
            log.info(
                "[%02d/%02d] npod=%d latent_size=%d hidden_size=%d "
                "c_features=%d c_depth=%d",
                i,
                len(missing),
                run_config.npod,
                run_config.latent_size,
                run_config.hidden_size,
                run_config.c_features,
                run_config.c_depth,
            )

            torch.manual_seed(run_config.seed)
            result = dmdaelib.run_once(run_config, dataset)

            with torch.no_grad():
                metrics, in_set = _predict_metrics(
                    result.model,
                    result.dmdc,
                    run_config,
                    zmask=dataset.zmask,
                    eval_mask=dataset.test_mask,
                )

            losses = result.losses
            record = {
                "config": asdict(run_config),
                "metrics": {
                    "dice": float(metrics.dice),
                    "iou": float(metrics.iou),
                    "dmdaelib.spectral_radius": float(metrics.a_radius),
                    "loss_train": float(losses[-1, 0, 0]),
                    "loss_val": float(losses[-1, 1, 0]),
                    "loss_test": float(result.loss_test[0]),
                },
                "losses": losses.detach().cpu().numpy(),
                "loss_test": result.loss_test.detach().cpu().numpy(),
                "train_metrics": {
                    name: value.detach().cpu().numpy()
                    for name, value in result.train_metrics._asdict().items()
                },
                "pred_mask": in_set.numpy().astype(np.uint8),
                "zmask": dataset.zmask.detach().cpu().numpy().astype(np.uint8),
            }
            results[run_config.suffix] = record

            tmpfile = outfile.with_name(f"{outfile.name}.tmp")
            torch.save(results, tmpfile)
            tmpfile.replace(outfile)

            log.info(
                "       -> Dice: %.4f | IoU: %.4f | |lam|max: %.3f | test loss: %.3e",
                record["metrics"]["dice"],
                record["metrics"]["iou"],
                record["metrics"]["dmdaelib.spectral_radius"],
                record["metrics"]["loss_test"],
            )

            del result
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        log.info("All requested sweep results are cached; only plotting.")

    current = {key: record for key, record in results.items() if key in current_keys}
    log.info(
        "Sweep results: '%s' (%d of %d runs match the current study).",
        outfile,
        len(current),
        len(results),
    )
    _print_sweep_summary(current, title=f"Sweep: {filename.stem} (m{config.dataidx})")
    _plot_sweep(outfile, current, overwrite=True)

    return 0


# }}}


if __name__ == "__main__":

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default device used by PyTorch",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Output sweep cache file (.pkl)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-run cached configurations",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only log warnings and errors",
    )
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "--dataidx",
        type=int,
        default=0,
        help="Index of the matrix to sweep on",
    )
    parser.add_argument(
        "--study",
        choices=[*SWEEP_STUDIES, "all"],
        default=argparse.SUPPRESS,
        help="Named study to run (default: all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Grid resolution per axis",
    )
    parser.add_argument(
        "--maxit",
        type=int,
        default=None,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs per stage",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        help="Parameter domain bounding box (xmin xmax ymin ymax)",
    )
    parser.add_argument(
        "--ridge-eps",
        type=float,
        default=None,
        help="Ridge regularization of the DMD solve",
    )
    parser.add_argument(
        "--dmd-method",
        choices=["pinv", "ridge"],
        default=None,
        help="DMD solver method",
    )
    parser.add_argument(
        "--c-alpha",
        type=float,
        default=None,
        help="Blending parameter alpha for ComplexBlendedQuadratic",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Horizons of the multi-step prediction stages",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=None,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--gamma-rec",
        type=float,
        default=None,
        help="Weight of the reconstruction loss",
    )
    parser.add_argument(
        "--gamma-lin",
        type=float,
        default=None,
        help="Weight of the latent linearity loss",
    )
    parser.add_argument(
        "--gamma-pre",
        type=float,
        default=None,
        help="Weight of the prediction loss",
    )

    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    with torch_default_device(args.device):
        torch.set_float32_matmul_precision("highest")

        overrides = {
            k: v
            for k, v in vars(args).items()
            if v is not None
            and k not in {"filename", "device", "outfile", "study", "force", "quiet"}
        }
        raise SystemExit(
            sweep(
                args.filename,
                outfile=args.outfile,
                study=getattr(args, "study", "all"),
                overwrite=args.force,
                **overrides,
            )
        )

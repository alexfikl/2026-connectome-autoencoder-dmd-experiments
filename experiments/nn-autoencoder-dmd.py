# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch

import dmdaelib
from commonlib import make_argument_parser
from nneuroutil.dmdc import DenseDMDc
from nneuroutil.helpers import machine_info
from nneuroutil.interop.torch import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)
from nneuroutil.visualization import figure

log = logging.getLogger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains or runs inference with a DMD Autoencoder: encode -> evolve linearly in
latent space -> decode.

The state is represented by polar observables: a unit direction in a POD basis
and the log of its norm. The latent operator is a ridge DMD fit on the encoded
training transitions. Training is staged: reconstruction only, then one-step
latent linearity, then multi-step prediction with increasing horizons.

Examples:

    > {SCRIPT_PATH.name} --device cuda train dataset.npz
    > {SCRIPT_PATH.name} --device cuda infer dataset-ae-dmd.pkl
"""


# {{{ helpers


def _plot_dataset(
    filename: pathlib.Path,
    zmask: torch.Tensor,
    *,
    bbox: tuple[float, float, float, float],
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(zmask.cpu().numpy(), extent=bbox, cmap="binary")


def _plot_loss(
    filename: pathlib.Path,
    losses: torch.Tensor,
    *,
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        losses_np = losses.cpu().numpy()
        labels = ("Rec", "Lin", "Pre")
        labels = labels[: losses_np.shape[2] - 1]

        for k, label in enumerate(labels):
            (line,) = ax.semilogy(losses_np[:, 0, 1 + k], label=f"Train. {label}")
            ax.semilogy(
                losses_np[:, 1, 1 + k],
                label=f"Valid. {label}",
                color=line.get_color(),
                ls="--",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(ncols=2)


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


def _plot_predict(
    filename: pathlib.Path,
    model: dmdaelib.DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    config: dmdaelib.DMDConfig,
    *,
    zmask: torch.Tensor,
    tag: str,
    overwrite: bool,
    plot: bool = True,
) -> dmdaelib.ErrorMetrics:
    metrics, in_set = _predict_metrics(model, dmdc, config, zmask=zmask)

    if not plot:
        return metrics

    log.info("    Converged:    %d/%d", torch.sum(in_set), in_set.numel())
    log.info("    IoU:          %.8e", metrics.iou)
    log.info("    Dice:         %.8e", metrics.dice)
    log.info("    |lam|max(A):  %.8e", metrics.a_radius)

    with figure(
        filename.with_stem(f"{filename.stem}-predict-{tag}"),
        overwrite=overwrite,
    ) as fig:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap

        ax = fig.gca()

        xmin, xmax, ymin, ymax = config.bbox
        zmask_np = zmask.cpu().numpy().astype(bool)
        in_set_np = in_set.numpy().astype(bool)
        ny, nx = zmask_np.shape
        dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0.0
        dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0.0
        x_edges = np.linspace(xmin - 0.5 * dx, xmax + 0.5 * dx, nx + 1)
        y_edges = np.linspace(ymin - 0.5 * dy, ymax + 0.5 * dy, ny + 1)

        # NOTE: distinguish the true positives from the false positives, i.e. the
        # predicted set outside of the true converged set, so that the latter
        # stand out from the correctly captured interior
        state = np.where(in_set_np, np.where(zmask_np, 2, 1), 0)
        ax.imshow(
            state,
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap=ListedColormap(["white", "0.35", "black"]),
            vmin=-0.5,
            vmax=2.5,
            interpolation="nearest",
        )
        ax.grid(visible=False, which="both")

        # NOTE: overlay the exact discrete cell boundary of the true converged set
        padded = np.pad(zmask_np, 1, constant_values=False)
        hi, hj = np.nonzero(padded[1:, 1:-1] ^ padded[:-1, 1:-1])
        vi, vj = np.nonzero(padded[1:-1, 1:] ^ padded[1:-1, :-1])

        h_segs = np.stack(
            [
                np.column_stack([x_edges[hj], y_edges[hi]]),
                np.column_stack([x_edges[hj + 1], y_edges[hi]]),
            ],
            axis=1,
        )
        v_segs = np.stack(
            [
                np.column_stack([x_edges[vj], y_edges[vi]]),
                np.column_stack([x_edges[vj], y_edges[vi + 1]]),
            ],
            axis=1,
        )
        segments = np.concatenate([h_segs, v_segs], axis=0)
        ax.add_collection(LineCollection(segments, colors="red", linewidths=2.0))  # ty: ignore[invalid-argument-type]

    return metrics


# }}}


# {{{ plot metrics


def _plot_error_metrics(
    filename: pathlib.Path,
    metrics: dmdaelib.ErrorMetrics,
    *,
    final: dmdaelib.ErrorMetrics | None = None,
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-error"),
        ncols=2,
        overwrite=overwrite,
    ) as fig:
        ax_set, ax_spec = fig.axes
        epochs = np.arange(metrics.iou.numel())

        ax_set.plot(epochs, metrics.iou.cpu().numpy(), label="IoU")
        ax_set.plot(epochs, metrics.dice.cpu().numpy(), label="Dice")
        ax_set.set_xlabel("Epoch")
        ax_set.set_ylabel("Score")
        ax_set.set_ylim(0.0, 1.0)
        ax_set.legend()

        if final is not None:
            ax_set.axhline(float(final.iou), color="C0", ls=":", lw=1.0)
            ax_set.axhline(float(final.dice), color="C1", ls=":", lw=1.0)

        ax_spec.plot(
            epochs,
            metrics.a_radius.cpu().numpy(),
            label=r"$|\lambda|_{\max}(A)$",
        )
        ax_spec.set_xlabel("Epoch")
        ax_spec.set_ylabel("Magnitude")
        ax_spec.legend()

        if final is not None:
            ax_spec.axhline(float(final.a_radius), color="C0", ls=":", lw=1.0)


def _plot_train_metrics(
    filename: pathlib.Path,
    metrics: dmdaelib.TrainMetrics,
    *,
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-train"),
        ncols=2,
        overwrite=overwrite,
    ) as fig:
        ax_lr, ax_validate = fig.axes
        epochs = np.arange(metrics.lr.numel())

        ax_lr.semilogy(epochs, metrics.lr.cpu().numpy(), label="LR")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.legend()

        ax_validate.semilogy(
            epochs, metrics.lin_validate.cpu().numpy(), label="Linearity"
        )
        ax_validate.semilogy(
            epochs, metrics.pre_validate.cpu().numpy(), label="Prediction"
        )
        ax_validate.set_xlabel("Epoch")
        ax_validate.set_ylabel("Validation")
        ax_validate.legend()


# }}}


# {{{ train


def train(
    filename: pathlib.Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    weights: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from commonlib import load_matrix_datasets

    machine = machine_info(SCRIPT_PATH)
    log.info("Machine:\n%s", machine.pretty())

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    provenance = {**machine._asdict(), "device": device}

    # {{{ load parameters

    if overrides is None:
        overrides = {}

    checkpoint: dict[str, Any] | None = None
    if weights is not None:
        if not weights.exists():
            log.error("Weights file does not exist: '%s'.", weights)
            return 1

        checkpoint = torch.load(weights, map_location=device, weights_only=True)
        config = dmdaelib.DMDConfig.from_sources(checkpoint["config"], overrides)
        mat = checkpoint["mat"]
    else:
        config = dmdaelib.DMDConfig.from_sources(overrides)

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

    mem.add_record("After data load")
    log.info("Config:\n%s", config)

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-ae-dmd.npz"
        outfile = outfile.with_stem(f"{outfile.stem}-{config.suffix}")

    outfile = outfile.with_suffix("")
    weightsfile = outfile.with_suffix(".pkl")

    if not overwrite and weightsfile.exists():
        log.error(
            "Weights file already exist (use --force to overwrite): '%s'", weightsfile
        )
        return 1

    # }}}

    # {{{ model

    d = mat.shape[0]

    # NOTE: `config.seed` already seeds the numpy split generator; seed torch as
    # well so that the model initialization and shuffling are reproducible
    log.info("Using seed: %d", config.seed)
    torch.manual_seed(config.seed)

    model = dmdaelib.make_dmd_autoencoder(
        config, d, device=device, dtype=mat.dtype.to_real()
    )
    log.info("Network:\n%s", model)

    # NOTE: the staged schedule restarts from the loaded weights
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as exc:
            log.error(
                "Cannot resume from '%s': the model architecture does not match "
                "the current configuration.\n%s",
                weights,
                exc,
            )
            return 1
        log.info("Initialized from '%s'", weights)

    # }}}

    # {{{ dataset

    dataset = dmdaelib.prepare_dataset(config, mat)

    if checkpoint is None:
        model.fit_observables(dataset.x_train)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    stages = dmdaelib.make_stages(config)
    num_epochs = config.epochs * len(stages)
    error_metrics = dmdaelib.empty_error_metrics(num_epochs)

    def plot_predict(
        epoch: int, stage: dmdaelib.Stage, model: dmdaelib.DMDAutoencoder, *, last: bool
    ) -> None:
        if not stage.use_dmd:
            return

        dmdc = dmdaelib.build_latent_dmd(model, dataset.x_train, dataset.c_train)
        metrics = _plot_predict(
            outfile,
            model,
            dmdc,
            config,
            zmask=dataset.zmask,
            tag=stage.name,
            overwrite=overwrite,
            plot=last,
        )

        error_metrics.iou[epoch] = metrics.iou
        error_metrics.dice[epoch] = metrics.dice
        error_metrics.a_radius[epoch] = metrics.a_radius

    result = dmdaelib.run_once(
        config, dataset, model=model, stages=stages, callback=plot_predict
    )
    model = result.model
    losses = result.losses
    train_metrics = result.train_metrics
    dmdc = result.dmdc
    loss_test = result.loss_test
    num_epochs = len(losses)

    mem.add_record("After training")

    log.info(
        "[%03d/%03d] test_total %.5e test_rec %.5e test_lin %.5e test_pre %.5e",
        num_epochs,
        num_epochs,
        loss_test[0].item(),
        loss_test[1].item(),
        loss_test[2].item(),
        loss_test[3].item(),
    )

    final_error_metrics = _plot_predict(
        outfile,
        model,
        dmdc,
        config,
        zmask=dataset.zmask,
        tag="final",
        overwrite=overwrite,
    )

    mem.add_record("After test loss")

    # }}}

    # {{{ checkpoint

    from dataclasses import asdict

    torch.save(
        {
            # model
            "provenance": provenance,
            "model": model.state_dict(),
            "dmdc": {"A": dmdc.A, "B": dmdc.B},
            # loss and metrics
            "num_epochs_trained": len(losses),
            "losses": losses,
            "loss_test": loss_test,
            "train_metrics": train_metrics._asdict(),
            "error_metrics": error_metrics._asdict(),
            "final_error_metrics": final_error_metrics._asdict(),
            # dataset
            "inputfile": str(filename),
            "mat": mat,
            "zmask": dataset.zmask,
            # parameters
            "config": asdict(config),
        },
        weightsfile,
    )
    log.info("Saved checkpoint: '%s'.", weightsfile)

    mem.add_record("After checkpoint")

    # }}}

    # {{{ plot

    _plot_dataset(outfile, dataset.zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, overwrite=overwrite)
    _plot_train_metrics(outfile, train_metrics, overwrite=overwrite)
    _plot_error_metrics(
        outfile,
        error_metrics,
        final=final_error_metrics,
        overwrite=overwrite,
    )

    # }}}

    mem.add_record("Finished")
    log.info("Memory:\n%s", mem)

    return 0


# }}}


# {{{ infer


def infer(
    weightsfile: pathlib.Path,
    overrides: Mapping[str, Any],
    *,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not weightsfile.exists():
        log.error("Weights file does not exist: '%s'.", weightsfile)
        return 1

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    # {{{ load parameters

    if overrides is None:
        overrides = {}

    checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)
    config = dmdaelib.DMDConfig.from_sources(checkpoint["config"], overrides)

    losses = checkpoint["losses"]
    zmask = checkpoint["zmask"]
    dmdc = dmdaelib.load_latent_dmd(checkpoint)
    train_metrics = checkpoint.get("train_metrics")
    error_metrics = checkpoint.get("error_metrics")

    if outfile is None:
        outfile = weightsfile.with_suffix("")
    outfile = outfile.with_suffix("")

    # }}}

    # {{{ load model

    model = dmdaelib.make_dmd_autoencoder(
        config,
        checkpoint["mat"].shape[0],
        device=device,
        dtype=torch.float32,
    )
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        log.error(
            "Cannot load '%s': the model architecture does not match the "
            "checkpoint.\n%s",
            weightsfile,
            exc,
        )
        return 1
    model.eval()

    mem.add_record("After model load")

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, overwrite=overwrite)

    final_metrics = _plot_predict(
        outfile,
        model,
        dmdc,
        config,
        zmask=zmask,
        tag="final",
        overwrite=overwrite,
    )

    if train_metrics is not None:
        _plot_train_metrics(
            outfile, dmdaelib.TrainMetrics(**train_metrics), overwrite=overwrite
        )

    if error_metrics is not None:
        _plot_error_metrics(
            outfile,
            dmdaelib.ErrorMetrics(**error_metrics),
            final=final_metrics,
            overwrite=overwrite,
        )

    # }}}

    mem.add_record("Finished")

    log.info("Memory:\n%s", mem)

    return 0


# }}}


if __name__ == "__main__":
    parser, _ = make_argument_parser(
        description=SCRIPT_LONG_HELP,
        devices=available_device_names(),
    )
    parser.add_argument(
        "--ridge-eps",
        type=float,
        default=None,
        help=(
            "Ridge regularization of the DMD solve, relative to the mean diagonal "
            "of the Gram matrix (default: 1e-4, from checkpoint when resuming)"
        ),
    )
    parser.add_argument(
        "--dmd-method",
        choices=["pinv", "ridge"],
        default=None,
        help="DMD solver method: 'pinv' (truncated SVD) or 'ridge' (L2-regularized)",
    )
    parser.add_argument(
        "--c-features",
        type=int,
        default=None,
        help="Number of complex features of c in the encoder (default: 32)",
    )
    parser.add_argument(
        "--c-depth",
        type=int,
        default=None,
        help="Number of (complex linear, complex square) blocks for c (default: 1)",
    )
    parser.add_argument(
        "--c-alpha",
        type=float,
        default=None,
        help="Blending parameter alpha for ComplexBlendedQuadratic (default: 0.5)",
    )
    parser.add_argument(
        "--npod",
        type=int,
        default=None,
        help="Number of POD directions used to represent the state (default: 24)",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=None,
        help="Size of the latent space (default: 48)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden width of the encoder and decoder (default: 256)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Horizons of the multi-step prediction stages (default: 2 4 8 maxit-1)",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=None,
        help="Gradient clipping norm (default: 5)",
    )
    parser.add_argument(
        "--gamma-rec",
        type=float,
        default=None,
        help="Weight of the reconstruction loss (default: 1)",
    )
    parser.add_argument(
        "--gamma-lin",
        type=float,
        default=None,
        help="Weight of the latent linearity loss (default: 1)",
    )
    parser.add_argument(
        "--gamma-pre",
        type=float,
        default=None,
        help="Weight of the prediction loss (default: 1)",
    )

    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    with torch_default_device(args.device):
        # NOTE: the ridge DMD solve needs full float32 precision (no TF32)
        torch.set_float32_matmul_precision("highest")

        if args.command == "train":
            raise SystemExit(
                train(
                    args.filename,
                    overrides=vars(args),
                    weights=args.weights,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        elif args.command == "infer":
            raise SystemExit(
                infer(
                    args.filename,
                    overrides=vars(args),
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise SystemExit(1)

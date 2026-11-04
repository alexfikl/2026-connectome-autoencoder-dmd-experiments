# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools
import logging
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import array_api_compat
import dmdaelib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from commonlib import make_argument_parser
from nneuroutil.array_api_extras import default_rng
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


# {{{ features


def compute_features(
    model: dmdaelib.DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    *,
    bbox: tuple[float, float, float, float],
    resolution: int,
    maxit: int,
) -> dict[str, np.ndarray]:
    r"""Compute features of the trained model for classification.

    The latent basis is arbitrary (every matrix has its own encoder), so the
    features are either invariant to it or are lifted to the state space, where
    the node coordinates are shared between matrices.

    * ``eigvals``: the eigenvalues of the operator ``dmdc.A``, sorted by decreasing
      modulus.
    * ``jac_map``, ``jac_stats``: the gradient norm
      :math:`|\nabla_c \log |z_N||` of the decoded final log-norm with respect to
      the (physical) control, on the grid, and its (mean, std, median, p90, max).
    * ``inset_fraction``: fraction of the grid predicted in the set (decoded final
      norm below the escape radius). The following describe the decoded final states
      of these points, which approximate the attractors in the node space.
    * ``attractor_node_magnitude``: ``(d,)`` mean over the points of the per-node
      magnitude ``|z_i|`` of the final state.
    * ``attractor_node_phase``: ``(2, d)`` mean over the points of the unit phasor
      ``z_i / |z_i| * conj(c / |c|)``, i.e. the phase of each node relative to the
      control, as its (cos, sin) components. The modulus of the mean is the
      coherence of the phase over the points.
    * ``attractor_energy``: the first 8 normalized squared singular values of the
      final states (uncentered), i.e. how many directions the attractors span.
    """
    scale = max(abs(b) for b in bbox)

    # {{{ spectrum

    lam = torch.linalg.eigvals(dmdc.A.detach().cpu().to(torch.float64))  # ty: ignore[unresolved-attribute]
    lam = lam[torch.argsort(lam.abs(), descending=True)]

    # }}}

    # {{{ sensitivity to the control

    c = dmdaelib.grid_control(model, bbox, resolution, like=dmdc.A)  # ty: ignore[invalid-argument-type]
    c = c.detach().clone().requires_grad_()
    g_final = model.rollout(dmdaelib.encode_origin(model, c), dmdc, maxit)[:, -1]
    _, logn_final = model.decode(g_final)
    (grad,) = torch.autograd.grad(logn_final.sum(), c)

    # NOTE: `c` is normalized by `scale` and the log-norm by `dmdaelib.LOG_SCALE`
    jac = torch.linalg.vector_norm(grad, dim=-1) * (dmdaelib.LOG_SCALE / scale)
    jac = jac.detach().cpu()
    jac_stats = torch.stack([
        jac.mean(),
        jac.std(),
        jac.median(),
        torch.quantile(jac, 0.9),
        jac.max(),
    ])

    # }}}

    # {{{ attractors

    # NOTE: states of the in-set points at the end of the rollout, rebuilt from
    # the decoded direction (unit, in the orthonormal POD basis) and norm
    with torch.no_grad():
        g_last = model.rollout(dmdaelib.encode_origin(model, c.detach()), dmdc, maxit)
        direction, logn = model.decode(g_last[:, -1])
        norm = (torch.exp(logn * dmdaelib.LOG_SCALE) - dmdaelib.LOG_EPS).clamp(min=0.0)
        states = norm[:, None] * (direction @ cast(torch.Tensor, model.basis).T)

        inset = norm < model.escape_radius
        final = states[inset].cpu().to(torch.float64)
        c_inset = c.detach()[inset].cpu().to(torch.float64)

    nenergy = 8
    inset_fraction = torch.tensor(float(inset.float().mean()), device="cpu")
    nnodes = model.input_size
    attractor_node_magnitude = torch.zeros(nnodes, dtype=torch.float64, device="cpu")
    attractor_node_phase = torch.zeros((2, nnodes), dtype=torch.float64, device="cpu")
    attractor_energy = torch.zeros(nenergy, dtype=torch.float64, device="cpu")

    if final.shape[0] > 1:
        z = torch.view_as_complex(final.reshape(final.shape[0], -1, 2).contiguous())
        zc = torch.view_as_complex(c_inset.contiguous())

        attractor_node_magnitude = z.abs().mean(dim=0)

        phasor = z / z.abs().clamp(min=1.0e-12)
        phasor = phasor * (zc / zc.abs().clamp(min=1.0e-12)).conj()[:, None]
        phasor = phasor.mean(dim=0)
        attractor_node_phase = torch.stack([phasor.real, phasor.imag])

        energy = torch.linalg.svdvals(final) ** 2
        energy = energy[:nenergy] / energy.sum()
        attractor_energy[: energy.shape[0]] = energy

    # }}}

    return {
        "eigvals": lam.numpy(),
        "jac_map": jac.reshape(resolution, resolution).numpy(),
        "jac_stats": jac_stats.numpy(),
        "inset_fraction": inset_fraction.numpy(),
        "attractor_node_magnitude": attractor_node_magnitude.numpy(),
        "attractor_node_phase": attractor_node_phase.numpy(),
        "attractor_energy": attractor_energy.numpy(),
    }


def make_feature_vector(
    feats: Mapping[str, np.ndarray], *, neig: int = 3, nenergy: int = 3
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Gather the useful features of :func:`compute_features` in a single vector.

    The format matches ``make-features.py``: the feature vector, the boundaries of
    the feature groups (``ngroups + 1``) and a description of each group. The node
    magnitudes carry most of the information; the small groups are secondary. The
    node phases and ``inset_fraction`` are left out, since they are nearly constant
    and a shape quantity, respectively.
    """
    # NOTE: the operator is real, so the eigenvalues come in conjugate pairs that
    # have the same modulus; we only keep one of each
    eigvals = feats["eigvals"]
    moduli = np.sort(np.abs(eigvals[eigvals.imag >= 0]))[::-1][:neig]

    groups = {
        "Leading eigenvalue moduli": moduli,
        "Control sensitivity (log)": np.log(feats["jac_stats"]),
        "Attractor node magnitudes": feats["attractor_node_magnitude"],
        "Attractor energy": feats["attractor_energy"][:nenergy],
    }

    x = np.concatenate([np.ravel(v) for v in groups.values()]).astype(np.float64)
    starts = np.cumsum([0, *(np.size(v) for v in groups.values())])
    if not np.all(np.isfinite(x)):
        raise ValueError("features are not finite")

    return x, starts, tuple(groups)


# }}}


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


def _predict_metrics(
    model: dmdaelib.DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    config: dmdaelib.DMDConfig,
    *,
    zmask: torch.Tensor,
) -> tuple[dmdaelib.ErrorMetrics, torch.Tensor]:
    norm = dmdaelib.torch_predict(
        model,
        dmdc,
        bbox=config.bbox,
        resolution=config.resolution,
        maxit=config.maxit,
    )

    from nneuroutil.scores import dice_score, jaccard_index

    in_set = norm < config.escape_radius
    metrics = dmdaelib.ErrorMetrics(
        iou=cast(torch.Tensor, jaccard_index(in_set, zmask)).detach().cpu(),
        dice=cast(torch.Tensor, dice_score(in_set, zmask)).detach().cpu(),
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
        ax = fig.gca()
        ax.imshow(in_set.numpy(), extent=config.bbox, cmap="binary")

        # NOTE: overlay the true converged set as a red outline
        xmin, xmax, ymin, ymax = config.bbox
        zmask = zmask.cpu().numpy()
        xs = np.linspace(xmin, xmax, zmask.shape[1])
        ys = np.linspace(ymin, ymax, zmask.shape[0])
        ax.contour(xs, ys, np.flipud(zmask), levels=[0.5], colors="red", linewidths=1.0)

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
    from commonlib import (
        generate_grid,
        generate_timeseries_dataset,
        load_matrix_datasets,
    )
    from nneuroutil.data import random_split

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

    # NOTE: when resuming, the stored model configuration is used
    model_config: dict[str, Any]
    if checkpoint is not None:
        model_config = checkpoint["model_config"]
    else:
        model_config = {
            "input_size": d,
            "npod": config.npod,
            "latent_size": config.latent_size,
            "hidden_size": config.hidden_size,
            "c_features": config.c_features,
            "c_depth": config.c_depth,
            "c_alpha": config.c_alpha,
            "escape_radius": config.escape_radius,
            "dmd_method": config.dmd_method,
            "ridge_eps": config.ridge_eps,
        }

    model = dmdaelib.DMDAutoencoder(
        **model_config, device=device, dtype=mat.dtype.to_real()
    )
    log.info("Network:\n%s", model)

    # NOTE: the staged schedule restarts from the loaded weights
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        log.info("Initialized from '%s'", weights)

    # }}}

    # {{{ generate dataset

    # NOTE: `min_escape_iters=0` keeps all the trajectories, including the ones
    # that escape early: they carry the escape dynamics
    x, _, mask, zmask = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        generate_timeseries_dataset(
            mat,
            bbox=config.bbox,
            resolution=config.resolution,
            # NOTE: we add 1 so that x.shape == (n, maxit, 2d)
            maxit=config.maxit + 1,
            escape_radius=config.escape_radius,
            min_escape_iters=0,
        ),
    )

    c = generate_grid(
        *config.bbox,
        nx=config.resolution,
        ny=config.resolution,
        dtype=mat.dtype,
        xp=array_api_compat.array_namespace(mat),
        device=device,
    )

    # NOTE: the control is constant along a trajectory, so it is kept per
    # realization (shape `(n, 2)`) instead of being replicated over the window
    c = dmdaelib.normalize_control(c[mask], config.bbox)

    (x_train, c_train), (x_validate, c_validate), (x_test, c_test) = cast(
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        random_split(
            (x, c), config.split, rng=default_rng(torch, config.seed, device=device)
        ),
    )

    if checkpoint is None:
        model.fit_observables(x_train)

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

        dmdc = dmdaelib.build_latent_dmd(model, x_train, c_train)
        metrics = _plot_predict(
            outfile,
            model,
            dmdc,
            config,
            zmask=zmask,
            tag=stage.name,
            overwrite=overwrite,
            plot=last,
        )

        error_metrics.iou[epoch] = metrics.iou
        error_metrics.dice[epoch] = metrics.dice
        error_metrics.a_radius[epoch] = metrics.a_radius

    criterion = dmdaelib.DMDLoss(
        logn_clip=model.logn_clip,
        gamma_rec=config.gamma_rec,
        gamma_lin=config.gamma_lin,
        gamma_pre=config.gamma_pre,
    )

    _, losses, train_metrics = dmdaelib.torch_train(
        model,
        criterion,
        x_train,
        x_validate,
        c_train=c_train,
        c_validate=c_validate,
        stages=stages,
        lr=config.lr,
        max_norm=config.max_norm,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callback=plot_predict,
    )
    dmdc = dmdaelib.build_latent_dmd(model, x_train, c_train)

    mem.add_record("After training")

    loss_test = dmdaelib.evaluate(
        model,
        criterion,
        DataLoader(TensorDataset(x_test, c_test), batch_size=config.batch_size),
        dmdc,
        stages[-1].horizon,
    )
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
        zmask=zmask,
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
            "model_config": model_config,
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
            "zmask": zmask,
            # parameters
            "config": asdict(config),
        },
        weightsfile,
    )
    log.info("Saved checkpoint: '%s'.", weightsfile)

    mem.add_record("After checkpoint")

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
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
    features: bool = False,
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

    model_config = checkpoint["model_config"]
    model = dmdaelib.DMDAutoencoder(**model_config, device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mem.add_record("After model load")

    # }}}

    # {{{ features

    if features:
        featfile = outfile.with_name(f"{outfile.name}-features.npz")
        if not overwrite and featfile.exists():
            log.error("Features file exists (use --force to overwrite): '%s'", featfile)
            return 1

        feats = compute_features(
            model,
            dmdc,
            bbox=config.bbox,
            resolution=config.resolution,
            maxit=config.maxit,
        )
        feats["dmdc_A"] = dmdc.A.detach().cpu().numpy()  # ty: ignore[unresolved-attribute]
        feats["dmdc_B"] = dmdc.B.detach().cpu().numpy()  # ty: ignore[unresolved-attribute]
        feats["x"], feats["starts"], descrs = make_feature_vector(feats)
        feats["descrs"] = np.array(descrs)
        if "loss_test" in checkpoint:
            feats["loss_test"] = checkpoint["loss_test"].detach().cpu().numpy()

        eigvals = feats["eigvals"]
        log.info(
            "|lambda| (top 8): %s", np.array2string(np.abs(eigvals[:8]), precision=3)
        )
        log.info("jacobian (mean, std, median, p90, max): %s", feats["jac_stats"])

        np.savez(featfile, allow_pickle=True, **feats)
        log.info("Saved features: '%s'.", featfile)

        return 0

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


# {{{ merge


def _natural_sort_key(name: str) -> str:
    return re.sub(r"(\d+)", lambda match: match.group(1).zfill(20), name)


def merge(
    indirs: Sequence[pathlib.Path],
    *,
    pattern: str = "*-features.npz",
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not indirs:
        log.warning("No directories given.")
        return 0

    if outfile is None:
        outfile = indirs[0].parent / "dataset-features-ae-dmd.npz"
    outfile = outfile.with_suffix(".npz")

    if not overwrite and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
        return 1

    datasets = {}
    starts = None
    descrs = None

    for indir in indirs:
        if not indir.is_dir():
            log.error("Not a directory: '%s'.", indir)
            return 1

        name = indir.name
        if name in datasets:
            log.error("Duplicate dataset name: '%s'.", name)
            return 1

        found = sorted(
            indir.glob(pattern), key=lambda path: _natural_sort_key(path.name)
        )
        if not found:
            log.error("No files matching '%s' in '%s'.", pattern, indir)
            return 1

        xs = []
        for filename in found:
            with np.load(filename) as data:
                if starts is None:
                    starts, descrs = data["starts"], data["descrs"]

                if not (
                    np.array_equal(data["starts"], starts)
                    and np.array_equal(data["descrs"], descrs)
                ):
                    log.error("Inconsistent feature groups: '%s'.", filename)
                    return 1

                xs.append(data["x"])

        datasets[name] = np.stack(xs)
        log.info("Gathered %d matrices for '%s'.", len(xs), name)

    np.savez(outfile, datasets=datasets, starts=starts, descrs=descrs)
    log.info("Saved merged features: '%s'.", outfile)

    return 0


# }}}


# {{{ sweep

SWEEP_STUDIES: dict[str, dict[str, Sequence[int]]] = {
    "latent": {
        "npod": (24, 36, 48),
        "latent_size": (48, 64, 96, 128),
        "hidden_size": (256,),
        "c_features": (16,),
        "c_depth": (2,),
    },
    "control": {
        "npod": (48,),
        "latent_size": (128,),
        "hidden_size": (256,),
        "c_features": (0, 4, 8, 16, 32),
        "c_depth": (1, 2, 4),
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
    table.add_column("Dice", justify="right")
    table.add_column("IoU", justify="right")
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
            ("dice", "Dice", False, (0.0, 1.0)),
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
    from dataclasses import asdict, replace

    from commonlib import (
        generate_grid,
        generate_timeseries_dataset,
        load_matrix_datasets,
    )
    from nneuroutil.data import random_split

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

        d = mat.shape[0]

        # NOTE: the dataset is generated once and shared by all runs
        x, _, mask, zmask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            generate_timeseries_dataset(
                mat,
                bbox=config.bbox,
                resolution=config.resolution,
                maxit=config.maxit + 1,
                escape_radius=config.escape_radius,
                min_escape_iters=0,
            ),
        )

        c = generate_grid(
            *config.bbox,
            nx=config.resolution,
            ny=config.resolution,
            dtype=mat.dtype,
            xp=array_api_compat.array_namespace(mat),
            device=device,
        )
        c = dmdaelib.normalize_control(c[mask], config.bbox)

        (x_train, c_train), (x_validate, c_validate), (x_test, c_test) = cast(
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
            random_split(
                (x, c),
                config.split,
                rng=default_rng(torch, config.seed, device=device),
            ),
        )

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
            model = dmdaelib.DMDAutoencoder(
                input_size=d,
                npod=run_config.npod,
                latent_size=run_config.latent_size,
                hidden_size=run_config.hidden_size,
                c_features=run_config.c_features,
                c_depth=run_config.c_depth,
                c_alpha=run_config.c_alpha,
                escape_radius=run_config.escape_radius,
                dmd_method=run_config.dmd_method,
                ridge_eps=run_config.ridge_eps,
                device=device,
                dtype=mat.dtype.to_real(),
            )
            model.fit_observables(x_train)

            criterion = dmdaelib.DMDLoss(
                logn_clip=model.logn_clip,
                gamma_rec=run_config.gamma_rec,
                gamma_lin=run_config.gamma_lin,
                gamma_pre=run_config.gamma_pre,
            )
            stages = dmdaelib.make_stages(run_config)

            _, losses, train_metrics = dmdaelib.torch_train(
                model,
                criterion,
                x_train,
                x_validate,
                c_train=c_train,
                c_validate=c_validate,
                stages=stages,
                lr=run_config.lr,
                max_norm=run_config.max_norm,
                epochs=run_config.epochs,
                batch_size=run_config.batch_size,
            )

            with torch.no_grad():
                dmdc = dmdaelib.build_latent_dmd(model, x_train, c_train)
                metrics, in_set = _predict_metrics(model, dmdc, run_config, zmask=zmask)

                test_loader = DataLoader(
                    TensorDataset(x_test, c_test),
                    batch_size=run_config.batch_size,
                )
                loss_test = dmdaelib.evaluate(
                    model,
                    criterion,
                    test_loader,
                    dmdc,
                    stages[-1].horizon,
                )

            record = {
                "config": asdict(run_config),
                "metrics": {
                    "dice": float(metrics.dice),
                    "iou": float(metrics.iou),
                    "dmdaelib.spectral_radius": float(metrics.a_radius),
                    "loss_train": float(losses[-1, 0, 0]),
                    "loss_val": float(losses[-1, 1, 0]),
                    "loss_test": float(loss_test[0]),
                },
                "losses": losses.detach().cpu().numpy(),
                "loss_test": loss_test.detach().cpu().numpy(),
                "train_metrics": {
                    name: value.detach().cpu().numpy()
                    for name, value in train_metrics._asdict().items()
                },
                "pred_mask": in_set.numpy().astype(np.uint8),
                "zmask": zmask.detach().cpu().numpy().astype(np.uint8),
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

            del model
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
    parser, subparsers = make_argument_parser(
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
        help="Number of complex features of c in the encoder (default: 0, disabled)",
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
        help="Size of the latent space (default: 32)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden width of the encoder and decoder (default: 128)",
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
    parser.add_argument(
        "--features",
        action="store_true",
        help=(
            "With 'infer': compute classification features (spectrum, modes, "
            "strengths, control sensitivity), save them to '{basename}-features.npz' "
            "and skip the plots"
        ),
    )

    subparser = subparsers.add_parser(
        "merge",
        formatter_class=parser.formatter_class,
        description=(
            "Merge the features written by 'infer --features' into a single file "
            "for make-features.py consumers. Each given directory is treated as one "
            "dataset (class), named after the directory, and all its matching files "
            "are stacked together."
        ),
    )
    subparser.add_argument(
        "indirs",
        nargs="+",
        type=pathlib.Path,
        help="Directory containing the features of a single dataset",
    )
    subparser.add_argument(
        "--pattern",
        default="*-features.npz",
        help="Glob pattern used to find the feature files in each directory",
    )

    subparser = subparsers.add_parser(
        "sweep",
        formatter_class=parser.formatter_class,
        description=(
            "Train missing configurations of a hyperparameter sweep and plot "
            "all available results. Existing runs are reused unless --force is "
            "given."
        ),
    )
    subparser.add_argument("filename", type=pathlib.Path)
    subparser.add_argument(
        "--dataidx",
        type=int,
        default=0,
        help="Index of the matrix to sweep on",
    )
    subparser.add_argument(
        "--study",
        choices=["latent", "control", "all"],
        default="all",
        help="Named study to run",
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
                    features=args.features,
                )
            )
        elif args.command == "merge":
            raise SystemExit(
                merge(
                    args.indirs,
                    pattern=args.pattern,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        elif args.command == "sweep":
            fixed = ("npod", "latent_size", "hidden_size", "c_features", "c_depth")
            unsupported = {
                name: getattr(args, name)
                for name in fixed
                if getattr(args, name) is not None
            }
            if unsupported:
                log.error(
                    "Sweep axes are fixed by the study; unsupported overrides: %s",
                    unsupported,
                )
                raise SystemExit(1)

            overrides = {
                k: v
                for k, v in vars(args).items()
                if v is not None
                and k
                not in {
                    "command",
                    "filename",
                    "device",
                    "outfile",
                    *fixed,
                    "study",
                    "force",
                }
            }
            raise SystemExit(
                sweep(
                    args.filename,
                    outfile=args.outfile,
                    study=args.study,
                    overwrite=args.force,
                    **overrides,
                )
            )
        else:
            raise SystemExit(1)

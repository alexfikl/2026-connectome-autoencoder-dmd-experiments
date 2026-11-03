# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, cast

import array_api_compat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from commonlib import Config, compute_trajectory_mask, make_argument_parser
from nneuroutil.array_api_extras import default_rng
from nneuroutil.dmdc import DenseDMDc, build_dense_dmdc
from nneuroutil.helpers import (
    TicTocTimer,
    machine_info,
)
from nneuroutil.interop.torch import (
    ComplexLinear,
    ComplexQuadratic,
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

# NOTE: the log-norm of a state is ``log(|z| + LOG_EPS) / LOG_SCALE``
LOG_EPS = 1.0e-2
LOG_SCALE = 3.0


# {{{ configuration


@dataclass(frozen=True, slots=True)
class DMDConfig(Config):
    """Configuration for the DMD autoencoder network."""

    # NOTE: `epochs` is the number of epochs *per stage*
    epochs: int = 60
    lr: float = 2.0e-3
    batch_size: int = 512

    resolution: int = 64
    maxit: int = 16
    escape_radius: float = 100.0

    max_norm: float = 5.0
    ridge_eps: float = 1.0e-4
    dmd_method: Literal["pinv", "ridge"] = "ridge"

    # weights of the reconstruction, latent linearity, and prediction losses
    gamma_rec: float = 1.0
    gamma_lin: float = 1.0
    gamma_pre: float = 1.0

    # number of POD directions, latent size, and hidden width of the networks
    npod: int = 24
    latent_size: int = 32
    hidden_size: int = 128
    # number of complex features of the control c (0 disables the branch) and the
    # number of (complex linear, complex square) blocks in it
    c_features: int = 0
    c_depth: int = 1
    # horizons of the multi-step prediction stages (None -> 2, 4, 8, maxit - 1)
    horizons: tuple[int, ...] | None = None


class Stage(NamedTuple):
    name: str
    horizon: int
    """Number of latent steps in the prediction loss (0 to disable)."""
    use_dmd: bool
    """Whether the latent operator and the linearity loss are active."""


def make_stages(config: DMDConfig) -> tuple[Stage, ...]:
    horizons = config.horizons
    if horizons is None:
        horizons = (2, 4, 8, config.maxit - 1)

    horizons = tuple(sorted({h for h in horizons if 0 < h <= config.maxit - 1}))
    return (
        Stage("rec", horizon=0, use_dmd=False),
        Stage("lin", horizon=0, use_dmd=True),
        *(Stage(f"pre{h}", horizon=h, use_dmd=True) for h in horizons),
    )


# }}}


# {{{ model


class Observables(NamedTuple):
    direction: torch.Tensor
    """Unit direction in the POD basis of shape ``(batch, steps, npod)``.

    Pinned (clipped) and zero states have a zero direction.
    """
    logn: torch.Tensor
    """Scaled log-norm of shape ``(batch, steps)``. Pinned states hold the clip."""
    active: torch.Tensor
    """Unpinned states of shape ``(batch, steps)``."""
    transition: torch.Tensor
    """Transitions between unpinned states of shape ``(batch, steps - 1)``."""


class DMDAutoencoderResult(NamedTuple):
    obs: Observables
    """Polar observables of the input states."""
    enc: torch.Tensor
    """Encoded latent states of shape ``(batch, steps, latent_size)``."""
    rec_direction: torch.Tensor
    """Decoded direction of the encoded states."""
    rec_logn: torch.Tensor
    """Decoded log-norm of the encoded states."""
    lin_pred: torch.Tensor | None
    """One-step latent prediction ``g_n A`` of shape ``(batch, steps - 1, r)``."""
    roll_direction: torch.Tensor | None
    """Decoded direction of the latent rollout from the initial state."""
    roll_logn: torch.Tensor | None
    """Decoded log-norm of the latent rollout from the initial state."""


def _load_latent_dmd(checkpoint: Mapping[str, Any], latent_size: int) -> DenseDMDc[Any]:
    """Load the latent DMDc model, including legacy ``[[A, 0], [b, 1]]`` checkpoints."""
    if "dmdc" in checkpoint:
        return DenseDMDc(A=checkpoint["dmdc"]["A"], B=checkpoint["dmdc"]["B"])

    W = checkpoint["a_dmd"]
    return DenseDMDc(A=W[:latent_size, :latent_size], B=W[latent_size:, :latent_size])


class DMDAutoencoder(nn.Module):
    r"""Autoencoder with polar observables and a linear latent operator.

    The encoder maps ``(direction, log-norm, c)`` of a state to a latent vector
    ``g`` and the latent evolves as the affine map ``g -> g A + b``, fit as a DMDc
    model with a constant unit input. The decoder returns a direction and a
    log-norm.
    """

    def __init__(
        self,
        input_size: int,
        *,
        npod: int,
        latent_size: int,
        hidden_size: int,
        escape_radius: float,
        c_features: int = 0,
        c_depth: int = 1,
        dmd_method: Literal["pinv", "ridge"] = "ridge",
        ridge_eps: float = 1.0e-4,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.npod = npod
        self.latent_size = latent_size
        self.escape_radius = escape_radius
        self.dmd_method = dmd_method
        self.ridge_eps = ridge_eps
        self.dtype = dtype

        # NOTE: log-norm that pinned states are mapped to
        self.logn_clip = float(np.log(escape_radius + LOG_EPS) / LOG_SCALE)

        # POD basis of the (real, interleaved) states; see `fit_observables`
        self.register_buffer(
            "basis", torch.zeros(2 * input_size, npod, device=device, dtype=dtype)
        )

        # NOTE: the states are polynomials in the (complex) control `c`, so a
        # complex branch `c -> (w c + b)^2 -> ...` supplies the monomials (and the
        # phase doubling) that a real MLP has a hard time constructing from
        # `(Re c, Im c)`. `c` is stored stacked as `[Re c, Im c]`.
        self.c_branch: nn.Module | None = None
        if c_features > 0:
            layers: list[nn.Module] = []
            width = 1
            for _ in range(c_depth):
                layers.append(
                    ComplexLinear(
                        width, c_features, bias=True, device=device, dtype=dtype
                    )
                )
                layers.append(ComplexQuadratic())
                width = c_features

            self.c_branch = nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            nn.Linear(
                # NOTE: (u, log z, Re c, Im c, Re chi(c), Im chi(c))
                npod + 3 + 2 * c_features,
                hidden_size,
                device=device,
                dtype=dtype,
            ),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size, device=device, dtype=dtype),
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_size, hidden_size, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_size, npod + 1, device=device, dtype=dtype),
        )
        self.decoder_lin = nn.Linear(latent_size, npod + 1, device=device, dtype=dtype)

    @torch.no_grad()
    def fit_observables(self, x: torch.Tensor) -> None:
        """Compute the POD basis from the unpinned states in ``x``.

        The states are not centered, so that ``z_0 = 0`` stays at the origin.
        """
        xn = torch.linalg.vector_norm(x, dim=-1)
        _, _, vh = torch.linalg.svd(x[xn <= self.escape_radius], full_matrices=False)

        basis = cast(torch.Tensor, self.basis)
        basis.copy_(vh[: self.npod].T)

    def observe(self, x: torch.Tensor) -> Observables:
        """Polar observables of the interleaved real states ``x``."""
        masks = compute_trajectory_mask(x, escape_radius=self.escape_radius)
        active = masks.state
        basis = cast(torch.Tensor, self.basis)

        y = x @ basis
        ynorm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        valid = active[..., None] & (ynorm > 1.0e-8)
        direction = torch.where(valid, y / ynorm.clamp(min=1.0e-8), torch.zeros_like(y))

        logn = torch.log(torch.linalg.vector_norm(x, dim=-1) + LOG_EPS) / LOG_SCALE
        logn = torch.where(active, logn, torch.full_like(logn, self.logn_clip))

        return Observables(
            direction=direction,
            logn=logn,
            active=active,
            transition=masks.transition,
        )

    def encode(
        self, direction: torch.Tensor, logn: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        if self.c_branch is not None:
            c = torch.cat([c, self.c_branch(c)], dim=-1)

        c = c[:, None, :].expand(-1, direction.shape[1], -1)
        return self.encoder(torch.cat([direction, logn[..., None], c], dim=-1))

    def decode(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder_mlp(g) + self.decoder_lin(g)
        direction = out[..., : self.npod]
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)

        return direction, out[..., self.npod]

    def fit_dmd(self, X: torch.Tensor, Y: torch.Tensor) -> DenseDMDc[Any]:
        """Fit the affine latent dynamics ``Y = X A + b`` from transitions ``X -> Y``.

        This is a DMDc fit with a constant unit input ``u = 1``, so that ``B = b``.
        """
        U = torch.ones_like(X[:, :1])
        return build_dense_dmdc(X, U, Y, method=self.dmd_method, eps=self.ridge_eps)

    def rollout(
        self, g0: torch.Tensor, dmdc: DenseDMDc[Any], steps: int
    ) -> torch.Tensor:
        """Latent states ``g_1, ..., g_steps`` of shape ``(batch, steps, r)``."""
        u = torch.ones_like(g0[:, :1])

        g = g0
        out = []
        for _ in range(steps):
            g = dmdc.evolve(g, u)
            out.append(g)

        return torch.stack(out, dim=1)  # ty: ignore[invalid-argument-type]

    # ty: ignore[invalid-mutable-override]
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        dmdc: DenseDMDc[Any] | None = None,
        horizon: int = 0,
    ) -> DMDAutoencoderResult:
        obs = self.observe(x)
        enc = self.encode(obs.direction, obs.logn, c)
        rec_direction, rec_logn = self.decode(enc)
        lin_pred = roll_direction = roll_logn = None

        if dmdc is not None:
            u = torch.ones_like(enc[:, :-1, :1])
            lin_pred = cast(torch.Tensor, dmdc.evolve(enc[:, :-1], u))

            if not 0 <= horizon < x.shape[1]:
                raise ValueError(f"out of bounds horizon: {horizon}")

            if horizon > 0:
                roll = self.rollout(enc[:, 0], dmdc, horizon)
                roll_direction, roll_logn = self.decode(roll)

        return DMDAutoencoderResult(
            obs=obs,
            enc=enc,
            rec_direction=rec_direction,
            rec_logn=rec_logn,
            lin_pred=lin_pred,
            roll_direction=roll_direction,
            roll_logn=roll_logn,
        )


# }}}


# {{{ loss


class DMDLossResult(NamedTuple):
    total: torch.Tensor
    """Total value of the loss function. The gradient will be applied to this."""

    loss_rec: torch.Tensor
    """Reconstruction loss."""
    loss_lin: torch.Tensor
    """One-step latent linearity loss (NaN when inactive)."""
    loss_pre: torch.Tensor
    """Multi-step decoded prediction loss (NaN when inactive)."""

    def tensor(self) -> torch.Tensor:
        return torch.stack([self.total, self.loss_rec, self.loss_lin, self.loss_pre])

    def backward(self, *args: Any, **kwargs: Any) -> None:
        self.total.backward(*args, **kwargs)


class DMDLoss(nn.Module):
    r"""Loss on polar observables.

    The state error is the squared error of the unit direction plus the squared
    error of the log-norm, so that all states weigh the same regardless of
    their magnitude. Pinned states only enter through a censored log-norm term
    that is active when the prediction is below the clip.
    """

    def __init__(
        self,
        *,
        logn_clip: float,
        gamma_rec: float = 1.0,
        gamma_lin: float = 1.0,
        gamma_pre: float = 1.0,
    ) -> None:
        super().__init__()

        self.logn_clip = logn_clip
        self.gamma_rec = gamma_rec
        self.gamma_lin = gamma_lin
        self.gamma_pre = gamma_pre

    @property
    def dim(self) -> int:
        return 4

    def _state_error(
        self,
        direction: torch.Tensor,
        logn: torch.Tensor,
        obs: Observables,
        sl: slice,
    ) -> torch.Tensor:
        target = obs.direction[:, sl]
        active = obs.active[:, sl]
        has_direction = active & (torch.sum(target**2, dim=-1) > 0.5)

        error = torch.sum((direction - target) ** 2, dim=-1)
        error = torch.where(has_direction, error, torch.zeros_like(error))

        error_logn = torch.where(
            active,
            (logn - obs.logn[:, sl]) ** 2,
            torch.relu(self.logn_clip - logn) ** 2,
        )

        return error + error_logn

    # ty: ignore[invalid-mutable-override]
    def forward(self, result: DMDAutoencoderResult) -> DMDLossResult:
        obs = result.obs

        loss_rec = self._state_error(
            result.rec_direction, result.rec_logn, obs, slice(None)
        ).mean()
        total = self.gamma_rec * loss_rec

        # NOTE: terms that are not active are reported as NaN
        nan = torch.full_like(loss_rec, float("nan"))

        loss_lin = nan
        if result.lin_pred is not None:
            error = torch.sum((result.lin_pred - result.enc[:, 1:]) ** 2, dim=-1)
            error = torch.where(obs.transition, error, torch.zeros_like(error))
            loss_lin = error.sum() / obs.transition.sum().clamp(min=1)
            total = total + self.gamma_lin * loss_lin

        loss_pre = nan
        if result.roll_direction is not None:
            assert result.roll_logn is not None
            horizon = result.roll_direction.shape[1]
            loss_pre = self._state_error(
                result.roll_direction,
                result.roll_logn,
                obs,
                slice(1, horizon + 1),
            ).mean()
            total = total + self.gamma_pre * loss_pre

        return DMDLossResult(
            total=total, loss_rec=loss_rec, loss_lin=loss_lin, loss_pre=loss_pre
        )


# }}}


# {{{ torch_train


class TrainMetrics(NamedTuple):
    lr: torch.Tensor
    """Learning rate used for each epoch."""
    lin_validate: torch.Tensor
    """Validation one-step latent linearity loss."""
    pre_validate: torch.Tensor
    """Validation multi-step prediction loss."""


class ErrorMetrics(NamedTuple):
    iou: torch.Tensor
    """Intersection over union of the predicted and true converged sets."""
    dice: torch.Tensor
    """Dice score of the predicted and true converged sets."""
    a_radius: torch.Tensor
    """Spectral radius of the global DMD operator ``dmdc.A``."""


def _empty_train_metrics(epochs: int) -> TrainMetrics:
    return TrainMetrics(
        lr=torch.full((epochs,), float("nan")),
        lin_validate=torch.full((epochs,), float("nan")),
        pre_validate=torch.full((epochs,), float("nan")),
    )


def _empty_error_metrics(epochs: int) -> ErrorMetrics:
    return ErrorMetrics(
        iou=torch.full((epochs,), float("nan")),
        dice=torch.full((epochs,), float("nan")),
        a_radius=torch.full((epochs,), float("nan")),
    )


def build_latent_dmd(
    model: DMDAutoencoder,
    X: torch.Tensor,
    C: torch.Tensor,
    *,
    grad: bool = False,
) -> DenseDMDc[Any]:
    r"""Build the DMD operator in the latent space from the full dataset.

    ``X`` has shape ``(ntraj, maxit, 2 d)`` and ``C`` has shape ``(ntraj, 2)``.
    Only transitions between unpinned states are used. With *grad*, the fit is
    differentiable with respect to the encoder.
    """
    with torch.set_grad_enabled(grad):
        obs = model.observe(X)
        enc = model.encode(obs.direction, obs.logn, C)

        Xl = enc[:, :-1][obs.transition]
        Yl = enc[:, 1:][obs.transition]

        return model.fit_dmd(Xl, Yl)


class TrainCallback(Protocol):
    def __call__(
        self, epoch: int, stage: Stage, model: DMDAutoencoder, *, last: bool
    ) -> None: ...


def _evaluate(
    model: DMDAutoencoder,
    criterion: DMDLoss,
    loader: DataLoader,
    dmdc: DenseDMDc[Any] | None,
    horizon: int,
) -> torch.Tensor:
    """Average loss over all the batches in *loader*."""
    model.eval()

    with torch.no_grad():
        total = torch.zeros(criterion.dim, device=next(model.parameters()).device)
        for batch_x, batch_c in loader:
            total += criterion(model(batch_x, batch_c, dmdc, horizon)).tensor()

    return total / len(loader)


def torch_train(
    model: DMDAutoencoder,
    criterion: DMDLoss,
    X_train: torch.Tensor,
    X_validate: torch.Tensor,
    *,
    c_train: torch.Tensor,
    c_validate: torch.Tensor,
    stages: Sequence[Stage],
    lr: float = 2.0e-3,
    max_norm: float = 5.0,
    epochs: int = 60,
    batch_size: int = 512,
    callback: TrainCallback | None = None,
) -> tuple[DMDAutoencoder, torch.Tensor, TrainMetrics]:
    """Train the model in *stages* of *epochs* epochs each.

    The model is not compiled and the DMD operator is fit once per epoch
    on the full training set (when DMD is active). Validation also uses the
    full training set.
    """
    device = X_train.device

    loader = DataLoader(
        TensorDataset(X_train, c_train),
        batch_size=batch_size,
        shuffle=True,
        # NOTE: this is needed otherwise the DataLoader seems to want to use the
        # 'cpu' device even when there is another default device.. it is seeded
        # explicitly so that shuffling is reproducible
        generator=torch.Generator(device).manual_seed(torch.initial_seed()),
    )
    validate_loader = DataLoader(
        TensorDataset(X_validate, c_validate), batch_size=batch_size
    )

    timer = TicTocTimer()
    total_epochs = epochs * len(stages)
    losses = torch.empty((total_epochs, 2, criterion.dim), device=device)
    train_metrics = _empty_train_metrics(total_epochs)

    epoch = 0
    for stage in stages:
        log.info(
            "Stage '%s': horizon %d, DMD %s",
            stage.name,
            stage.horizon,
            "on" if stage.use_dmd else "off",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        for stage_epoch in range(epochs):
            # {{{ training

            timer.tic()

            with torch.no_grad():
                dmdc = (
                    build_latent_dmd(model, X_train, c_train) if stage.use_dmd else None
                )

            model.train()
            train_metrics.lr[epoch] = optimizer.param_groups[0]["lr"]

            loss_epoch = torch.zeros(criterion.dim, device=device)
            num_batches = 0

            for batch_x, batch_c in loader:
                optimizer.zero_grad()
                result = model(batch_x, batch_c, dmdc=dmdc, horizon=stage.horizon)
                loss = criterion(result)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

                loss_epoch += loss.tensor().detach()
                num_batches += 1

            loss_train = loss_epoch / num_batches
            scheduler.step()

            # }}}

            # {{{ validation

            with torch.no_grad():
                dmdc = (
                    build_latent_dmd(model, X_train, c_train) if stage.use_dmd else None
                )

            loss_validate = _evaluate(
                model, criterion, validate_loader, dmdc, stage.horizon
            )

            train_metrics.lin_validate[epoch] = loss_validate[2]
            train_metrics.pre_validate[epoch] = loss_validate[3]

            # }}}

            timer.toc()

            losses[epoch, 0] = loss_train
            losses[epoch, 1] = loss_validate

            radius = _spectral_radius(dmdc.A) if dmdc is not None else None  # ty: ignore[invalid-argument-type]
            log.info(
                "[%s %03d/%03d] total %.5e rec %.3e lin %.3e pre %.3e | "
                "val rec %.3e lin %.3e pre %.3e | lam %s (%s)",
                stage.name,
                stage_epoch + 1,
                epochs,
                loss_train[0],
                loss_train[1],
                loss_train[2],
                loss_train[3],
                loss_validate[1],
                loss_validate[2],
                loss_validate[3],
                "-" if radius is None else f"{radius.item():.3f}",
                timer.short(),
            )

            if callback is not None:
                with torch.no_grad():
                    callback(epoch, stage, model, last=stage_epoch == epochs - 1)

            epoch += 1

    return model, losses, train_metrics


# }}}


# {{{ torch_predict


def _normalize_control(
    c: torch.Tensor, bbox: tuple[float, float, float, float]
) -> torch.Tensor:
    """Stack a complex ``c`` into real pairs scaled by the size of the bbox."""
    return torch.stack([c.real, c.imag], dim=-1) / max(abs(b) for b in bbox)


def _grid_control(
    model: DMDAutoencoder,
    bbox: tuple[float, float, float, float],
    resolution: int,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Normalized controls of shape ``(resolution ** 2, 2)`` on the grid."""
    from commonlib import generate_grid

    assert model.dtype.is_floating_point

    c = cast(
        "torch.Tensor",
        generate_grid(
            *bbox,
            nx=resolution,
            ny=resolution,
            dtype=model.dtype.to_complex(),
            xp=array_api_compat.array_namespace(like),
        ),
    )
    return _normalize_control(c.reshape(-1), bbox)


def _encode_origin(model: DMDAutoencoder, c: torch.Tensor) -> torch.Tensor:
    """Encode the initial state ``z_0 = 0`` for each of the controls *c*."""
    n = c.shape[0]
    direction = torch.zeros((n, 1, model.npod), device=c.device, dtype=model.dtype)
    logn = torch.full(
        (n, 1),
        float(np.log(LOG_EPS) / LOG_SCALE),
        device=c.device,
        dtype=model.dtype,
    )

    return model.encode(direction, logn, c)[:, 0]


def torch_predict(
    model: DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
) -> torch.Tensor:
    """Predict the norm of ``z_maxit`` on the grid by encoding ``z_0 = 0`` once.

    The latent state is evolved linearly for *maxit* steps and only the final
    state is decoded.

    :returns: a real tensor of shape ``(resolution, resolution)``.
    """
    with torch.no_grad():
        c = _grid_control(model, bbox, resolution, like=dmdc.A)  # ty: ignore[invalid-argument-type]
        g0 = _encode_origin(model, c)
        _, logn_final = model.decode(model.rollout(g0, dmdc, maxit)[:, -1])

        norm = torch.exp(logn_final * LOG_SCALE) - LOG_EPS

    return norm.reshape(resolution, resolution)


# }}}


# {{{ features


def compute_features(
    model: DMDAutoencoder,
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

    c = _grid_control(model, bbox, resolution, like=dmdc.A)  # ty: ignore[invalid-argument-type]
    c = c.detach().clone().requires_grad_()
    g_final = model.rollout(_encode_origin(model, c), dmdc, maxit)[:, -1]
    _, logn_final = model.decode(g_final)
    (grad,) = torch.autograd.grad(logn_final.sum(), c)

    # NOTE: `c` is normalized by `scale` and the log-norm by `LOG_SCALE`
    jac = torch.linalg.vector_norm(grad, dim=-1) * (LOG_SCALE / scale)
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
        g_last = model.rollout(_encode_origin(model, c.detach()), dmdc, maxit)
        direction, logn = model.decode(g_last[:, -1])
        norm = (torch.exp(logn * LOG_SCALE) - LOG_EPS).clamp(min=0.0)
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


def _spectral_radius(A: torch.Tensor) -> torch.Tensor:
    A = A.detach().cpu()
    if not torch.isfinite(A).all():
        return torch.tensor(float("nan"))

    return torch.abs(torch.linalg.eigvals(A)).max()


def _plot_predict(
    filename: pathlib.Path,
    model: DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    config: DMDConfig,
    *,
    zmask: torch.Tensor,
    tag: str,
    overwrite: bool,
    plot: bool = True,
) -> ErrorMetrics:
    norm = torch_predict(
        model,
        dmdc,
        bbox=config.bbox,
        resolution=config.resolution,
        maxit=config.maxit,
    )

    from nneuroutil.scores import dice_score, jaccard_index

    in_set = norm < config.escape_radius
    metrics = ErrorMetrics(
        iou=cast(torch.Tensor, jaccard_index(in_set, zmask)).detach().cpu(),
        dice=cast(torch.Tensor, dice_score(in_set, zmask)).detach().cpu(),
        a_radius=_spectral_radius(dmdc.A),  # ty: ignore[invalid-argument-type]
    )

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
        ax.imshow(in_set.cpu().numpy(), extent=config.bbox, cmap="binary")

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
    metrics: ErrorMetrics,
    *,
    final: ErrorMetrics | None = None,
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
    metrics: TrainMetrics,
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
        config = DMDConfig.from_sources(checkpoint["config"], overrides)
        mat = checkpoint["mat"]
    else:
        config = DMDConfig.from_sources(overrides)

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
        model_config = checkpoint["model_config"].copy()
        if "npca" in model_config and "npod" not in model_config:
            model_config["npod"] = model_config.pop("npca")
    else:
        model_config = {
            "input_size": d,
            "npod": config.npod,
            "latent_size": config.latent_size,
            "hidden_size": config.hidden_size,
            "c_features": config.c_features,
            "c_depth": config.c_depth,
            "escape_radius": config.escape_radius,
            "dmd_method": config.dmd_method,
            "ridge_eps": config.ridge_eps,
        }

    model = DMDAutoencoder(**model_config, device=device, dtype=mat.dtype.to_real())
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
    c = _normalize_control(c[mask], config.bbox)

    (x_train, c_train), (x_validate, c_validate), (x_test, c_test) = cast(
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        random_split((x, c), config.split, rng=default_rng(torch, config.seed)),
    )

    if checkpoint is None:
        model.fit_observables(x_train)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    stages = make_stages(config)
    num_epochs = config.epochs * len(stages)
    error_metrics = _empty_error_metrics(num_epochs)

    def plot_predict(
        epoch: int, stage: Stage, model: DMDAutoencoder, *, last: bool
    ) -> None:
        if not stage.use_dmd:
            return

        dmdc = build_latent_dmd(model, x_train, c_train)
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

    criterion = DMDLoss(
        logn_clip=model.logn_clip,
        gamma_rec=config.gamma_rec,
        gamma_lin=config.gamma_lin,
        gamma_pre=config.gamma_pre,
    )

    _, losses, train_metrics = torch_train(
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
    dmdc = build_latent_dmd(model, x_train, c_train)

    mem.add_record("After training")

    loss_test = _evaluate(
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
    config = DMDConfig.from_sources(checkpoint["config"], overrides)

    losses = checkpoint["losses"]
    zmask = checkpoint["zmask"]
    dmdc = _load_latent_dmd(checkpoint, checkpoint["model_config"]["latent_size"])
    train_metrics = checkpoint.get("train_metrics")
    error_metrics = checkpoint.get("error_metrics")

    if outfile is None:
        outfile = weightsfile.with_suffix("")
    outfile = outfile.with_suffix("")

    # }}}

    # {{{ load model

    model_config = checkpoint["model_config"].copy()
    if "npca" in model_config and "npod" not in model_config:
        model_config["npod"] = model_config.pop("npca")

    model = DMDAutoencoder(**model_config, device=device, dtype=torch.float32)
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
        _plot_train_metrics(outfile, TrainMetrics(**train_metrics), overwrite=overwrite)

    if error_metrics is not None:
        _plot_error_metrics(
            outfile,
            ErrorMetrics(**error_metrics),
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
        else:
            raise SystemExit(1)

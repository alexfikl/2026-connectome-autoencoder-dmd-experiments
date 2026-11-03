# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, cast

import array_api_compat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from commonlib import Config, compute_trajectory_mask
from nneuroutil.dmdc import DenseDMDc, build_dense_dmdc
from nneuroutil.helpers import TicTocTimer
from nneuroutil.interop.torch import ComplexLinear, ComplexQuadratic

log = logging.getLogger("lib.dmdaelib")

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


def load_latent_dmd(checkpoint: Mapping[str, Any]) -> DenseDMDc[Any]:
    """Load the latent DMDc model from a checkpoint."""
    return DenseDMDc(A=checkpoint["dmdc"]["A"], B=checkpoint["dmdc"]["B"])


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


# {{{ dynamics helpers


def build_latent_dmd(
    model: DMDAutoencoder,
    X: torch.Tensor,
    C: torch.Tensor,
    *,
    grad: bool = False,
) -> DenseDMDc[Any]:
    r"""Build the DMD operator in the latent space from the transitions in ``X``.

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


def normalize_control(
    c: torch.Tensor, bbox: tuple[float, float, float, float]
) -> torch.Tensor:
    """Stack a complex ``c`` into real pairs scaled by the size of the bbox."""
    return torch.stack([c.real, c.imag], dim=-1) / max(abs(b) for b in bbox)


def grid_control(
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
    return normalize_control(c.reshape(-1), bbox)


def encode_origin(model: DMDAutoencoder, c: torch.Tensor) -> torch.Tensor:
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
    dtype = model.dtype
    assert dtype.is_floating_point

    with torch.no_grad():
        c = grid_control(model, bbox, resolution, like=dmdc.A)  # ty: ignore[invalid-argument-type]
        g0 = encode_origin(model, c)
        _, logn_final = model.decode(model.rollout(g0, dmdc, maxit)[:, -1])

        norm = torch.exp(logn_final * LOG_SCALE) - LOG_EPS

    return norm.reshape(resolution, resolution)


def spectral_radius(A: torch.Tensor) -> torch.Tensor:
    """Spectral radius of the matrix ``A``."""
    A = A.detach().cpu()
    if not torch.isfinite(A).all():
        return torch.tensor(float("nan"))

    return torch.abs(torch.linalg.eigvals(A)).max()


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


def empty_train_metrics(epochs: int) -> TrainMetrics:
    return TrainMetrics(
        lr=torch.full((epochs,), float("nan")),
        lin_validate=torch.full((epochs,), float("nan")),
        pre_validate=torch.full((epochs,), float("nan")),
    )


def empty_error_metrics(epochs: int) -> ErrorMetrics:
    return ErrorMetrics(
        iou=torch.full((epochs,), float("nan")),
        dice=torch.full((epochs,), float("nan")),
        a_radius=torch.full((epochs,), float("nan")),
    )


class TrainCallback(Protocol):
    def __call__(
        self, epoch: int, stage: Stage, model: DMDAutoencoder, *, last: bool
    ) -> None: ...


def evaluate(
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
            total += criterion(
                model(batch_x, batch_c, dmdc=dmdc, horizon=horizon)
            ).tensor()

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

    The model is not compiled and the DMD operator is fit (and differentiated
    through) at every optimizer step on the current batch data. Validation uses
    the full training set.
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
    train_metrics = empty_train_metrics(total_epochs)

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

            model.train()
            train_metrics.lr[epoch] = optimizer.param_groups[0]["lr"]

            loss_epoch = torch.zeros(criterion.dim, device=device)
            num_batches = 0

            timer.tic()
            for batch_x, batch_c in loader:
                dmdc = (
                    build_latent_dmd(model, batch_x, batch_c, grad=True)
                    if stage.use_dmd
                    else None
                )

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

            loss_validate = evaluate(
                model, criterion, validate_loader, dmdc, stage.horizon
            )

            train_metrics.lin_validate[epoch] = loss_validate[2]
            train_metrics.pre_validate[epoch] = loss_validate[3]

            # }}}

            timer.toc()

            losses[epoch, 0] = loss_train
            losses[epoch, 1] = loss_validate

            radius = spectral_radius(dmdc.A) if dmdc is not None else None  # ty: ignore[invalid-argument-type]
            log.info(
                "[%s %03d/%03d] total %.3e rec %.3e lin %.3e pre %.3e | "
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

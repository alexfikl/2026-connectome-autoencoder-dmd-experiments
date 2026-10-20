# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, cast

import array_api_compat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from commonlib import Config, compute_trajectory_mask, make_argument_parser
from nneuroutil.helpers import (
    TicTocTimer,
    machine_info,
    module_logger,
)
from nneuroutil.torch_extras import (
    ComplexLinear,
    ComplexQuadratic,
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)
from nneuroutil.visualization import figure

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains or runs inference with a DMD Autoencoder: encode -> evolve linearly in
latent space -> decode.

The state is represented by polar observables: a unit direction in a PCA basis
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
    # number of trajectories sampled per step to fit the operator in training
    # (None -> all the training trajectories)
    dmd_subset: int | None = None

    # weights of the reconstruction, latent linearity, and prediction losses
    gamma_rec: float = 1.0
    gamma_lin: float = 1.0
    gamma_pre: float = 1.0

    # number of PCA directions, latent size, and hidden width of the networks
    npca: int = 24
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
    """Unit direction in the PCA basis of shape ``(batch, steps, npca)``.

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
    lin_pred: torch.Tensor | None = None
    """One-step latent prediction ``[g_n, 1] A`` of shape ``(batch, steps - 1, r)``."""
    roll_direction: torch.Tensor | None = None
    """Decoded direction of the latent rollout from the initial state."""
    roll_logn: torch.Tensor | None = None
    """Decoded log-norm of the latent rollout from the initial state."""


def _fit_linear_operator(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    method: Literal["pinv", "ridge"],
    eps: float,
) -> torch.Tensor:
    """Fit ``Y ~ X A`` with no control term.

    The ridge parameter is relative to the average diagonal of ``X^H X``.
    """
    r = X.shape[1]
    XhX = X.mH @ X
    XhY = X.mH @ Y
    XhX = 0.5 * (XhX + XhX.mH)

    if method == "pinv":
        A = torch.linalg.pinv(XhX) @ XhY
    else:
        reg = eps * torch.trace(XhX) / r
        A = torch.linalg.solve(
            XhX + reg * torch.eye(r, device=X.device, dtype=X.dtype), XhY
        )

    return cast(torch.Tensor, A)


class DMDAutoencoder(nn.Module):
    r"""Autoencoder with polar observables and a linear latent operator.

    The encoder maps ``(direction, log-norm, c)`` of a state to a latent vector
    ``g`` and the latent evolves as ``[g, 1] -> [g, 1] A_dmd``, where the last
    column of ``A_dmd`` is fixed to ``(0, ..., 0, 1)`` so that the constant
    coordinate is invariant. The decoder returns a direction and a log-norm.
    """

    def __init__(
        self,
        input_size: int,
        *,
        npca: int,
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
        self.npca = npca
        self.latent_size = latent_size
        self.escape_radius = escape_radius
        self.dmd_method = dmd_method
        self.ridge_eps = ridge_eps
        self.dtype = dtype

        # NOTE: log-norm that pinned states are mapped to
        self.logn_clip = float(np.log(escape_radius + LOG_EPS) / LOG_SCALE)

        # PCA basis of the (real, interleaved) states; see `fit_observables`
        self.register_buffer(
            "basis", torch.zeros(2 * input_size, npca, device=device, dtype=dtype)
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
                npca + 3 + 2 * c_features, hidden_size, device=device, dtype=dtype
            ),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size, device=device, dtype=dtype),
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_size, hidden_size, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_size, npca + 1, device=device, dtype=dtype),
        )
        self.decoder_lin = nn.Linear(latent_size, npca + 1, device=device, dtype=dtype)

    @torch.no_grad()
    def fit_observables(self, x: torch.Tensor) -> None:
        """Compute the PCA basis from the unpinned states in ``x``.

        The states are not centered, so that ``z_0 = 0`` stays at the origin.
        """
        xn = torch.linalg.vector_norm(x, dim=-1)
        _, _, vh = torch.linalg.svd(x[xn <= self.escape_radius], full_matrices=False)

        basis = cast(torch.Tensor, self.basis)
        basis.copy_(vh[: self.npca].T)

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
        direction = out[..., : self.npca]
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)

        return direction, out[..., self.npca]

    def fit_dmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Fit the operator on ``[g, 1]`` from latent transitions ``X -> Y``."""
        X1 = torch.cat([X, torch.ones_like(X[:, :1])], dim=-1)
        A = _fit_linear_operator(X1, Y, method=self.dmd_method, eps=self.ridge_eps)

        # NOTE: the constant coordinate is invariant
        last = torch.zeros(self.latent_size + 1, 1, device=A.device, dtype=A.dtype)
        last[-1, 0] = 1.0
        return torch.cat([A, last], dim=1)

    def rollout(
        self, g0: torch.Tensor, A_dmd: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Latent states ``g_1, ..., g_steps`` of shape ``(batch, steps, r)``."""
        g = torch.cat([g0, torch.ones_like(g0[:, :1])], dim=-1)

        out = []
        for _ in range(steps):
            g = g @ A_dmd
            out.append(g[:, : self.latent_size])

        return torch.stack(out, dim=1)

    # ty: ignore[invalid-mutable-override]
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        A_dmd: torch.Tensor | None = None,
        horizon: int = 0,
    ) -> DMDAutoencoderResult:
        obs = self.observe(x)
        enc = self.encode(obs.direction, obs.logn, c)
        rec_direction, rec_logn = self.decode(enc)

        result = DMDAutoencoderResult(
            obs=obs, enc=enc, rec_direction=rec_direction, rec_logn=rec_logn
        )
        if A_dmd is None:
            return result

        g = torch.cat([enc[:, :-1], torch.ones_like(enc[:, :-1, :1])], dim=-1)
        result = result._replace(lin_pred=(g @ A_dmd)[..., : self.latent_size])

        horizon = min(horizon, x.shape[1] - 1)
        if horizon > 0:
            roll = self.rollout(enc[:, 0], A_dmd, horizon)
            roll_direction, roll_logn = self.decode(roll)
            result = result._replace(roll_direction=roll_direction, roll_logn=roll_logn)

        return result


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
    """Spectral radius of the global DMD operator ``A_dmd``."""


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
    subset: int | None = None,
) -> torch.Tensor:
    r"""Build the DMD operator in the latent space from the full dataset.

    ``X`` has shape ``(ntraj, maxit, 2 d)`` and ``C`` has shape ``(ntraj, 2)``.
    Only transitions between unpinned states are used. With *grad*, the fit is
    differentiable with respect to the encoder. With *subset*, the fit uses that
    many randomly sampled trajectories.
    """
    if subset is not None and subset < X.shape[0]:
        idx = torch.randperm(X.shape[0], device=X.device)[:subset]
        X, C = X[idx], C[idx]

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
    A_dmd: torch.Tensor | None,
    horizon: int,
) -> torch.Tensor:
    """Average loss over all the batches in *loader*."""
    model.eval()

    with torch.no_grad():
        total = torch.zeros(criterion.dim, device=next(model.parameters()).device)
        for batch_x, batch_c in loader:
            total += criterion(model(batch_x, batch_c, A_dmd, horizon)).tensor()

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
    dmd_subset: int | None = None,
    callback: TrainCallback | None = None,
) -> tuple[DMDAutoencoder, torch.Tensor, TrainMetrics]:
    """Train the model in *stages* of *epochs* epochs each.

    The model is not compiled and the DMD operator is fit (and differentiated
    through) at every optimizer step on *dmd_subset* random training
    trajectories (all of them by default). Validation always uses the full
    training set.
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

            model.train()
            train_metrics.lr[epoch] = optimizer.param_groups[0]["lr"]

            loss_epoch = torch.zeros(criterion.dim, device=device)
            num_batches = 0

            timer.tic()
            for batch_x, batch_c in loader:
                A_dmd = (
                    build_latent_dmd(
                        model, X_train, c_train, grad=True, subset=dmd_subset
                    )
                    if stage.use_dmd
                    else None
                )

                optimizer.zero_grad()
                result = model(batch_x, batch_c, A_dmd, stage.horizon)
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
                A_dmd = (
                    build_latent_dmd(model, X_train, c_train) if stage.use_dmd else None
                )

            loss_validate = _evaluate(
                model, criterion, validate_loader, A_dmd, stage.horizon
            )

            train_metrics.lin_validate[epoch] = loss_validate[2]
            train_metrics.pre_validate[epoch] = loss_validate[3]

            # }}}

            timer.toc()

            losses[epoch, 0] = loss_train
            losses[epoch, 1] = loss_validate

            radius = _spectral_radius(A_dmd[:-1, :-1]) if A_dmd is not None else None
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
    direction = torch.zeros((n, 1, model.npca), device=c.device, dtype=model.dtype)
    logn = torch.full(
        (n, 1),
        float(np.log(LOG_EPS) / LOG_SCALE),
        device=c.device,
        dtype=model.dtype,
    )

    return model.encode(direction, logn, c)[:, 0]


def torch_predict(
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
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
        c = _grid_control(model, bbox, resolution, like=A_dmd)
        g0 = _encode_origin(model, c)
        _, logn_final = model.decode(model.rollout(g0, A_dmd, maxit)[:, -1])

        norm = torch.exp(logn_final * LOG_SCALE) - LOG_EPS

    return norm.reshape(resolution, resolution)


# }}}


# {{{ features


def compute_features(
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int,
    maxit: int,
) -> dict[str, np.ndarray]:
    r"""Compute features of the trained model for classification.

    The latent basis is arbitrary (every matrix has its own encoder), so the
    features are either invariant to it or are lifted to the state space, where
    the ROI coordinates are shared between matrices.

    With the row convention ``g -> g A_dmd`` and ``A_dmd = V diag(lambda) V^-1``,
    the latent state is ``g_k = sum_j b_j lambda_j^k W_j`` with ``b = g V`` and
    ``W = V^-1``. We then have

    * ``eigvals``: the eigenvalues, sorted by decreasing modulus.
    * ``modes``: the modes ``W_j`` pushed through the Jacobian of the decoder (at
      the mean encoded initial state) and the PCA basis, as unit vectors of shape
      ``(nmodes, d, 2)``, where the last axis holds the coefficients of
      ``(Re z_i, Im z_i)``. The phase is fixed so that the largest entry is real.
    * ``mode_profiles``: ``(nmodes, d)`` phase-free ROI profile of each mode.
    * ``mode_gain``: the norm of the pushed-through modes before normalization.
    * ``strength_mean``, ``strength_std``: statistics over the grid of the
      strengths ``|b_j(c)| * gain_j`` of each mode in the encode-once rollout.
    * ``logn_strength_mean``: the same, but for the effect on the decoded
      log-norm instead of the direction.
    * ``jac_map``, ``jac_stats``: the gradient norm
      :math:`|\nabla_c \log |z_N||` of the decoded final log-norm with respect to
      the (physical) control, on the grid, and its (mean, std, median, p90, max).
    """
    scale = max(abs(b) for b in bbox)
    r = model.latent_size

    # {{{ spectrum

    lam, V = torch.linalg.eig(A_dmd.detach().cpu().to(torch.float64))
    order = torch.argsort(lam.abs(), descending=True)
    lam, V = lam[order], V[:, order]
    W = torch.linalg.inv(V)

    # }}}

    # {{{ strengths

    c = _grid_control(model, bbox, resolution, like=A_dmd)
    with torch.no_grad():
        g0 = _encode_origin(model, c)
        g1 = torch.cat([g0, torch.ones_like(g0[:, :1])], dim=-1)
        b = g1.cpu().to(torch.complex128) @ V

    # }}}

    # {{{ modes in the state space

    def decoded(g: torch.Tensor) -> torch.Tensor:
        direction, logn = model.decode(g)
        return torch.cat([direction, logn[..., None]], dim=-1)

    jac_dec = torch.autograd.functional.jacobian(decoded, g0.mean(dim=0))
    jac_dec = jac_dec.detach().cpu().to(torch.complex128)

    # NOTE: shape (npca + 1, nmodes)
    lifted = jac_dec @ W[:, :r].T
    basis = cast(torch.Tensor, model.basis).detach().cpu().to(torch.complex128)
    modes = basis @ lifted[: model.npca]

    gain = torch.linalg.vector_norm(modes, dim=0)
    modes = modes / gain.clamp(min=1.0e-12)
    pivot = modes.gather(0, modes.abs().argmax(dim=0, keepdim=True))[0]
    modes = modes * (pivot.abs() / pivot)[None, :]
    modes = modes.T.reshape(modes.shape[1], -1, 2)

    profiles = torch.linalg.vector_norm(modes, dim=-1)

    amplitude = b.abs()
    strength = amplitude * gain[None, :]
    logn_strength = amplitude * lifted[model.npca].abs()[None, :]

    # }}}

    # {{{ sensitivity to the control

    c = c.detach().clone().requires_grad_()
    g_final = model.rollout(_encode_origin(model, c), A_dmd, maxit)[:, -1]
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

    return {
        "eigvals": lam.numpy(),
        "modes": modes.numpy(),
        "mode_profiles": profiles.numpy(),
        "mode_gain": gain.numpy(),
        "strength_mean": strength.mean(dim=0).numpy(),
        "strength_std": strength.std(dim=0).numpy(),
        "logn_strength_mean": logn_strength.mean(dim=0).numpy(),
        "jac_map": jac.reshape(resolution, resolution).numpy(),
        "jac_stats": jac_stats.numpy(),
    }


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
    A_dmd: torch.Tensor,
    config: DMDConfig,
    *,
    zmask: torch.Tensor,
    tag: str,
    overwrite: bool,
    plot: bool = True,
) -> ErrorMetrics:
    norm = torch_predict(
        model,
        A_dmd,
        bbox=config.bbox,
        resolution=config.resolution,
        maxit=config.maxit,
    )

    from nneuroutil.scores import dice_score, jaccard_index

    in_set = norm < config.escape_radius
    metrics = ErrorMetrics(
        iou=cast(torch.Tensor, jaccard_index(in_set, zmask)).detach().cpu(),
        dice=cast(torch.Tensor, dice_score(in_set, zmask)).detach().cpu(),
        a_radius=_spectral_radius(A_dmd[:-1, :-1]),
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
        model_config = checkpoint["model_config"]
    else:
        model_config = {
            "input_size": d,
            "npca": config.npca,
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
        random_split((x, c), config.split, rng=np.random.default_rng(config.seed)),
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

        A_dmd = build_latent_dmd(model, x_train, c_train)
        metrics = _plot_predict(
            outfile,
            model,
            A_dmd,
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
        dmd_subset=config.dmd_subset,
        callback=plot_predict,
    )
    A_dmd = build_latent_dmd(model, x_train, c_train)

    mem.add_record("After training")

    loss_test = _evaluate(
        model,
        criterion,
        DataLoader(TensorDataset(x_test, c_test), batch_size=config.batch_size),
        A_dmd,
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
        A_dmd,
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
            "a_dmd": A_dmd,
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
    A_dmd = checkpoint["a_dmd"]
    train_metrics = checkpoint.get("train_metrics")
    error_metrics = checkpoint.get("error_metrics")

    if outfile is None:
        outfile = weightsfile.with_suffix("")
    outfile = outfile.with_suffix("")

    # }}}

    # {{{ load model

    model = DMDAutoencoder(
        **checkpoint["model_config"], device=device, dtype=torch.float32
    )
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
            A_dmd,
            bbox=config.bbox,
            resolution=config.resolution,
            maxit=config.maxit,
        )
        feats["a_dmd"] = A_dmd.detach().cpu().numpy()
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
        A_dmd,
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


if __name__ == "__main__":
    parser = make_argument_parser(
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
        "--dmd-subset",
        type=int,
        default=None,
        help=(
            "Number of training trajectories used to fit the DMD operator at each "
            "training step (default: all)"
        ),
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
        "--npca",
        type=int,
        default=None,
        help="Number of PCA directions used to represent the state (default: 24)",
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
        else:
            raise SystemExit(1)

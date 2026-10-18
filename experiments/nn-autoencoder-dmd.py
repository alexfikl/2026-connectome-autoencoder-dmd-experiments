# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import logging
import pathlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import array_api_compat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from commonlib import Config, compute_trajectory_mask, make_argument_parser
from nneuroutil.dmdc import build_dense_dmdc
from nneuroutil.helpers import (
    TicTocTimer,
    machine_info,
    module_logger,
)
from nneuroutil.torch_extras import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
    view_as_complex,
    view_as_real,
)
from nneuroutil.visualization import figure

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains or runs inference with a DMD Autoencoder neural network with quadratic
activation functions.

Examples:

    > {SCRIPT_PATH.name} --device cuda train dataset.npz
    > {SCRIPT_PATH.name} --device cuda infer dataset-ae-dmd-quad.pkl
"""


# {{{ configuration


@dataclass(frozen=True, slots=True)
class DMDConfig(Config):
    """Configuration for the DMD autoencoder network."""

    epochs: int = 100
    lr: float = 5.0e-3
    maxit: int = 32
    batch_size: int = 512
    escape_radius: float = 10.0

    window_size: int = 32
    nwindows: int = 1
    dmd_warmup_epochs: int = 10
    fixed_point_rtol: float = 1.0e-3
    ridge_eps: float = 1.0e-3

    # NOTE: None values get defaults from the data
    latent_size: int | None = None
    alpha: float = 0.75
    norm_factor: float | None = None

    dmd_method: Literal["pinv", "ridge"] = "pinv"
    hidden_size: int | None = None
    rollout_steps: int = 1

    # weights of the reconstruction, latent linearity, and prediction losses
    gamma_rec: float = 1.0
    gamma_lin: float = 1.0
    gamma_pre: float = 1.0


# }}}


# {{{ model


class LearnableBlendedEncoder(nn.Module):
    r"""Encodes state :math:`z \in \mathbb{C}^d` using a learnable blended activation:

    .. math::

        y_i = W_1 (\alpha_i (W_2 z)_i + (1 - \alpha_i) (W_2 z)_i^2 + b_i)

    where :math:`\alpha_i = \sigma(\theta_i) \in [0, 1]` is a per-channel learnable
    parameter initialized from *alpha*.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        alpha: float = 0.5,
        hidden_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        from nneuroutil.torch_extras import (
            Bias,
            ComplexBlendedQuadratic,
            ComplexLinear,
            kaiming_uniform_,
        )

        super().__init__()
        self.input_size = in_features
        self.hidden_size = hidden_size if hidden_size is not None else out_features
        self.latent_size = out_features

        # layer 1
        linear_in = ComplexLinear(
            self.input_size,
            self.hidden_size,
            bias=False,
            interleaved=True,
            device=device,
            dtype=dtype,
        )
        kaiming_uniform_(
            linear_in.weight, param=alpha, nonlinearity="cblended_quadratic"
        )

        # layer 2
        linear_out = ComplexLinear(
            self.hidden_size,
            self.latent_size,
            bias=False,
            interleaved=True,
            device=device,
            dtype=dtype,
        )
        kaiming_uniform_(linear_out.weight, nonlinearity="linear")

        self.model = nn.Sequential(
            linear_in,
            ComplexBlendedQuadratic(
                alpha=alpha,
                num_parameters=self.hidden_size,
                learnable=True,
                interleaved=True,
                device=device,
                dtype=dtype,
            ),
            linear_out,
            Bias(2 * self.latent_size, device=device, dtype=dtype),
        )

    # ty: ignore[invalid-mutable-override]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DMDAutoencoderResult(NamedTuple):
    A_dmd: torch.Tensor
    """Same-space DMD linear operator on the latent state."""
    B_dmd: torch.Tensor
    """DMDc control operator mapping the constant control ``c`` into the latent."""

    x_out: torch.Tensor
    """Output of the autoencoder (decoded state)."""
    y_latent: torch.Tensor | None
    """Encoded state trajectory (same as encoder input, for loss_lin)."""
    yhat_latent: torch.Tensor | None
    """DMDc one-step prediction of the latent trajectory (steps 1..n)."""
    yhat_out: torch.Tensor | None
    """Decoded DMDc one-step prediction (back in physical space)."""


class DMDAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        latent_size: int | None = None,
        hidden_size: int | None = None,
        dmd_method: Literal["pinv", "ridge"] = "pinv",
        rollout_steps: int = 1,
        dmd_enabled: bool = False,
        escape_radius: float | None = None,
        fixed_point_rtol: float | None = None,
        ridge_eps: float = 1.0e-3,
        alpha: float | None = None,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        if dtype is None:
            dtype = torch.float32

        super().__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.latent_size = 2 * input_size if latent_size is None else latent_size

        self.hidden_size = hidden_size
        self.dmd_method = dmd_method
        self.rollout_steps = rollout_steps
        self.dtype = dtype

        self.dmd_enabled = dmd_enabled
        self.escape_radius = escape_radius
        self.fixed_point_rtol = fixed_point_rtol
        self.ridge_eps = ridge_eps

        # {{{ construct autoencoder

        from nneuroutil.torch_extras import ComplexLinear, kaiming_uniform_

        self.encoder = LearnableBlendedEncoder(
            self.input_size,
            self.latent_size,
            alpha=0.5 if alpha is None else alpha,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
        )
        self.decoder = ComplexLinear(
            self.latent_size,
            self.output_size,
            bias=True,
            interleaved=True,
            device=device,
            dtype=dtype,
        )
        kaiming_uniform_(self.decoder.weight, nonlinearity="linear")

        # }}}

    @torch.compiler.disable
    @torch.no_grad()
    def fit_dmd(
        self,
        x_latent: torch.Tensor,
        x_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.latent_size
        control_size = c_in.shape[-1]

        cx_latent = view_as_complex(x_latent)
        window_size = cx_latent.shape[1]

        X = cx_latent[:, :-1, :].reshape(-1, r)
        Y = cx_latent[:, 1:, :].reshape(-1, r)

        # NOTE: the control is constant along a trajectory, so it is simply repeated
        U = c_in[:, None, :].expand(-1, window_size - 1, -1).reshape(-1, control_size)

        # NOTE: drop transitions above the escape radius or at a fixed point
        if self.escape_radius is not None or self.fixed_point_rtol is not None:
            mask = compute_trajectory_mask(
                x_in, escape_radius=self.escape_radius, rtol=self.fixed_point_rtol
            ).transition.reshape(-1)

            X = X[mask]
            Y = Y[mask]
            U = U[mask]

        # build DMDc operators
        dmd = build_dense_dmdc(X, U, Y, method=self.dmd_method, eps=self.ridge_eps)

        assert dmd.A.shape == (r, r)
        assert dmd.B.shape == (control_size, r)

        return cast(torch.Tensor, dmd.A), cast(torch.Tensor, dmd.B)

    def rollout_latent(
        self,
        cx_latent: torch.Tensor,
        c_in: torch.Tensor,
        A_dmd: torch.Tensor,
        B_dmd: torch.Tensor,
    ) -> torch.Tensor:
        if self.rollout_steps == 1:
            yhat_complex = cx_latent[:, :-1, :] @ A_dmd + c_in[:, None, :] @ B_dmd
            return view_as_real(yhat_complex)
        else:
            yhat_steps = [cx_latent[:, :1, :]]
            steps = cx_latent.shape[1] - 1
            for _ in range(steps):
                yhat_steps.append(yhat_steps[-1] @ A_dmd + c_in[:, None, :] @ B_dmd)

            return view_as_real(torch.cat(yhat_steps[1:], dim=1))

    # ty: ignore[invalid-mutable-override]
    def forward(self, x_in: torch.Tensor, c_in: torch.Tensor) -> DMDAutoencoderResult:
        r = self.latent_size
        batch_size, window_size = x_in.shape[:2]

        device = x_in.device
        dtype = torch.complex64

        # encode the state trajectory (the control `c` is handled by DMDc below)
        x_latent = self.encoder(x_in)
        assert x_latent.shape == (batch_size, window_size, 2 * r)

        if self.training and self.dmd_enabled:
            # same-space DMDc on the encoded trajectory
            A_dmd, B_dmd = self.fit_dmd(x_latent, x_in, c_in)

            # evolve with the DMDc operator (either 1-step teacher-forcing or
            # multi-step rollout depending on rollout_steps). The control is
            # constant along the window
            cx_latent = view_as_complex(x_latent)
            yhat_latent = self.rollout_latent(cx_latent, c_in, A_dmd, B_dmd)
            assert yhat_latent.shape == (batch_size, window_size - 1, 2 * r)

            yhat_out = self.decoder(yhat_latent)
            assert yhat_out.shape == (batch_size, window_size - 1, x_in.shape[-1])
        else:
            A_dmd = torch.eye(r, r, device=device, dtype=dtype)
            B_dmd = torch.zeros((c_in.shape[-1], r), device=device, dtype=dtype)
            yhat_latent = None
            yhat_out = None

        x_out = self.decoder(x_latent)
        assert x_out.shape == x_in.shape

        return DMDAutoencoderResult(
            A_dmd=A_dmd,
            B_dmd=B_dmd,
            x_out=x_out,
            y_latent=x_latent if self.training and self.dmd_enabled else None,
            yhat_latent=yhat_latent,
            yhat_out=yhat_out,
        )


# }}}


# {{{ loss


class DMDLossResult(NamedTuple):
    total: torch.Tensor
    """Total value of the loss function. The gradient will be applied to this."""

    loss_rec: torch.Tensor
    """Reconstruction loss."""
    loss_lin: torch.Tensor
    """Linear approximation loss."""
    loss_pre: torch.Tensor
    """Linear prediction loss."""

    def tensor(self) -> torch.Tensor:
        return torch.stack([self.total, self.loss_rec, self.loss_lin, self.loss_pre])

    def detach(self) -> DMDLossResult:
        return DMDLossResult(
            total=self.total.detach(),
            loss_rec=self.loss_rec.detach(),
            loss_lin=self.loss_lin.detach(),
            loss_pre=self.loss_pre.detach(),
        )

    def backward(self, *args: Any, **kwargs: Any) -> None:
        self.total.backward(*args, **kwargs)


def masked_mse(
    y: torch.Tensor,
    yhat: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """Weighted mean squared error over the ``(batch, window)`` dimensions."""
    if weight is None:
        return torch.mean((y - yhat) ** 2)

    error = torch.sum(weight.unsqueeze(-1) * (y - yhat) ** 2)
    return error / (torch.sum(weight) * y.shape[-1] + 1.0e-8)


class DMDLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma_rec: float = 1.0,
        gamma_lin: float = 1.0,
        gamma_pre: float = 1.0,
        scale: float = 1.0,
        escape_radius: float | None = None,
        fixed_point_rtol: float | None = None,
    ) -> None:
        super().__init__()

        self.gamma_rec = gamma_rec / scale
        self.gamma_lin = gamma_lin / scale
        self.gamma_pre = gamma_pre / scale
        # escape radius (in dataset units) used to mask out pinned states
        self.escape_radius = escape_radius
        # relative step size below which a trajectory is considered settled
        self.fixed_point_rtol = fixed_point_rtol

    @property
    def dim(self) -> int:
        return 4

    # ty: ignore[invalid-mutable-override]
    def forward(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        *,
        y_latent: torch.Tensor | None = None,
        yhat_latent: torch.Tensor | None = None,
        yhat_out: torch.Tensor | None = None,
    ) -> DMDLossResult:
        # masks that drop pinned (clipped) and settled (fixed-point) states:
        # they are artifacts of the clipping / carry no dynamics (the escape
        # transition into the first pinned state is kept)
        if self.escape_radius is None and self.fixed_point_rtol is None:
            state_weight = None
            transition_weight = None
        else:
            mask = compute_trajectory_mask(
                x_in, escape_radius=self.escape_radius, rtol=self.fixed_point_rtol
            )
            state_weight = mask.state.to(x_in.dtype)
            transition_weight = mask.transition.to(x_in.dtype)

        # reconstruction loss
        loss_rec = masked_mse(x_in, x_out, state_weight)
        loss_rec = self.gamma_rec * loss_rec

        # latent space linearity loss (one-step, teacher-forced)
        if y_latent is not None and yhat_latent is not None:
            loss_lin = masked_mse(y_latent[:, 1:, :], yhat_latent, transition_weight)
            loss_lin = self.gamma_lin * loss_lin
        else:
            loss_lin = torch.zeros_like(loss_rec)

        # linear reconstruction loss (one-step, teacher-forced)
        if yhat_out is not None:
            loss_pre = masked_mse(x_in[:, 1:, :], yhat_out, transition_weight)
            loss_pre = self.gamma_pre * loss_pre
        else:
            loss_pre = torch.zeros_like(loss_rec)

        return DMDLossResult(
            total=loss_rec + loss_lin + loss_pre,
            loss_rec=loss_rec,
            loss_lin=loss_lin,
            loss_pre=loss_pre,
        )


# }}}


# {{{ torch_train


class ControlWindowsDataset:
    """Sliding windows over ``x`` paired with the per-realization control ``c``.

    The control is constant along a trajectory, so it is fetched by realization
    index rather than being replicated over the window/time dimension.
    """

    def __init__(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        window_size: int,
        nwindows: int,
    ) -> None:
        from nneuroutil.data import SlidingWindowDataset

        self.windows = SlidingWindowDataset(
            x, window_size=window_size, nwindows=nwindows
        )
        if c.shape[0] != self.windows.nrealizations:
            raise ValueError(
                f"'c' must have one row per realization: {tuple(c.shape)} vs "
                f"{self.windows.nrealizations}"
            )

        self.c = c

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        (window,) = self.windows[index]
        return (
            cast("torch.Tensor", window),
            self.c[index // self.windows.nwindows],
        )


def torch_train(
    model: DMDAutoencoder,
    criterion: DMDLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X_train: torch.Tensor,
    X_validate: torch.Tensor,
    *,
    c_train: torch.Tensor,
    c_validate: torch.Tensor,
    max_norm: float = 100.0,
    window_size: int = 32,
    nwindows: int = 16,
    epochs: int = 50,
    batch_size: int = 2048,
    dmd_warmup_epochs: int = 0,
    patience_early_stop: int = 0,
    callback: Callable[[int, DMDAutoencoder], None] | None = None,
) -> tuple[DMDAutoencoder, torch.optim.Optimizer, torch.Tensor]:
    device = X_train.device
    compiled_model = torch.compile(model)
    # compiled_model = model

    train_dataset = ControlWindowsDataset(
        X_train, c_train, window_size=window_size, nwindows=nwindows
    )
    validate_dataset = ControlWindowsDataset(
        X_validate, c_validate, window_size=window_size, nwindows=nwindows
    )

    loader = DataLoader(
        train_dataset,  # ty: ignore[invalid-argument-type]
        batch_size=batch_size,
        shuffle=True,
        # NOTE: this is needed otherwise the DataLoader seems to want to use the
        # 'cpu' device even when there is another default device.. it is seeded
        # explicitly so that shuffling is reproducible
        generator=torch.Generator(device).manual_seed(torch.initial_seed()),
    )
    validate_loader = DataLoader(
        validate_dataset,  # ty: ignore[invalid-argument-type]
        batch_size=batch_size,
    )

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    num_bad_epochs = 0

    timer = TicTocTimer()
    losses = torch.empty((epochs, 2, criterion.dim), device=device)

    for epoch in range(epochs):
        # {{{ training

        model.train()
        model.dmd_enabled = epoch >= dmd_warmup_epochs
        if epoch == dmd_warmup_epochs and dmd_warmup_epochs > 0:
            log.info("Enabling DMD at epoch %03d", epoch + 1)

        loss_epoch = torch.zeros(criterion.dim, device=device)
        num_batches = 0

        timer.tic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, (batch_x, batch_c) in enumerate(loader):
            optimizer.zero_grad()

            predictions = compiled_model(batch_x, batch_c)
            loss = criterion(
                batch_x,
                x_out=predictions.x_out,
                y_latent=predictions.y_latent,
                yhat_latent=predictions.yhat_latent,
                yhat_out=predictions.yhat_out,
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_norm
            )

            optimizer.step()

            loss_epoch += loss.tensor().detach()
            num_batches += 1

            if i in {0, len(loader) - 1}:
                with torch.no_grad():
                    eigenvalues = torch.linalg.eigvals(predictions.A_dmd.cpu())
                    eigmax = eigenvalues[torch.abs(eigenvalues).argmax()].item()

                log.info(
                    r"    \[batch %03d] "
                    r"lam (%.3e, %.3e) total %.3e rec %.3e lin %.3e pre %.3e grad %.3e",
                    i,
                    eigmax.real,
                    eigmax.imag,
                    loss.total.item(),
                    loss.loss_rec.item(),
                    loss.loss_lin.item(),
                    loss.loss_pre.item(),
                    grad_norm,
                )

        loss_train = loss_epoch / num_batches

        # }}}

        # {{{ validation

        # NOTE: keep the model in training mode so that DMD is computed, but
        # do not track gradients.
        model.train()
        with torch.no_grad():
            loss_val_epoch = torch.zeros(criterion.dim, device=device)
            num_val_batches = 0

            for batch_x, batch_c in validate_loader:
                predictions = compiled_model(batch_x, batch_c)
                loss = criterion(
                    batch_x,
                    x_out=predictions.x_out,
                    y_latent=predictions.y_latent,
                    yhat_latent=predictions.yhat_latent,
                    yhat_out=predictions.yhat_out,
                )

                loss_val_epoch += loss.tensor()
                num_val_batches += 1

            loss_validate = loss_val_epoch / num_val_batches

        # }}}

        scheduler.step(loss_validate[0])  # ty: ignore[invalid-argument-type]

        timer.toc()

        losses[epoch, 0] = loss_train
        losses[epoch, 1] = loss_validate

        log.info(
            "[%03d/%03d] total %.5e rec %.3e lin %.3e pre %.3e (%s)",
            epoch + 1,
            epochs,
            loss_train[0],
            loss_train[1],
            loss_train[2],
            loss_train[3],
            timer.short(),
        )

        # {{{ best model and early stopping

        # NOTE: only consider epochs with DMD enabled. During the warmup the
        # validation loss only contains the reconstruction term, so it is not
        # comparable to the full DMD loss and would always be selected.
        if model.dmd_enabled:
            loss_v = loss_validate[0].item()
            if loss_v < best_loss:
                best_loss = loss_v
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                num_bad_epochs = 0
            elif patience_early_stop > 0:
                num_bad_epochs += 1
                if num_bad_epochs >= patience_early_stop:
                    log.info(
                        "Early stopping at epoch %03d (best %03d, loss %.5e)",
                        epoch + 1,
                        best_epoch,
                        best_loss,
                    )
                    losses = losses[: epoch + 1]
                    break

        # }}}

        if callback is not None:
            with torch.no_grad():
                callback(epoch, compiled_model)  # ty: ignore[invalid-argument-type]

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Best model at epoch %03d (loss %.5e)", best_epoch, best_loss)

    return model, optimizer, losses


# }}}


# {{{ torch_predict


def build_latent_dmd(
    model: DMDAutoencoder,
    Z: torch.Tensor,
    C: torch.Tensor,
    *,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build the DMDc operators in the latent space from the full dataset.

    ``Z`` has shape ``(ntraj, maxit, 2 d)`` — the state ``z_n`` in interleaved
    real format — and ``C`` has shape ``(ntraj, 1)`` — the constant control.
    """
    r = model.latent_size
    window_size = Z.shape[1]
    control_size = C.shape[-1]

    with torch.no_grad():
        model.eval()

        Xs = []
        Ys = []
        Us = []
        for start in range(0, Z.shape[0], block_size):
            z_block = Z[start : start + block_size]
            c_block = C[start : start + block_size]
            cx = view_as_complex(model.encoder(z_block))

            X_block = cx[:, :-1, :].reshape(-1, r)
            Y_block = cx[:, 1:, :].reshape(-1, r)
            U_block = (
                c_block[:, None, :]
                .expand(-1, window_size - 1, -1)
                .reshape(-1, control_size)
            )

            # drop pinned -> pinned and settled transitions (see DMDAutoencoder.forward)
            if model.escape_radius is not None or model.fixed_point_rtol is not None:
                mask = compute_trajectory_mask(
                    z_block,
                    escape_radius=model.escape_radius,
                    rtol=model.fixed_point_rtol,
                )
                w = mask.transition.reshape(-1)
                if torch.any(w):
                    X_block = X_block[w]
                    Y_block = Y_block[w]
                    U_block = U_block[w]

            Xs.append(X_block)
            Ys.append(Y_block)
            Us.append(U_block)

        X = torch.cat(Xs)
        Y = torch.cat(Ys)
        U = torch.cat(Us)

        dmd = build_dense_dmdc(X, U, Y, method=model.dmd_method, eps=model.ridge_eps)
        return cast("torch.Tensor", dmd.A), cast("torch.Tensor", dmd.B)


def torch_predict(
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
    B_dmd: torch.Tensor,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
    norm_factor: float = 1.0,
    device: Any = None,
) -> torch.Tensor:
    from commonlib import generate_grid

    d = int(model.input_size)  # state dim (the control `c` is handled separately)
    dtype = model.dtype
    assert isinstance(dtype, torch.dtype)
    assert dtype.is_floating_point

    with torch.no_grad():
        z = torch.zeros((resolution, resolution, 2 * d), dtype=dtype, device=device)
        c = cast(
            "torch.Tensor",
            generate_grid(
                *bbox,
                nx=resolution,
                ny=resolution,
                dtype=dtype.to_complex(),
                xp=array_api_compat.array_namespace(A_dmd),
                device=device,
            ),
        )
        c = (c / norm_factor)[..., None]

        for _ in range(1, maxit):
            # NOTE: the model already learned `A * norm_factor` so no rescaling
            # needed here. We just need it at the end to go back to physical space
            g = model.encoder(z)
            g_next = view_as_complex(g) @ A_dmd + c @ B_dmd
            z = model.decoder(view_as_real(g_next))

        z = z * norm_factor

    return view_as_complex(z)


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

        for k, label in enumerate(labels):
            (line,) = ax.semilogy(losses_np[:, 0, 1 + k], label=f"Train {label}")
            ax.semilogy(
                losses_np[:, 1, 1 + k],
                label=f"Val {label}",
                color=line.get_color(),
                ls="--",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize="x-small", ncols=2)


def _plot_predict(
    filename: pathlib.Path,
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
    B_dmd: torch.Tensor,
    config: DMDConfig,
    *,
    zmask: torch.Tensor,
    tag: str,
    overwrite: bool,
) -> None:
    assert config.norm_factor is not None

    Zpredict = torch_predict(
        model,
        A_dmd,
        B_dmd,
        bbox=config.bbox,
        resolution=2 * config.resolution,
        maxit=config.maxit,
        norm_factor=config.norm_factor,
    )

    escape_radius = config.escape_radius
    in_set = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2
    log.info("    Converged:    %d/%d", torch.sum(in_set), in_set.numel())

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

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-ae-dmd-quad.npz"

    outfile = outfile.with_stem(f"{outfile.stem}-{config.suffix}").with_suffix("")
    weightsfile = outfile.with_suffix(".pkl")

    if not overwrite and weightsfile.exists():
        log.error(
            "Weights file already exist (use --force to overwrite): '%s'", weightsfile
        )
        return 1

    # }}}

    # {{{ train

    from torch import optim

    d = mat.shape[0]

    # NOTE: `config.seed` already seeds the numpy split generator; seed torch as
    # well so that the model initialization and shuffling are reproducible
    log.info("Using seed: %d", config.seed)
    torch.manual_seed(config.seed)

    # NOTE: when resuming, the stored architecture is used to rebuild the model
    model_config: dict[str, Any]
    if checkpoint is not None:
        model_config = checkpoint["model_config"]
    else:
        latent_size = 2 * d if config.latent_size is None else config.latent_size
        model_config = {
            "input_size": d,
            "latent_size": latent_size,
            "hidden_size": config.hidden_size,
            "dmd_method": config.dmd_method,
            "rollout_steps": config.rollout_steps,
            "fixed_point_rtol": config.fixed_point_rtol,
            "ridge_eps": config.ridge_eps,
            "alpha": config.alpha,
        }

    model = DMDAutoencoder(**model_config, device=device, dtype=mat.dtype.to_real())

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.3,
        min_lr=1.0e-5,
        cooldown=2,
    )

    log.info("Network:\n%s", model)

    losses_previous: torch.Tensor | None = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        losses_previous = checkpoint["losses"]
        log.info(
            "Resuming from '%s' (%d epochs trained)",
            weights,
            checkpoint["num_epochs_trained"],
        )

    # {{{ generate dataset

    min_escape_iters = 1 * (config.maxit + 1) // 4

    x, _, mask, zmask = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        generate_timeseries_dataset(
            mat,
            bbox=config.bbox,
            resolution=config.resolution,
            # NOTE: we add 1 so that x.shape == (n, maxit, 2d)
            maxit=config.maxit + 1,
            escape_radius=config.escape_radius,
            min_escape_iters=min_escape_iters,
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

    # NOTE: normalize the picked states to unit second moment so that the
    # kaiming initialization of the quadratic activations is valid; the escape
    # radius is scaled to match
    if config.norm_factor is None:
        from dataclasses import replace

        norm_factor = torch.sqrt(torch.mean(torch.sum(x**2, dim=-1)))
        config = replace(config, norm_factor=norm_factor.item())
    assert config.norm_factor is not None

    model.escape_radius = config.escape_radius / config.norm_factor
    log.info("State normalization: norm_factor %.5e", config.norm_factor)

    # NOTE: the control is constant along a trajectory, so it is kept per
    # realization (shape `(n, 1)`) instead of being replicated over the window;
    # it is passed to the model separately from the sliding windows
    c = (c[mask] / config.norm_factor)[:, None]
    x = x / config.norm_factor

    (x_train, c_train), (x_validate, c_validate), (x_test, c_test) = cast(
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        random_split((x, c), config.split, rng=np.random.default_rng(config.seed)),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    def plot_predict(epoch: int, m: DMDAutoencoder) -> None:
        A_dmd, B_dmd = build_latent_dmd(m, x_train, c_train)
        _plot_predict(
            outfile,
            m,
            A_dmd,
            B_dmd,
            config,
            zmask=zmask,
            tag=f"{epoch:04d}",
            overwrite=overwrite,
        )

    assert config.norm_factor is not None

    target_scale = torch.mean(x_train**2).item()
    criterion = DMDLoss(
        gamma_rec=config.gamma_rec,
        gamma_lin=config.gamma_lin,
        gamma_pre=config.gamma_pre,
        scale=target_scale,
        escape_radius=config.escape_radius / config.norm_factor,
        fixed_point_rtol=config.fixed_point_rtol,
    )

    _, optimizer, losses = torch_train(
        model,
        criterion,
        optimizer,
        scheduler,
        x_train,
        x_validate,
        c_train=c_train,
        c_validate=c_validate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        window_size=config.window_size,
        nwindows=config.nwindows,
        dmd_warmup_epochs=config.dmd_warmup_epochs,
        patience_early_stop=config.early_stop_patience,
        callback=plot_predict,
    )
    A_dmd, B_dmd = build_latent_dmd(model, x_train, c_train)

    mem.add_record("After training")

    model.eval()
    with torch.no_grad():
        test_dataset = ControlWindowsDataset(
            x_test, c_test, window_size=config.window_size, nwindows=config.nwindows
        )
        test_loader = DataLoader(
            test_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=config.batch_size,
        )

        loss_test_epoch = torch.zeros(criterion.dim, device=device)
        num_test_batches = 0

        for batch_x, batch_c in test_loader:
            x_latent = model.encoder(batch_x)
            cx_latent = view_as_complex(x_latent)
            yhat_latent = model.rollout_latent(cx_latent, batch_c, A_dmd, B_dmd)
            yhat_out = model.decoder(yhat_latent)
            x_out = model.decoder(x_latent)

            loss = criterion(
                batch_x,
                x_out=x_out,
                y_latent=x_latent,
                yhat_latent=yhat_latent,
                yhat_out=yhat_out,
            )

            loss_test_epoch += loss.tensor()
            num_test_batches += 1

        loss_test = loss_test_epoch / num_test_batches
        log.info(
            "[%03d/%03d] test_total %.5e test_rec %.5e test_lin %.5e test_pre %.5e",
            config.epochs,
            config.epochs,
            loss_test[0].item(),
            loss_test[1].item(),
            loss_test[2].item(),
            loss_test[3].item(),
        )

    mem.add_record("After test loss")

    # }}}

    # {{{ checkpoint

    from dataclasses import asdict

    if losses_previous is not None:
        losses = torch.cat((losses_previous.to(losses.device), losses))

    num_epochs_trained = len(losses)
    torch.save(
        {
            # model
            "provenance": provenance,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,
            "a_dmd": A_dmd,
            "b_dmd": B_dmd,
            # loss
            "num_epochs_trained": num_epochs_trained,
            "losses": losses,
            "loss_test": loss_test,
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

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, overwrite=overwrite)
    _plot_predict(
        outfile,
        model,
        A_dmd,
        B_dmd,
        config,
        zmask=zmask,
        tag="final",
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
    config = DMDConfig.from_sources(checkpoint["config"], overrides)

    losses = checkpoint["losses"]
    zmask = checkpoint["zmask"]
    A_dmd = checkpoint["a_dmd"]
    B_dmd = checkpoint["b_dmd"]

    if outfile is None:
        outfile = weightsfile.with_suffix("")
    outfile = outfile.with_suffix("")

    # }}}

    # {{{ load model

    assert config.norm_factor is not None

    model = DMDAutoencoder(
        **checkpoint["model_config"], device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["model"])
    model.escape_radius = config.escape_radius / config.norm_factor
    model.fixed_point_rtol = config.fixed_point_rtol
    model.ridge_eps = config.ridge_eps
    model.dmd_method = config.dmd_method
    model.rollout_steps = config.rollout_steps
    model.eval()

    mem.add_record("After model load")

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, overwrite=overwrite)
    _plot_predict(
        outfile,
        model,
        A_dmd,
        B_dmd,
        config,
        zmask=zmask,
        tag="final",
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
        "--dmd-warmup-epochs",
        type=int,
        default=None,
        help=(
            "Train this many epochs without DMD before enabling it "
            "(from checkpoint when resuming)"
        ),
    )
    parser.add_argument(
        "--fixed-point-rtol",
        type=float,
        default=None,
        help=(
            "Relative step size below which a trajectory is considered settled "
            "and masked out (default: 1e-3, from checkpoint when resuming)"
        ),
    )
    parser.add_argument(
        "--ridge-eps",
        type=float,
        default=None,
        help=(
            "Ridge regularization for the DMDc least-squares solve "
            "(default: 1e-3, from checkpoint when resuming)"
        ),
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=None,
        help=(
            "Latent (complex) dimension of the autoencoder "
            "(default: 2 * d, from checkpoint when resuming)"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Blend between the linear and quadratic activation, in [0, 1] "
            "(default: 0.75, from checkpoint when resuming)"
        ),
    )
    parser.add_argument(
        "--dmd-method",
        choices=["pinv", "ridge"],
        default=None,
        help="DMD solver method: 'pinv' (truncated SVD) or 'ridge' (L2-regularized)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden dimension of the encoder (default: latent_size)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Number of rollout steps for DMD prediction in training "
        "(1 for one-step teacher-forcing)",
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
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

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

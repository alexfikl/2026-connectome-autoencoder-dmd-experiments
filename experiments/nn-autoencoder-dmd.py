# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import array_api_compat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from nneuroutil.dmdc import build_dense_dmdc
from nneuroutil.helpers import (
    FuzzyChoiceAction,
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


# {{{ masks


class TrajectoryMasks(NamedTuple):
    state: torch.Tensor
    transition: torch.Tensor


def compute_masks(
    x: torch.Tensor,
    escape_radius: float | None = None,
    *,
    rtol: float | None = None,
    atol: float = 1.0e-8,
) -> TrajectoryMasks:
    """Compute the masks of usable states and transitions in ``x``.

    Two kinds of states carry no dynamics and are masked out:

    * *pinned* (clipped) states: their squared norm exceeds the escape radius
      (see ``commonlib.generate_dataset``). The transition into the first
      pinned state (the escape transition) is kept.
    * *settled* states: the trajectory has converged to a fixed point and only
      moves by a relative step ``|z_{n + 1} - z_n| / (|z_n| + atol)`` below *rtol*.

    With both *escape_radius* and *rtol* unset nothing is masked.
    """
    batch_size, window_size = x.shape[:2]
    device = x.device

    zx = view_as_complex(x)
    zn = torch.sum(zx.real**2 + zx.imag**2, dim=-1)

    if escape_radius is None:
        active = torch.ones_like(zn, dtype=torch.bool)
    else:
        active = zn <= escape_radius**2

    # a transition is dynamic if the trajectory still moves along it
    if rtol is None:
        moving = torch.ones(
            (batch_size, window_size - 1), dtype=torch.bool, device=device
        )
    else:
        dz = torch.linalg.vector_norm(zx[:, 1:, :] - zx[:, :-1, :], dim=-1)
        zn = torch.sqrt(zn[:, :-1])
        moving = dz > rtol * (zn + atol)

    # a state is transient if the trajectory moves into or out of it
    empty = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    moving_in = torch.cat([empty, moving], dim=1)
    moving_out = torch.cat([moving, empty], dim=1)

    return TrajectoryMasks(
        state=active & (moving_in | moving_out),
        transition=active[:, :-1] & moving,
    )


# }}}


# {{{ model


def make_layers(
    sizes: tuple[int, ...],
    nonlinearities: tuple[str, ...],
    *,
    alpha: float | None = None,
    device: str | torch.device | None = None,
    dtype: Any = None,
) -> list[nn.Module]:
    """Construct a sequential module of layers of the form ``f(W x) + b``."""
    from itertools import pairwise

    from nneuroutil.torch_extras import (
        Bias,
        ComplexBlendedQuadratic,
        ComplexLinear,
        ComplexQuadratic,
        LeakyModReLU,
        kaiming_uniform_,
    )

    layers = []
    for i, ((n_in, n_out), nonlinearity) in enumerate(
        zip(pairwise(sizes), nonlinearities, strict=True)
    ):
        # build nonlinearity
        if nonlinearity == "cquadratic":
            activation = ComplexQuadratic(interleaved=True)
        elif nonlinearity == "cblended_quadratic":
            activation = ComplexBlendedQuadratic(
                0.5 if alpha is None else alpha, interleaved=True
            )
        elif nonlinearity == "leaky_modrelu":
            activation = LeakyModReLU(
                bias=-0.1,
                alpha=0.1 if alpha is None else alpha,
                interleaved=True,
            )
        else:
            raise ValueError(f"unsupported nonlinearity: {nonlinearity!r}")

        # build linear layer
        linear = ComplexLinear(
            n_in, n_out, bias=False, interleaved=True, device=device, dtype=dtype
        )
        kaiming_uniform_(
            linear.weight,
            # NOTE: the init is done on the pre-activations, so first layer is linear
            nonlinearity="linear" if i == 0 else nonlinearity,
            param=alpha,
        )

        # add the whole thing as `f(W z) + b`
        layers.append(linear)
        layers.append(activation)
        layers.append(Bias(2 * n_out, device=device, dtype=dtype))

    return layers


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
        encoder_layer_sizes: int | tuple[int, ...] | None = None,
        decoder_layer_sizes: int | tuple[int, ...] | None = None,
        dmd_enabled: bool = False,
        escape_radius: float | None = None,
        settle_tol: float | None = None,
        control_size: int = 1,
        ridge_eps: float = 1.0e-3,
        alpha: float | None = None,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        if encoder_layer_sizes is None:
            encoder_layer_sizes = input_size

        if isinstance(encoder_layer_sizes, int):
            encoder_layer_sizes = (encoder_layer_sizes,)

        if decoder_layer_sizes is None:
            decoder_layer_sizes = encoder_layer_sizes[::-1]

        if isinstance(decoder_layer_sizes, int):
            decoder_layer_sizes = (decoder_layer_sizes,)

        if not encoder_layer_sizes:
            raise ValueError("'encoder_layer_sizes' cannot be empty")

        if not decoder_layer_sizes:
            raise ValueError("'decoder_layer_sizes' cannot be empty")

        if dtype is None:
            dtype = torch.float32

        super().__init__()

        self.input_size = input_size
        self.output_size = decoder_layer_sizes[-1]
        self.latent_size = encoder_layer_sizes[-1]
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.dtype = dtype

        self.dmd_enabled = dmd_enabled
        # escape radius (in dataset units, i.e. after norm_factor scaling) used
        # to mask out pinned states; None disables the masking
        self.escape_radius = escape_radius
        # relative step size below which a trajectory is considered settled
        # (dimensionless, so no norm_factor scaling needed); None disables it
        self.settle_tol = settle_tol
        # number of (complex) control inputs fed to the DMDc operator
        self.control_size = control_size
        # ridge regularization for the DMDc least-squares solve
        self.ridge_eps = ridge_eps

        # {{{ construct autoencoder

        # encoder
        nlayers = len(encoder_layer_sizes)
        encoder_layers = make_layers(
            (self.input_size, *encoder_layer_sizes),
            # (*(("cblended_quadratic",) * (nlayers - 1)), "cquadratic"),
            ("leaky_modrelu",) * nlayers,
            alpha=alpha,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        nlayers = len(decoder_layer_sizes)
        decoder_layers = make_layers(
            (self.latent_size, *decoder_layer_sizes),
            # ("cquadratic", *(("cblended_quadratic",) * (nlayers - 1))),
            ("leaky_modrelu",) * nlayers,
            alpha=alpha,
            device=device,
            dtype=dtype,
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # }}}

    def fit_dmd(
        self,
        cx_latent: torch.Tensor,
        x_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.latent_size
        window_size = cx_latent.shape[1]

        X = cx_latent[:, :-1, :].reshape(-1, r)
        Y = cx_latent[:, 1:, :].reshape(-1, r)
        # the control is constant along a trajectory, so it is simply repeated
        U = (
            c_in[:, None, :]
            .expand(-1, window_size - 1, -1)
            .reshape(-1, self.control_size)
        )

        # drop pinned -> pinned and settled transitions: the former are
        # artifacts of the clipping in generate_dataset, not actual dynamics
        # (the escape transition into the first pinned state is kept), while
        # the latter only sit at a fixed point
        if self.escape_radius is not None or self.settle_tol is not None:
            w = compute_masks(x_in, self.escape_radius, rtol=self.settle_tol).transition
            w = w.reshape(-1)
            if torch.any(w):
                X = X[w]
                Y = Y[w]
                U = U[w]

        dmd = build_dense_dmdc(X, U, Y, method="ridge", eps=self.ridge_eps)
        assert dmd.A.shape == (r, r)
        assert dmd.B.shape == (self.control_size, r)
        return cast("torch.Tensor", dmd.A), cast("torch.Tensor", dmd.B)

    # ty: ignore[invalid-mutable-override]
    def forward(self, x_in: torch.Tensor, c_in: torch.Tensor) -> DMDAutoencoderResult:
        r = self.latent_size
        batch_size, window_size = x_in.shape[:2]

        # encode the state trajectory (the control `c` is handled by DMDc below)
        x_latent = self.encoder(x_in)
        assert x_latent.shape == (batch_size, window_size, 2 * r)

        if self.training and self.dmd_enabled:
            # same-space DMDc on the encoded trajectory
            cx_latent = view_as_complex(x_latent)

            with torch.no_grad():
                A_dmd, B_dmd = self.fit_dmd(cx_latent, x_in, c_in)

            # evolve with the DMDc operator. NOTE: only one-step (teacher-forced)
            # predictions are used for training: a multi-step rollout overflows
            # as soon as A is expansive along the escape directions, which
            # produces inf losses and NaN gradients
            yhat_latent = cx_latent[:, :-1, :] @ A_dmd + c_in[:, None, :] @ B_dmd
            yhat_latent = view_as_real(yhat_latent)
            assert yhat_latent.shape == (batch_size, window_size - 1, 2 * r)

            yhat_out = self.decoder(yhat_latent)
            assert yhat_out.shape == (batch_size, window_size - 1, x_in.shape[-1])
        else:
            A_dmd = torch.eye(r, r, device=x_in.device, dtype=torch.complex64)
            B_dmd = torch.zeros(
                (self.control_size, r), device=x_in.device, dtype=torch.complex64
            )
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
        settle_tol: float | None = None,
    ) -> None:
        super().__init__()

        self.gamma_rec = gamma_rec / scale
        self.gamma_lin = gamma_lin / scale
        self.gamma_pre = gamma_pre / scale
        # escape radius (in dataset units) used to mask out pinned states
        self.escape_radius = escape_radius
        # relative step size below which a trajectory is considered settled
        self.settle_tol = settle_tol

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
        if self.escape_radius is None and self.settle_tol is None:
            state_weight = None
            transition_weight = None
        else:
            masks = compute_masks(x_in, self.escape_radius, rtol=self.settle_tol)
            state_weight = masks.state.to(x_in.dtype)
            transition_weight = masks.transition.to(x_in.dtype)

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
        # 'cpu' device even when there is another default device..
        generator=torch.Generator(device),
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

        loss_v = loss_validate[0].item()
        if loss_v < best_loss:
            best_loss = loss_v
            best_state = model.state_dict()
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
        # model.load_state_dict(best_state)
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
                .reshape(-1, model.control_size)
            )

            # drop pinned -> pinned and settled transitions (see DMDAutoencoder.forward)
            if model.escape_radius is not None or model.settle_tol is not None:
                w = compute_masks(
                    z_block, model.escape_radius, rtol=model.settle_tol
                ).transition
                w = w.reshape(-1)
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

        dmd = build_dense_dmdc(X, U, Y, method="ridge", eps=model.ridge_eps)
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
    import array_api_compat

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
    zmask: torch.Tensor,
    *,
    basename: pathlib.Path,
    bbox: tuple[float, float, float, float],
    overwrite: bool,
) -> None:
    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(zmask.cpu().numpy(), extent=bbox, cmap="binary")


def _plot_loss(
    losses: torch.Tensor,
    *,
    basename: pathlib.Path,
    overwrite: bool,
) -> None:
    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
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
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
    B_dmd: torch.Tensor,
    *,
    tag: str,
    basename: pathlib.Path,
    bbox: tuple[float, float, float, float],
    resolution: int,
    maxit: int,
    norm_factor: float,
    escape_radius: float,
    overwrite: bool,
) -> None:
    Zpredict = torch_predict(
        model,
        A_dmd,
        B_dmd,
        bbox=bbox,
        resolution=2 * resolution,
        maxit=maxit,
        norm_factor=norm_factor,
    )
    in_set = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2
    log.info("    Converged:    %d/%d", torch.sum(in_set), in_set.numel())

    with figure(
        basename.with_stem(f"{basename.stem}-predict-{tag}"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.imshow(in_set.cpu().numpy(), extent=bbox, cmap="binary")


# }}}


# {{{ train


def train(
    filename: pathlib.Path,
    *,
    weights: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    escape_radius: float | None = 10.0,
    settle_tol: float | None = None,
    ridge_eps: float | None = None,
    resolution: int | None = None,
    maxit: int | None = None,
    epochs: int = 100,
    batch_size: int | None = None,
    window_size: int | None = None,
    nwindows: int | None = None,
    dmd_warmup_epochs: int | None = None,
    split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    early_stop_patience: int = 0,
    overwrite: bool = False,
) -> int:
    from commonlib import (
        determine_escape_radius,
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

    checkpoint: dict[str, Any] | None = None
    if weights is not None:
        if not weights.exists():
            log.error("Weights file does not exist: '%s'.", weights)
            return 1

        checkpoint = torch.load(weights, map_location=device, weights_only=True)

        # NOTE: keep the original provenance and record each resume
        previous = checkpoint["provenance"]
        previous.setdefault("resumes", []).append(provenance)
        provenance = previous

        dataidx = checkpoint["dataidx"] if dataidx is None else dataidx
        bbox = checkpoint["bbox"] if bbox is None else bbox
        resolution = checkpoint["resolution"] if resolution is None else resolution
        maxit = checkpoint["maxit"] if maxit is None else maxit
        batch_size = checkpoint["batch_size"] if batch_size is None else batch_size
        window_size = checkpoint["window_size"] if window_size is None else window_size
        nwindows = checkpoint["nwindows"] if nwindows is None else nwindows
        dmd_warmup_epochs = (
            checkpoint["dmd_warmup_epochs"]
            if dmd_warmup_epochs is None
            else dmd_warmup_epochs
        )
        split = checkpoint["split"] if split is None else split
        seed = checkpoint["seed"] if seed is None else seed

        settle_tol = checkpoint["settle_tol"] if settle_tol is None else settle_tol
        ridge_eps = checkpoint["ridge_eps"] if ridge_eps is None else ridge_eps
        lr = checkpoint["lr"]
        norm_factor = checkpoint["norm_factor"]

        datakey = checkpoint["datakey"]
        escape_radius = checkpoint["escape_radius"]
    else:
        dataidx = 5 if dataidx is None else dataidx
        bbox = (-0.03, 0.03, -0.03, 0.03) if bbox is None else bbox
        resolution = 128 if resolution is None else resolution
        maxit = 32 if maxit is None else maxit
        batch_size = 512 if batch_size is None else batch_size
        window_size = 32 if window_size is None else window_size
        nwindows = 1 if nwindows is None else nwindows
        dmd_warmup_epochs = 10 if dmd_warmup_epochs is None else dmd_warmup_epochs
        split = (0.8, 0.1, 0.1) if split is None else split
        if seed is None:
            seed = int(np.random.SeedSequence().entropy)  # ty: ignore[invalid-argument-type]

        settle_tol = 1.0e-3 if settle_tol is None else settle_tol
        ridge_eps = 1.0e-3 if ridge_eps is None else ridge_eps
        lr = 1.0e-4
        norm_factor = 1.0

    # }}}

    # {{{ validate inputs

    try:
        matrices = cast(
            torch.Tensor,
            load_matrix_datasets(filename, key=datakey, device=device, xp=torch),
        )
        mat = matrices[dataidx].to(torch.complex64)
    except Exception as exc:
        log.error("Failed to load dataset '%s': %s", filename.name, exc)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    mem.add_record("After data load")

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-ae-dmd-quad.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{device.type}"
    )
    basename = outfile.with_suffix("")
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

    # NOTE: when resuming, the stored architecture is used to rebuild the model
    model_config: dict[str, Any]
    if checkpoint is not None:
        model_config = checkpoint["model_config"]
    else:
        model_config = {
            "input_size": d,
            "encoder_layer_sizes": (d, d // 2),
            "decoder_layer_sizes": (d // 2, d),
            "control_size": 1,
            "alpha": None,
        }

    model = DMDAutoencoder(**model_config, device=device, dtype=mat.dtype.to_real())
    # NOTE: the data is scaled by norm_factor, so scale the radius as well
    model.escape_radius = escape_radius / norm_factor
    # NOTE: settle_tol is a relative step size, so it needs no scaling
    model.settle_tol = settle_tol
    model.ridge_eps = ridge_eps

    optimizer = optim.Adam(model.parameters(), lr=lr)
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

    min_escape_iters = 1 * (maxit + 1) // 4

    x, _, mask, zmask = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        generate_timeseries_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            # NOTE: we add 1 so that x.shape == (n, maxit, 2d)
            maxit=maxit + 1,
            escape_radius=escape_radius,
            min_escape_iters=min_escape_iters,
        ),
    )

    c = generate_grid(
        *bbox,
        nx=resolution,
        ny=resolution,
        dtype=mat.dtype,
        xp=array_api_compat.array_namespace(mat),
        device=device,
    )

    # NOTE: the control is constant along a trajectory, so it is kept per
    # realization (shape `(n, 1)`) instead of being replicated over the window;
    # it is passed to the model separately from the sliding windows
    c = (c[mask] / norm_factor)[:, None]
    x = x / norm_factor

    (x_train, c_train), (x_validate, c_validate), (x_test, c_test) = cast(
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        random_split((x, c), split, rng=np.random.default_rng(seed)),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    def plot_predict(epoch: int, m: DMDAutoencoder) -> None:
        A_dmd, B_dmd = build_latent_dmd(m, x_train, c_train)
        _plot_predict(
            m,
            A_dmd,
            B_dmd,
            tag=f"{epoch:04d}",
            basename=basename,
            bbox=bbox,
            resolution=resolution,
            maxit=maxit,
            norm_factor=norm_factor,
            escape_radius=escape_radius,
            overwrite=overwrite,
        )

    target_scale = torch.mean(torch.conj(x_train) * x_train).real.item()
    criterion = DMDLoss(
        scale=target_scale,
        escape_radius=escape_radius / norm_factor,
        settle_tol=settle_tol,
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
        epochs=epochs,
        batch_size=batch_size,
        window_size=window_size,
        nwindows=nwindows,
        dmd_warmup_epochs=dmd_warmup_epochs,
        patience_early_stop=early_stop_patience,
        callback=plot_predict,
    )
    A_dmd, B_dmd = build_latent_dmd(model, x_train, c_train)

    mem.add_record("After training")

    model.train()
    with torch.no_grad():
        test_dataset = ControlWindowsDataset(
            x_test, c_test, window_size=window_size, nwindows=nwindows
        )
        test_loader = DataLoader(
            test_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=batch_size,
        )

        loss_test_epoch = torch.zeros(criterion.dim, device=device)
        num_test_batches = 0

        for batch_x, batch_c in test_loader:
            predictions = model(batch_x, batch_c)
            loss = criterion(
                batch_x,
                x_out=predictions.x_out,
                y_latent=predictions.y_latent,
                yhat_latent=predictions.yhat_latent,
                yhat_out=predictions.yhat_out,
            )

            loss_test_epoch += loss.tensor()
            num_test_batches += 1

        loss_test = loss_test_epoch / num_test_batches
        log.info(
            "[%03d/%03d] test_rec %.5e test_lin %.5e test_pre %.5e",
            epochs,
            epochs,
            loss_test[1].item(),
            loss_test[2].item(),
            loss_test[3].item(),
        )

    mem.add_record("After test loss")

    # }}}

    # {{{ checkpoint

    if losses_previous is not None:
        losses = torch.cat((losses_previous.to(losses.device), losses))

    num_epochs_trained = len(losses)
    torch.save(
        {
            # model
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,
            "a_dmd": A_dmd,
            "b_dmd": B_dmd,
            "ridge_eps": ridge_eps,
            "norm_factor": norm_factor,
            # loss
            "num_epochs_trained": num_epochs_trained,
            "losses": losses,
            "loss_test": loss_test,
            # dataset
            "inputfile": str(filename),
            "datakey": datakey,
            "dataidx": dataidx,
            "bbox": bbox,
            "escape_radius": escape_radius,
            "settle_tol": settle_tol,
            "resolution": resolution,
            "maxit": maxit,
            "zmask": zmask,
            # hyperparameters
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "lr": lr,
            "batch_size": batch_size,
            "window_size": window_size,
            "nwindows": nwindows,
            "dmd_warmup_epochs": dmd_warmup_epochs,
            "split": split,
            "seed": seed,
            # provenance
            "provenance": provenance,
        },
        weightsfile,
    )
    log.info("Saved checkpoint: '%s'.", weightsfile)

    mem.add_record("After checkpoint")

    # }}}

    # }}}

    # {{{ plot

    _plot_dataset(zmask, basename=basename, bbox=bbox, overwrite=overwrite)
    _plot_loss(losses, basename=basename, overwrite=overwrite)
    _plot_predict(
        model,
        A_dmd,
        B_dmd,
        tag="final",
        basename=basename,
        bbox=bbox,
        resolution=resolution,
        maxit=maxit,
        norm_factor=norm_factor,
        escape_radius=escape_radius,
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
    *,
    outfile: pathlib.Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    resolution: int | None = None,
    maxit: int | None = None,
    overwrite: bool = False,
) -> int:
    if not weightsfile.exists():
        log.error("Weights file does not exist: '%s'.", weightsfile)
        return 1

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)

    # NOTE: parameters stored in the checkpoint, overridable from the command line
    bbox = checkpoint["bbox"] if bbox is None else bbox
    resolution = checkpoint["resolution"] if resolution is None else resolution
    maxit = checkpoint["maxit"] if maxit is None else maxit
    escape_radius = checkpoint["escape_radius"]
    settle_tol = checkpoint.get("settle_tol")
    norm_factor = checkpoint["norm_factor"]

    losses = checkpoint["losses"]
    zmask = checkpoint["zmask"]
    A_dmd = checkpoint["a_dmd"]
    B_dmd = checkpoint["b_dmd"]
    ridge_eps = checkpoint.get("ridge_eps", 1.0e-3)

    if outfile is None:
        basename = weightsfile.with_suffix("")
    else:
        basename = outfile.with_suffix("")

    # {{{ load model

    model = DMDAutoencoder(
        **checkpoint["model_config"], device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["model"])
    model.escape_radius = escape_radius / norm_factor
    model.settle_tol = settle_tol
    model.ridge_eps = ridge_eps
    model.eval()

    mem.add_record("After model load")

    # }}}

    # {{{ plot

    _plot_dataset(zmask, basename=basename, bbox=bbox, overwrite=overwrite)
    _plot_loss(losses, basename=basename, overwrite=overwrite)
    _plot_predict(
        model,
        A_dmd,
        B_dmd,
        tag="final",
        basename=basename,
        bbox=bbox,
        resolution=resolution,
        maxit=maxit,
        norm_factor=norm_factor,
        escape_radius=escape_radius,
        overwrite=overwrite,
    )

    # }}}

    mem.add_record("Finished")

    log.info("Memory:\n%s", mem)

    return 0


# }}}


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    # {{{ parser

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default device used by the framework",
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    # {{{ train

    subparser = subparsers.add_parser(
        "train",
        formatter_class=HelpFormatter,
        description="Train a DMD autoencoder network on a dataset.",
    )
    subparser.add_argument("filename", type=pathlib.Path)
    subparser.add_argument(
        "-w",
        "--weights",
        type=pathlib.Path,
        default=None,
        help="Checkpoint to resume training from (starts from scratch if not given)",
    )
    subparser.add_argument(
        "--dataidx",
        type=int,
        default=None,
        help="Index of the matrix to reconstruct (from checkpoint when resuming)",
    )
    subparser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=None,
        help=(
            "Resolution (in the c complex plane) of the images "
            "(from checkpoint when resuming)"
        ),
    )
    subparser.add_argument(
        "--maxit",
        type=int,
        default=None,
        help="Maximum number of iterations (from checkpoint when resuming)",
    )
    subparser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("RE_MIN", "RE_MAX", "IM_MIN", "IM_MAX"),
        help="Bounding box in the complex plane (from checkpoint when resuming)",
    )
    subparser.add_argument(
        "--dmd-warmup-epochs",
        type=int,
        default=None,
        help=(
            "Train this many epochs without DMD before enabling it "
            "(from checkpoint when resuming)"
        ),
    )
    subparser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training loss does not improve for this many epochs (0 disables)",
    )
    subparser.add_argument(
        "--settle-tol",
        type=float,
        default=None,
        help=(
            "Relative step size below which a trajectory is considered settled "
            "and masked out (default: 1e-3, from checkpoint when resuming)"
        ),
    )
    subparser.add_argument(
        "--ridge-eps",
        type=float,
        default=None,
        help=(
            "Ridge regularization for the DMDc least-squares solve "
            "(default: 1e-3, from checkpoint when resuming)"
        ),
    )

    # }}}

    # {{{ infer

    subparser = subparsers.add_parser(
        "infer",
        formatter_class=HelpFormatter,
        description="Run inference with a trained DMD autoencoder network.",
    )
    subparser.add_argument("weightsfile", type=pathlib.Path)
    subparser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=None,
        help="Override the image resolution stored in the checkpoint",
    )
    subparser.add_argument(
        "--maxit",
        type=int,
        default=None,
        help="Override the maximum number of iterations stored in the checkpoint",
    )
    subparser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("RE_MIN", "RE_MAX", "IM_MIN", "IM_MAX"),
        help="Override the bounding box stored in the checkpoint",
    )

    # }}}

    # }}}

    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        bbox = None if args.bbox is None else tuple(args.bbox)

        if args.command == "train":
            raise SystemExit(
                train(
                    args.filename,
                    weights=args.weights,
                    dataidx=args.dataidx,
                    resolution=args.resolution,
                    bbox=bbox,
                    maxit=args.maxit,
                    dmd_warmup_epochs=args.dmd_warmup_epochs,
                    early_stop_patience=args.early_stop_patience,
                    settle_tol=args.settle_tol,
                    ridge_eps=args.ridge_eps,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        elif args.command == "infer":
            raise SystemExit(
                infer(
                    args.weightsfile,
                    bbox=bbox,
                    resolution=args.resolution,
                    maxit=args.maxit,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise SystemExit(1)

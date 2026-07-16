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

from nneuroutil.array_api_extras import interleave
from nneuroutil.helpers import FuzzyChoiceAction, TicTocTimer, module_logger
from nneuroutil.torch_extras import available_device_names, torch_default_device
from nneuroutil.visualization import figure

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a DMD Autoencoder neural network with quadratic activation functions.

Example:

    > {SCRIPT_PATH.name} --device cuda dataset.npz
"""


# {{{ model


def view_as_complex(x: torch.Tensor):
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))


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
        kaiming_uniform_,
    )

    layers = []
    for (n_in, n_out), nonlinearity in zip(
        pairwise(sizes), nonlinearities, strict=True
    ):
        # build nonlinearity
        if nonlinearity == "quadratic":
            activation = ComplexQuadratic(interleaved=True)
        elif nonlinearity == "blended_quadratic":
            activation = ComplexBlendedQuadratic(
                0.5 if alpha is None else alpha, interleaved=True
            )
        else:
            raise ValueError(f"unsupported nonlinearity: {nonlinearity!r}")

        # build linear layer
        linear = ComplexLinear(
            n_in, n_out, bias=False, interleaved=True, device=device, dtype=dtype
        )
        kaiming_uniform_(
            linear.weight,
            nonlinearity=nonlinearity,
            param=alpha,
        )

        # add the whole thing as `f(W z) + b`
        layers.append(linear)
        layers.append(activation)
        layers.append(Bias(2 * n_out, device=device, dtype=dtype))

    return layers


class DMDAutoencoderResult(NamedTuple):
    A_dmd: torch.Tensor
    """DMD approximated linear operator in the latent space."""

    x_out: torch.Tensor
    """Output of the autoencoder."""

    y_latent: torch.Tensor
    """Output of the encoder."""

    yhat_latent: torch.Tensor
    """DMD reconstructed latent vector :attr:`x_latent`."""

    yhat_out: torch.Tensor
    """Output of the decoder applied to :atr:`yhat_latent`."""


class DMDAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        encoder_layer_sizes: int | tuple[int, ...] | None = None,
        decoder_layer_sizes: int | tuple[int, ...] | None = None,
        dmd_enabled: bool = False,
        eps: float = 1.0e-6,
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
        self.eps = eps

        # {{{ construct autoencoder

        # encoder
        nlayers = len(encoder_layer_sizes)
        encoder_layers = make_layers(
            (self.input_size, *encoder_layer_sizes),
            (*(("blended_quadratic",) * (nlayers - 1)), "quadratic"),
            alpha=alpha,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        nlayers = len(decoder_layer_sizes)
        decoder_layers = make_layers(
            (self.latent_size, *decoder_layer_sizes),
            ("quadratic", *(("blended_quadratic",) * (nlayers - 1))),
            alpha=alpha,
            device=device,
            dtype=dtype,
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # }}}

    def forward(self, x_in: torch.Tensor) -> DMDAutoencoderResult:
        # NOTE: x has a shape of (batch_size, window_size, 2 r) and we want to
        # do DMD on it, so we go
        # 1. X = x[:, :-1, :].reshape(-1, 2 * r)
        #    Y = x[:, 1:, :].reshape(-1, 2 * r)
        # 2. build the DMD operator for this batch with (X, Y)

        r = self.latent_size
        batch_size, window_size, d = x_in.shape
        d = (d - 2) // 2

        x = x_in[:, :, :-2]
        c = x_in[:, :, -2:]
        c = c.repeat(1, 1, d)
        assert c.shape == (batch_size, window_size, 2 * d)

        # go to latent space
        x_latent = self.encoder(x)
        y_latent = self.encoder(x - c)
        assert x_latent.shape == (batch_size, window_size, 2 * r)
        assert y_latent.shape == (batch_size, window_size, 2 * r)

        x_out = self.decoder(x_latent)
        assert x_out.shape == (batch_size, window_size, 2 * d)

        # shape of the latent space
        if self.training and self.dmd_enabled:
            # separate data for DMD
            # NOTE: perform DMD on the complex variables to maintain the complex
            # dynamics (the rest of the autoencoder is also complex-friendly)
            cx_latent = view_as_complex(x_latent)
            cy_latent = view_as_complex(y_latent)
            X = cx_latent[:, :-1, :].reshape(-1, r)
            Y = cy_latent[:, 1:, :].reshape(-1, r)

            # construct DMD operator in latent space
            A_dmd = Y.T @ torch.linalg.pinv(X.T, rtol=self.eps)
            assert A_dmd.shape == (r, r)

            # predict with DMD in latent space (single-step)
            yhat_latent = cx_latent @ A_dmd.T
            yhat_latent = interleave(yhat_latent.real, yhat_latent.imag, axis=-1)
            assert yhat_latent.shape == (batch_size, window_size, 2 * r)

            yhat_out = self.decoder(yhat_latent) + c
            assert yhat_out.shape == (batch_size, window_size, 2 * d)
        else:
            A_dmd = torch.eye(r, r, device=x.device, dtype=x.dtype)
            yhat_latent = x_latent
            yhat_out = x_out

        return DMDAutoencoderResult(
            A_dmd=A_dmd,
            x_out=torch.concat([x_out, x_in[:, :, -2:]], dim=-1),
            y_latent=y_latent,
            yhat_latent=yhat_latent,
            yhat_out=torch.concat([yhat_out, x_in[:, :, -2:]], dim=-1),
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


class DMDLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma_rec: float = 1.0,
        gamma_lin: float = 1.0,
        gamma_pre: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.mse = nn.MSELoss()
        self.gamma_rec = gamma_rec / scale
        self.gamma_lin = gamma_lin / scale
        self.gamma_pre = gamma_pre / scale

    @property
    def dim(self) -> int:
        return 4

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        *,
        y_latent: torch.Tensor,
        yhat_latent: torch.Tensor,
        yhat_out: torch.Tensor,
    ) -> DMDLossResult:
        # reconstruction loss
        loss_rec = self.mse(x_in, x_out)
        loss_rec = self.gamma_rec * loss_rec

        # latent space linearity loss
        loss_lin = self.mse(y_latent[:, 1:, :], yhat_latent[:, :-1, :])
        loss_lin = self.gamma_lin * loss_lin

        # linear reconstruction loss
        loss_pre = self.mse(x_in[:, 1:, :], yhat_out[:, :-1, :])
        loss_pre = self.gamma_pre * loss_pre

        return DMDLossResult(
            total=loss_rec + loss_lin + loss_pre,
            loss_rec=loss_rec,
            loss_lin=loss_lin,
            loss_pre=loss_pre,
        )


# }}}


# {{{ torch_train


def torch_train(
    model: DMDAutoencoder,
    criterion: DMDLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X_train: torch.Tensor,
    X_validate: torch.Tensor,
    *,
    max_norm: float = 100.0,
    window_size: int = 32,
    nwindows: int = 16,
    epochs: int = 50,
    batch_size: int = 2048,
    dmd_warmup_epochs: int = 0,
    patience_early_stop: int = 0,
    callback: Callable[[int, DMDAutoencoder], None] | None = None,
) -> tuple[DMDAutoencoder, torch.optim.Optimizer, torch.Tensor]:
    from nneuroutil.data import SlidingWindowDataset

    device = X_train.device
    compiled_model = torch.compile(model)

    train_dataset = SlidingWindowDataset(
        X_train, window_size=window_size, nwindows=nwindows
    )
    validate_dataset = SlidingWindowDataset(
        X_validate, window_size=window_size, nwindows=nwindows
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

        for i, (batch_x,) in enumerate(loader):
            optimizer.zero_grad()

            predictions = compiled_model(batch_x)
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
                    eigmax = eigenvalues[eigenvalues.real.argmax()].item()

                log.info(
                    r"    \[batch %03d] lambda_max %.5e+%.5e loss %.5e grad %.5e",
                    i,
                    eigmax.real,
                    eigmax.imag,
                    loss.total.item(),
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

            for (batch_x,) in validate_loader:
                predictions = compiled_model(batch_x)
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
            "[%03d/%03d] train %.5e validate %.5e (%s)",
            epoch + 1,
            epochs,
            loss_train[0],
            loss_validate[0],
            timer,
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
        model.load_state_dict(best_state)
        log.info(
            "Restored best model from epoch %03d (loss %.5e)", best_epoch, best_loss
        )

    return model, optimizer, losses


# }}}


# {{{ torch_predict


def build_latent_dmd(
    model: DMDAutoencoder,
    Z: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Build the DMD operator in the latent space.

    This uses the encoder from *model* to squeeze Z into the latent space and then
    performs a DMD reconstruction on complex-valued latent variables. The resulting
    complex operator is returned.
    """

    r = model.latent_size
    n = Z.shape[0] * (Z.shape[1] - 1)

    complex_dtype = torch.complex64 if Z.dtype == torch.float32 else torch.complex128

    Xs = torch.empty((n, r), dtype=complex_dtype, device=Z.device)
    Ys = torch.empty((n, r), dtype=complex_dtype, device=Z.device)

    with torch.no_grad():
        model.eval()

        offset = 0
        for start in range(0, Z.shape[0], block_size):
            x_latent = model.encoder(Z[start : start + block_size])
            # x_latent shape: (batch, window, 2*r) interleaved
            z_latent = view_as_complex(x_latent)

            size = x_latent.shape[0] * (x_latent.shape[1] - 1)
            Xs[offset : offset + size] = z_latent[:, :-1, :].reshape(-1, r)
            Ys[offset : offset + size] = z_latent[:, 1:, :].reshape(-1, r)

            offset += size

        return Ys.T @ torch.linalg.pinv(Xs.T, rtol=model.eps)


def torch_predict(
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
    device: Any = None,
) -> torch.Tensor:
    import array_api_compat

    from commonlib import generate_grid

    d = int(model.input_size)

    dtype = model.dtype
    assert isinstance(dtype, torch.dtype)
    assert dtype.is_floating_point

    with torch.no_grad():
        z = torch.zeros((resolution, resolution, 2 * d), dtype=dtype, device=device)
        c = generate_grid(
            *bbox,
            nx=resolution,
            ny=resolution,
            dtype=dtype.to_complex(),
            xp=array_api_compat.array_namespace(A_dmd),
            device=device,
        )
        c = torch.view_as_real(cast("torch.Tensor", c))
        c = c.repeat(1, 1, d)

        for _ in range(1, maxit):
            g = model.encoder(z)
            g_complex = view_as_complex(g) @ A_dmd.T
            g = interleave(g_complex.real, g_complex.imag, axis=-1)
            z = model.decoder(g) + c

    return view_as_complex(z)


# }}}


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int = 5,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    epochs: int = 100,
    batch_size: int = 512,
    window_size: int = 32,
    nwindows: int = 16,
    dmd_warmup_epochs: int = 0,
    early_stop_patience: int = 0,
    overwrite: bool = False,
) -> int:
    from commonlib import determine_escape_radius, generate_grid

    device = torch.get_default_device()
    log.info("Device: %s", device)

    # {{{ vallidate inputs

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    try:
        data = np.load(filename)
    except Exception:
        log.error("Failed to load data from file: '%s'.", filename)
        return 1

    if datakey not in data:
        log.error("File does not contain '%s' dataset: '%s'.", datakey, filename)
        return 1

    matrices = data[datakey]
    if matrices.ndim != 3:
        log.error("Expected a 3 dimensional array: %s.", matrices.shape)
        return 1

    # NOTE: we don't really need complex128 for this example, so save some memory
    mat = torch.from_numpy(matrices[dataidx]).to(torch.complex64).to(device)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-ae-dmd-quad.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{device.type}"
    )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ plot predict

    def plot_predict(epoch: int, m: DMDAutoencoder) -> None:
        A_dmd = build_latent_dmd(m, x_train[:, :, :-2])
        Zpredict = torch_predict(
            m,
            A_dmd,
            bbox=bbox,
            resolution=2 * resolution,
            maxit=maxit,
        )
        mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

        with figure(
            basename.with_stem(f"{basename.stem}-predict-{epoch:04d}"),
            overwrite=overwrite,
        ) as fig:
            ax = fig.gca()
            ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ train

    from torch import optim

    d = mat.shape[0]
    model = DMDAutoencoder(
        d,
        encoder_layer_sizes=(d, d // 2),
        decoder_layer_sizes=(d // 2, d),
        device=device,
        dtype=mat.dtype.to_real(),
    )
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.3,
        min_lr=1.0e-5,
        cooldown=2,
    )

    log.info("Network:\n%s", model)

    weightsfile = outfile.with_suffix(".pkl")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]
        loss_test = checkpoint.get("loss_test", 0.0)
        zmask = checkpoint["zmask"]
        A_dmd = checkpoint["a_dmd"]

        model.eval()
    else:
        from commonlib import generate_timeseries_dataset
        from nneuroutil.data import SlidingWindowDataset, random_split

        # {{{ generate dataset

        x, _, zmask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            generate_timeseries_dataset(
                mat,
                bbox=bbox,
                resolution=resolution,
                # NOTE: we add 1 so that x.shape == (n, maxit, 2d)
                maxit=maxit + 1,
                escape_radius=escape_radius,
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
        c = c[zmask]
        c = torch.view_as_real(c)
        c = c.unsqueeze(1).expand(-1, x.shape[1], -1)

        x = torch.concat([x, c], dim=-1)
        (x_train,), (x_validate,), (x_test,) = random_split((x,), (0.8, 0.1, 0.1))

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # }}}

        # {{{ train

        target_scale = torch.mean(x_train.conj() * x_train).real.item()
        criterion = DMDLoss(scale=target_scale)

        _, optimizer, losses = torch_train(
            model,
            criterion,
            optimizer,
            scheduler,
            x_train,
            x_validate,
            epochs=epochs,
            batch_size=batch_size,
            window_size=window_size,
            nwindows=nwindows,
            dmd_warmup_epochs=dmd_warmup_epochs,
            patience_early_stop=early_stop_patience,
            callback=plot_predict,
        )
        A_dmd = build_latent_dmd(model, x_train)

        model.train()
        with torch.no_grad():
            test_dataset = SlidingWindowDataset(
                x_test, window_size=window_size, nwindows=nwindows
            )
            test_loader = DataLoader(
                test_dataset,  # ty: ignore[invalid-argument-type]
                batch_size=batch_size,
            )

            loss_test_epoch = torch.zeros(criterion.dim, device=device)
            num_test_batches = 0

            for (batch_x,) in test_loader:
                predictions = model(batch_x)
                loss = criterion(
                    batch_x,
                    x_out=predictions.x_out,
                    y_latent=predictions.y_latent,
                    yhat_latent=predictions.yhat_latent,
                    yhat_out=predictions.yhat_out,
                )

                loss_test_epoch += loss.tensor()
                num_test_batches += 1

            loss_test = (loss_test_epoch / num_test_batches)[0].item()
            log.info("[%03d/%03d] test %.5e", epochs, epochs, loss_test)

        # }}}

        # {{{ checkpoint

        num_epochs_trained = len(losses)
        torch.save(
            {
                # model
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "num_epochs_trained": num_epochs_trained,
                "losses": losses,
                "loss_test": loss_test,
                "zmask": zmask,
                "a_dmd": A_dmd,
                # parameters
                "inputfile": str(filename),
                "dataidx": dataidx,
                "bbox": bbox,
                "escape_radius": escape_radius,
                "resolution": resolution,
                "maxit": maxit,
                "dmd_warmup_epochs": dmd_warmup_epochs,
                "early_stop_patience": early_stop_patience,
            },
            weightsfile,
        )

        # }}}

    # }}}

    # {{{ plot fractal

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(zmask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ plot loss

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        losses = losses.cpu().numpy()
        ax.semilogy(losses[:, 0, 0], label="Train")
        ax.semilogy(losses[:, 1, 0], label="Validation")
        ax.axhline(loss_test, color="k", ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    # }}}

    # {{{ plot prediction

    Zpredict = torch_predict(
        model,
        A_dmd,
        bbox=bbox,
        resolution=2 * resolution,
        maxit=maxit,
    )
    mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

    with figure(
        basename.with_stem(f"{basename.stem}-predict"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    return 0


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
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "--dataidx",
        type=int,
        default=5,
        help="Index of the matrix to reconstruct",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Resolution (in the c complex plane) of the images",
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default deviced used by the framework",
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
        "--dmd-warmup-epochs",
        type=int,
        default=0,
        help="Train this many epochs without DMD before enabling it (0 disables)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training loss does not improve for this many epochs (0 disables)",
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

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        raise SystemExit(
            main(
                args.filename,
                dataidx=args.dataidx,
                resolution=args.resolution,
                dmd_warmup_epochs=args.dmd_warmup_epochs,
                early_stop_patience=args.early_stop_patience,
                outfile=args.outfile,
                overwrite=args.force,
            )
        )

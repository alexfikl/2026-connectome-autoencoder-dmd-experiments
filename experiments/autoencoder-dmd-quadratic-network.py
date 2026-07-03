# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from nneuroutil.helpers import FuzzyChoiceAction, TicTocTimer, module_logger
from nneuroutil.torch_extras import available_device_names, torch_default_device
from nneuroutil.typing import Array3D

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a standard Autoencoder neural network.

Example:

    > {SCRIPT_PATH.name} dataset.npz
"""


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
        kaiming_uniform_,
    )

    layers = []
    for (n_in, n_out), nonlinearity in zip(
        pairwise(sizes), nonlinearities, strict=True
    ):
        # build nonlinearity
        if nonlinearity == "quadratic":
            activation = ComplexQuadratic()
        elif nonlinearity == "blended_quadratic":
            activation = ComplexBlendedQuadratic(0.5 if alpha is None else alpha)
        else:
            raise ValueError(f"unsupported nonlinearity: {nonlinearity!r}")

        # build linear layer
        linear = ComplexLinear(n_in, n_out, bias=False, device=device, dtype=dtype)
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

    y_out: torch.Tensor
    """Output of the autoencoder."""
    x_latent: torch.Tensor
    """Output of the encoder."""
    xhat_latent: torch.Tensor
    """DMD reconstructed latent vector :attr:`x_latent`."""
    yhat_out: torch.Tensor
    """Output of the decoder applied to :atr:`xhat_latent`."""


class DMDAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        encoder_layer_sizes: int | tuple[int, ...] | None = None,
        decoder_layer_sizes: int | tuple[int, ...] | None = None,
        dmd_enabled: bool = False,
        alpha: float | None = None,
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

        # {{{ construct autoencoder

        # encoder
        nlayers = len(encoder_layer_sizes)
        encoder_layers = make_layers(
            (self.input_size, *encoder_layer_sizes),
            (*(("blended_quadratic",) * (nlayers - 1)), "quadratic"),
            alpha=alpha,
            dtype=dtype,
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        nlayers = len(decoder_layer_sizes)
        decoder_layers = make_layers(
            (self.latent_size, *decoder_layer_sizes),
            ("quadratic", *(("blended_quadratic",) * (nlayers - 1))),
            alpha=alpha,
            dtype=dtype,
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # }}}

    def forward(self, x: torch.Tensor) -> DMDAutoencoderResult:
        # NOTE: x has a shape of (batch_size, window_size, 2 d) and we want to
        # do DMD on it, so we go
        # 1. X = x[:, :-1, :].reshape(-1, d)
        #    Y = x[:, 1:, :].reshape(-1, d)
        # 2. build the DMD operator for this batch with (X, Y)

        # go to latent space
        x_latent = self.encoder(x)
        y_out = self.decoder(x_latent)

        d = x_latent.shape[-1]
        assert d == 2 * self.latent_size

        # shape of the latent space
        if self.training and self.dmd_enabled:
            # construct DMD operator in latent space
            X = x_latent[:, :-1, :].reshape(-1, d)
            Y = x_latent[:, 1:, :].reshape(-1, d)

            A_dmd = Y.T @ torch.linalg.pinv(X.T, rtol=1.0e-6)

            # predict with DMD in latent space (single-step)
            xhat_latent = x_latent @ A_dmd.T
            yhat_out = self.decoder(xhat_latent)
        else:
            A_dmd = torch.eye(d, d, device=x.device, dtype=x.dtype)
            xhat_latent = x_latent
            yhat_out = y_out

        return DMDAutoencoderResult(
            A_dmd=A_dmd,
            y_out=y_out,
            x_latent=x_latent,
            xhat_latent=xhat_latent,
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
        y_out: torch.Tensor,
        *,
        x_latent: torch.Tensor,
        xhat_latent: torch.Tensor,
        yhat_out: torch.Tensor,
    ) -> DMDLossResult:
        # reconstruction loss
        loss_rec = self.mse(x_in, y_out)
        loss_rec = self.gamma_rec * loss_rec

        # latent space linearity loss
        loss_lin = self.mse(x_latent[:, 1:, :], xhat_latent[:, :-1, :])
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


def torch_train_impl(
    model: DMDAutoencoder,
    criterion: DMDLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X: torch.Tensor,
    *,
    max_norm: float = 1.0,
    window_size: int = 32,
    nwindows: int = 16,
    epochs: int = 50,
    batch_size: int = 2048,
) -> tuple[DMDAutoencoder, torch.optim.Optimizer, torch.Tensor]:
    import torch

    device = X.device
    if epochs == 0:
        return model, optimizer, torch.zeros((epochs, criterion.dim), device=device)

    from nneuroutil.torch_extras import SlidingWindowDataset

    # NOTE: the compiled model seems to be slower? That's annoying..
    # compiled_model = torch.compile(model, dynamic=False)
    compiled_model = model

    dataset = SlidingWindowDataset(X, window_size=window_size, nwindows=nwindows)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # NOTE: this is needed otherwise the DataLoader seems to want to use the
        # 'cpu' device even when there is another default device..
        generator=torch.Generator(device),
    )

    num_per_batches = X.shape[0] * nwindows // batch_size
    losses = torch.empty((epochs, criterion.dim), device=device)

    timer = TicTocTimer()

    for epoch in range(epochs):
        log.info("Epoch %03d/%03d", epoch + 1, epochs)
        loss_epoch = torch.zeros(criterion.dim, device=device)

        timer.tic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, (batch_x,) in enumerate(loader):
            optimizer.zero_grad()

            # compute loss function
            predictions = compiled_model(batch_x)
            loss = criterion(
                batch_x,
                y_out=predictions.y_out,
                x_latent=predictions.x_latent,
                xhat_latent=predictions.xhat_latent,
                yhat_out=predictions.yhat_out,
            )

            # compute gradient
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_norm
            )

            # update weights
            optimizer.step()

            # update loss
            loss_epoch += loss.tensor().detach()

            with torch.no_grad():
                eigenvalues = torch.linalg.eigvals(predictions.A_dmd.cpu())
                eigmax = eigenvalues[eigenvalues.real.argmax()].item()

            if i in {0, len(loader) - 1}:
                log.info(
                    r"    \[batch %03d] lambda_max %.5e+%.5e loss %.5e grad %.5e",
                    i,
                    eigmax.real,
                    eigmax.imag,
                    loss.total.item(),
                    grad_norm,
                )

        losses[epoch] = loss_epoch / num_per_batches
        scheduler.step(losses[epoch, 0])  # ty: ignore[invalid-argument-type]

        timer.toc()
        log.info("    Loss %.5e (%s)", losses[epoch, 0], timer)

    return compiled_model, optimizer, losses


def torch_train(
    model: DMDAutoencoder,
    criterion: DMDLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X: torch.Tensor,
    *,
    max_norm: float = 100.0,
    window_size: int = 32,
    nwindows: int = 16,
    epochs: int = 50,
    batch_size: int = 2048,
    dmd_enabled: bool = True,
) -> tuple[DMDAutoencoder, torch.optim.Optimizer, torch.Tensor]:
    # NOTE: we're training in two rounds:
    #   1. doing 'n_bare_epochs' epochs is just AE
    #   2. doing the rest with DMD
    # Reasoning: adding DMD is slow and can make the training go in weird directions.
    # Having a reasonable start for the AE weights already in there works nicely.

    # n_bare_epochs = epochs // 3
    n_bare_epochs = 0
    model, optimizer, losses = torch_train_impl(
        model,
        criterion,
        optimizer,
        scheduler,
        X,
        max_norm=float("inf"),
        window_size=window_size,
        nwindows=nwindows,
        epochs=n_bare_epochs,
        batch_size=batch_size,
    )

    if dmd_enabled:
        model.dmd_enabled = True

        model, optimizer, losses_dmd = torch_train_impl(
            model,
            criterion,
            optimizer,
            scheduler,
            X,
            max_norm=max_norm,
            window_size=window_size,
            nwindows=nwindows,
            epochs=epochs - n_bare_epochs,
            batch_size=batch_size,
        )
        losses = torch.concat([losses, losses_dmd])

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
    performs a standard DMD reconstruction. The resulting operator is returned.
    """

    d = 2 * model.latent_size
    n = Z.shape[0] * (Z.shape[1] - 1)

    Xs = torch.empty((n, d), dtype=Z.dtype, device=Z.device)
    Ys = torch.empty((n, d), dtype=Z.dtype, device=Z.device)

    with torch.no_grad():
        model.eval()

        offset = 0
        for start in range(0, Z.shape[0], block_size):
            x_latent = model.encoder(Z[start : start + block_size])

            size = x_latent.shape[0] * (x_latent.shape[1] - 1)
            Xs[offset : offset + size] = x_latent[:, :-1, :].reshape(-1, d)
            Ys[offset : offset + size] = x_latent[:, 1:, :].reshape(-1, d)

            offset += n

        return Ys.T @ torch.linalg.pinv(Xs.T, rtol=1.0e-6)


def torch_predict(
    model: DMDAutoencoder,
    A_dmd: torch.Tensor,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
    device: Any = None,
) -> Array3D:
    import torch

    d = int(model.input_size)
    dtype = model.dtype
    assert isinstance(dtype, torch.dtype)
    assert dtype.is_floating_point

    with torch.no_grad():
        # Koopman DMD prediction: z_{k+1} = decoder(A_dmd @ encoder(z_k))
        z = torch.zeros((resolution, resolution, 2 * d), dtype=dtype, device=device)
        for _ in range(1, maxit):
            g = model.encoder(z)
            g = g @ A_dmd.T
            z = model.decoder(g)

    return torch.complex(z[:, :, :d], z[:, :, d:])


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
    overwrite: bool = False,
) -> int:
    from commonlib import determine_escape_radius

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

    # {{{ train

    from torch import optim

    d = mat.shape[0]
    model = DMDAutoencoder(
        d,
        encoder_layer_sizes=(d, d // 2),
        decoder_layer_sizes=(d // 2, d),
    )
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-3)

    log.info("Network:\n%s", model)

    weightsfile = outfile.with_suffix(".pkl")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]
        A_dmd = checkpoint["A_dmd"]
        Zmask = checkpoint["Zmask"]

        compiled_model = cast("DMDAutoencoder", torch.compile(model))
    else:
        from commonlib import generate_training_dataset

        # {{{ generate dataset

        X, _, Zlast = generate_training_dataset(
            mat,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
            in_dmd=True,
        )
        Zmask = torch.sum(Zlast.real**2 + Zlast.imag**2, dim=-1) < escape_radius**2

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # }}}

        # {{{ train

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.3,
            min_lr=1.0e-5,
            cooldown=2,
        )

        target_scale = torch.mean(X.conj() * X).real.item()
        criterion = DMDLoss(scale=target_scale)

        compiled_model, optimizer, losses = torch_train(
            model,
            criterion,
            optimizer,
            scheduler,
            X,
            epochs=epochs,
            batch_size=batch_size,
        )
        A_dmd = build_latent_dmd(model, X)

        # }}}

        # {{{ checkpoint

        torch.save(
            {
                # model
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "losses": losses,
                "A_dmd": A_dmd,
                "Zmask": Zmask,
                "Zlast": Zlast,
                # parameters
                "inputfile": str(filename),
                "dataidx": dataidx,
                "bbox": bbox,
                "escape_radius": escape_radius,
                "resolution": resolution,
                "maxit": maxit,
            },
            weightsfile,
        )

        # }}}

    # }}}

    from nneuroutil.visualization import figure

    # {{{ plot fractal

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(Zmask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ plot loss

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        data = losses.cpu().numpy()
        ax.semilogy(data[:, 0], label="Total")
        ax.semilogy(data[:, 1], label="Reconstruction")
        ax.semilogy(data[:, 2], label="Linearity")
        ax.semilogy(data[:, 3], label="Prediction")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    # }}}

    # {{{ plot prediction

    Zpredict = torch_predict(
        compiled_model,
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
                overwrite=args.force,
            )
        )

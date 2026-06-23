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
from nneuroutil.torch_extras import available_device_names
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
        alpha: float = 0.1,
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
        from nneuroutil.dmd import build_dmd

        # NOTE: x has a shape of (batch_size, window_size, 2 d) and we want to
        # do DMD on it, so we go
        # 1. X = x[:, :-1, :].reshape(-1, d).T
        #    Y = x[:, 1:, :].reshape(-1, d).T
        # 2. build the DMD operator for this batch with (X, Y)

        # go to latent space
        x_latent = self.encoder(x)

        # construct DMD operator in latent space
        d = x.shape[-1]
        X = x_latent[:, :-1, :].reshape(-1, d)
        Y = x_latent[:, 1:, :].reshape(-1, d)
        dmd = build_dmd(X, Y)

        # predict with DMD in latent space
        # NOTE: x_latent is not transposed so we write these out explicitly
        xhat_latent = torch.einsum("ij,...j->...i", dmd.Vh, x_latent)
        xhat_latent = torch.einsum("ij,...j->...i", dmd.Ahat, xhat_latent)
        xhat_latent = torch.einsum("ji,...j->...i", torch.conj(dmd.Vh), xhat_latent)

        return DMDAutoencoderResult(
            y_out=self.decoder(x_latent),
            x_latent=x_latent,
            xhat_latent=xhat_latent,
            yhat_out=self.decoder(xhat_latent),
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

        # latent space linearity loss
        loss_lin = self.mse(x_latent, xhat_latent)

        # linear reconstruction loss
        loss_pre = self.mse(y_out, yhat_out)

        total = (
            self.gamma_rec * loss_rec
            + self.gamma_lin * loss_lin
            + self.gamma_pre * loss_pre
        )

        return DMDLossResult(
            total=total,
            loss_rec=loss_rec,
            loss_lin=loss_lin,
            loss_pre=loss_pre,
        )


# }}}


# {{{ torch_train


def torch_train(
    model: nn.Module,
    criterion: DMDLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    window_size: int = 32,
    epochs: int = 50,
    batch_size: int = 2048,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.Tensor]:
    import torch

    timer = TicTocTimer()

    device = X.device
    # compiled_model = torch.compile(model)
    compiled_model = model

    from nneuroutil.torch_extras import SlidingWindowDataset

    dataset = SlidingWindowDataset(X, Y, window_size=window_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # NOTE: this is needed otherwise the DataLoader seems to want to use the
        # 'cpu' device even when there is another default device..
        generator=torch.Generator(device),
    )

    num_batches = X.shape[0] // batch_size
    losses = torch.empty((epochs, criterion.dim), device=device)

    for epoch in range(epochs):
        loss_epoch = torch.zeros(criterion.dim, device=device)
        timer.tic()

        for i, (batch_x, batch_y) in enumerate(loader):
            assert i >= 0
            optimizer.zero_grad()

            # compute loss function
            predictions = compiled_model(batch_x)
            loss = criterion(
                batch_y,
                y_out=predictions.y_out,
                x_latent=predictions.x_latent,
                xhat_latent=predictions.xhat_latent,
                yhat_out=predictions.yhat_out,
            )

            # compute gradient
            loss.backward()

            # update weights
            optimizer.step()

            loss_epoch += loss.tensor().detach()

        losses[epoch] = loss_epoch / num_batches
        scheduler.step(losses[epoch, 0])  # ty: ignore[invalid-argument-type]

        timer.toc()
        log.info(
            "[%03d/%03d] loss %.5e (%s)", epoch + 1, epochs, losses[epoch, 0], timer
        )

    return model, optimizer, losses


# }}}


# {{{ torch_predict


def torch_predict(
    model: nn.Module,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
    device: Any = None,
) -> Array3D:
    import array_api_compat
    import torch

    d = int(model.input_size)  # ty: ignore[invalid-argument-type]
    dtype = model.dtype
    assert isinstance(dtype, torch.dtype)
    assert dtype.is_floating_point

    from commonlib import generate_grid

    with torch.no_grad():
        z = torch.zeros((resolution, resolution, 2 * d), dtype=dtype, device=device)
        c = generate_grid(
            *bbox,
            nx=resolution,
            ny=resolution,
            dtype=dtype.to_complex(),
            xp=array_api_compat.array_namespace(z),
            device=device,
        )
        c = c[:, :, None]

        for _ in range(1, maxit):
            Az = model(z)
            z[:, :, :d] = Az[:, :, :d] + c.real
            z[:, :, d:] = Az[:, :, d:] + c.imag

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
    batch_size: int = 2048,
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
        encoder_layer_sizes=(d, d),
        decoder_layer_sizes=(d, d),
    )
    optimizer = optim.Adam(model.parameters(), lr=1.0e-3)

    log.info("Network:\n%s", model)

    weightsfile = outfile.with_suffix(".pkl")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]
        Zmask = checkpoint["Zmask"]

        compiled_model = cast("DMDAutoencoder", torch.compile(model))
    else:
        from commonlib import generate_training_dataset

        # {{{ generate dataset

        X, Y, Zlast = generate_training_dataset(
            mat,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
            in_dmd=True,
        )
        Zmask = torch.sum(Zlast.real**2 + Zlast.imag**2, dim=-1) < escape_radius**2

        # }}}

        # {{{ train

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.3,
            min_lr=1.0e-5,
            cooldown=2,
        )

        target_scale = torch.mean(Y.conj() * Y).real.item()
        criterion = DMDLoss(scale=target_scale)

        compiled_model, optimizer, losses = torch_train(
            model,
            criterion,
            optimizer,
            scheduler,
            X,
            Y,
            epochs=epochs,
            batch_size=batch_size,
        )

        # }}}

        # {{{ checkpoint

        torch.save(
            {
                # model
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "losses": losses,
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

        ax.semilogy(losses.cpu().numpy())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    # }}}

    # {{{ plot prediction

    Zpredict = torch_predict(
        compiled_model, bbox=bbox, resolution=2 * resolution, maxit=maxit
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

    with torch.device(args.device):
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

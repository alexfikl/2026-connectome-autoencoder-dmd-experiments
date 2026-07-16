# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from nneuroutil.helpers import FuzzyChoiceAction, module_logger
from nneuroutil.torch_extras import available_device_names, torch_default_device
from nneuroutil.visualization import figure

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains an autoencoder-like neural network with quadratic activation functions.

Example:

    > {SCRIPT_PATH.name} --device cuda dataset.npz
"""


# {{{ model


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        encoder_layer_sizes: int | tuple[int, ...] | None = None,
        decoder_layer_sizes: int | tuple[int, ...] | None = None,
        alpha: float = 0.5,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        if encoder_layer_sizes is None:
            encoder_layer_sizes = (input_size,)

        if isinstance(encoder_layer_sizes, int):
            encoder_layer_sizes = (encoder_layer_sizes,)

        if decoder_layer_sizes is None:
            decoder_layer_sizes = (encoder_layer_sizes[-1],)

        if isinstance(decoder_layer_sizes, int):
            decoder_layer_sizes = (decoder_layer_sizes,)

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

        from itertools import pairwise

        from nneuroutil.torch_extras import (
            Bias,
            ComplexBlendedQuadratic,
            ComplexLinear,
            ComplexQuadratic,
            kaiming_uniform_,
        )

        # encoder
        layer_sizes = (self.input_size, *encoder_layer_sizes)
        encoder_layers = []
        for i, (n_in, n_out) in enumerate(pairwise(layer_sizes)):
            is_last = i == len(encoder_layer_sizes) - 1

            # build nonlinearity
            layer_nonlinearity = "quadratic" if is_last else "blended_quadratic"
            param = None if is_last else alpha

            activation = (
                ComplexBlendedQuadratic(alpha, interleaved=True)
                if i != len(encoder_layer_sizes) - 1
                else ComplexQuadratic(interleaved=True)
            )

            # build linear layer
            linear = ComplexLinear(
                n_in, n_out, bias=False, interleaved=True, device=device, dtype=dtype
            )
            kaiming_uniform_(
                linear.weight,
                nonlinearity=layer_nonlinearity,
                param=param,
            )

            encoder_layers.append(linear)
            encoder_layers.append(activation)
            encoder_layers.append(Bias(2 * n_out, device=device, dtype=dtype))

        # decoder
        layer_sizes = (self.latent_size, *decoder_layer_sizes)
        decoder_layers = []
        for n_in, n_out in pairwise(layer_sizes):
            linear = ComplexLinear(
                n_in, n_out, interleaved=True, device=device, dtype=dtype
            )
            kaiming_uniform_(linear.weight, nonlinearity="linear")

            decoder_layers.append(linear)

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # }}}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


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
    epochs: int = 50,
    batch_size: int = 4096,
    early_stop_patience: int = 0,
    overwrite: bool = False,
) -> int:
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
        from commonlib import determine_escape_radius

        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-ae-quad.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{device.type}"
    )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ plot predict

    def plot_predict(epoch: int, m: nn.Module) -> None:
        from commonlib import torch_predict

        Zpredict = torch_predict(
            m,
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
    model = Autoencoder(
        d,
        encoder_layer_sizes=(d, d // 2),
        decoder_layer_sizes=(d,),
        device=device,
        dtype=mat.dtype.to_real(),
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-3)
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

        model.eval()
        plot_predict(epochs, model)
    else:
        from commonlib import (
            generate_timeseries_dataset,
            make_timeseries_datasets,
            torch_train,
        )

        # {{{ generate dataset

        x, y, zmask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            generate_timeseries_dataset(
                mat,
                bbox=bbox,
                resolution=resolution,
                maxit=maxit,
                escape_radius=escape_radius,
            ),
        )
        xy_train, xy_validate, xy_test = cast(
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
            make_timeseries_datasets(x, y, (0.8, 0.1, 0.1)),
        )

        # }}}

        # {{{ train

        _, y_train = xy_train
        target_scale = torch.mean(y_train.conj() * y_train).real.item()

        _, optimizer, losses = torch_train(
            model,
            criterion,
            optimizer,
            scheduler,
            TensorDataset(*xy_train),
            TensorDataset(*xy_validate),
            epochs=epochs,
            batch_size=batch_size,
            target_scale=target_scale,
            patience_early_stop=early_stop_patience,
            callback=plot_predict,
        )

        model.eval()
        with torch.no_grad():
            x_test, y_test = xy_test
            predictions = model(x_test)
            loss_test = criterion(predictions, y_test) / target_scale

            loss_test = loss_test.item()
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
                # parameters
                "inputfile": str(filename),
                "dataidx": dataidx,
                "bbox": bbox,
                "escape_radius": escape_radius,
                "resolution": resolution,
                "maxit": maxit,
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
                early_stop_patience=args.early_stop_patience,
                outfile=args.outfile,
                overwrite=args.force,
            )
        )

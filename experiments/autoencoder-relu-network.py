# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from nneuroutil.helpers import TicTocTimer, module_logger, tictoc

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a standard Autoencoder neural network.

Example:

    > {SCRIPT_PATH.name} dataset.npz
"""


# {{{ model


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        alpha: float = 0.1,
        use_complex_linear: bool = True,
        dtype: Any = None,
    ) -> None:
        if dtype is None:
            dtype = torch.float32

        super().__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = input_size
        self.dtype = dtype

        # {{{ construct autoencoder

        from nneuroutil.torch_extras import ComplexLinear

        if use_complex_linear:
            fac = 1
            cls: type[nn.Module] = ComplexLinear
        else:
            fac = 2
            cls: type[nn.Module] = nn.Linear

        self.encoder = nn.Sequential(
            cls(fac * self.input_size, fac * 256),
            nn.LeakyReLU(alpha),
            cls(fac * 256, fac * 128),
            nn.LeakyReLU(alpha),
            cls(fac * 128, fac * 64),
            nn.LeakyReLU(alpha),
            cls(fac * 64, fac * self.latent_size),
        )
        self.decoder = nn.Sequential(
            cls(fac * self.latent_size, fac * 64),
            nn.LeakyReLU(alpha),
            cls(fac * 64, fac * 128),
            nn.LeakyReLU(alpha),
            cls(fac * 128, fac * 256),
            nn.LeakyReLU(alpha),
            cls(fac * 256, fac * self.output_size),
        )

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
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 128,
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
        outfile = filename.parent / (
            f"{filename.stem}-ae-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{device.type}.npz"
        )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ generate dataset

    d = mat.shape[0]
    bbox = (-0.03, 0.03, -0.03, 0.03)
    log.info("Generating dataset on %s", bbox)

    with tictoc("generate_dataset"):
        from commonlib import generate_dataset

        Z, c = generate_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            escape_radius=escape_radius,
            maxit=maxit,
        )

    mask = torch.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, dim=-1) < escape_radius**2
    log.info("Points:")
    log.info("    Converged: %d", torch.sum(mask))
    log.info("    Max: %.5e (radius %.5e)", Z.abs().max(), escape_radius)
    log.info("    Mean: %.5e", Z.abs().mean())

    from nneuroutil.visualization import figure

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ build torch dataset

    from commonlib import iter_batches, split_dataset

    # Compute target scale for Normalized MSE (NMSE) loss to resolve vanishing gradients
    X, Y = split_dataset(Z, C=c, escape_radius=escape_radius, split_real_imaginary=True)
    target_scale = torch.mean(Y.conj() * Y).real
    log.info("Dataset: %s", X.shape)

    # }}}

    # {{{ train

    from torch import optim

    model = Autoencoder(d, d, use_complex_linear=False, dtype=X.dtype)
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
        checkpoint = torch.load(weightsfile)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]

        compiled_model = cast("Autoencoder", torch.compile(model))
    else:
        timer = TicTocTimer()

        epochs = 100
        batch_size = 4096
        num_batches = X.shape[0] // batch_size
        losses = torch.empty(epochs)

        compiled_model = cast("Autoencoder", torch.compile(model))

        for epoch in range(epochs):
            loss_epoch = torch.tensor(0.0)
            timer.tic()

            for batch_x, batch_y in iter_batches(X, Y, batch_size=batch_size):
                optimizer.zero_grad(set_to_none=True)

                # compute loss function
                predictions = compiled_model(batch_x)
                loss = criterion(predictions, batch_y) / target_scale

                # compute gradient
                loss.backward()

                # update weights
                optimizer.step()

                loss_epoch += loss.detach()

            losses[epoch] = loss_epoch.item() / num_batches
            scheduler.step(losses[epoch])

            timer.toc()
            log.info("[%03d/%03d] loss %.5e (%s)", epoch, epochs, loss_epoch, timer)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "losses": losses,
                "mask": mask,
            },
            weightsfile,
        )

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        ax.semilogy(losses.cpu().numpy())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    # }}}

    # {{{ forecast

    from commonlib import torch_predict

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
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="The default deviced used by the framework",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Resolution (in the c complex plane) of the images",
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

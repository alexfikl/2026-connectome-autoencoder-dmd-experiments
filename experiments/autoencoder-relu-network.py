# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-FileCopyrightText: 2026 Vlad Arambașa <vlad.arambasa98@e-uvt.ro>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from nneuroutil.helpers import FuzzyChoiceAction, module_logger
from nneuroutil.torch_extras import available_device_names, torch_default_device

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
        device: str | torch.device | None = None,
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
            cls(fac * self.input_size, fac * 256, device=device, dtype=dtype),
            nn.LeakyReLU(alpha),
            cls(fac * 256, fac * 128, device=device, dtype=dtype),
            nn.LeakyReLU(alpha),
            cls(fac * 128, fac * 64, device=device, dtype=dtype),
            nn.LeakyReLU(alpha),
            cls(fac * 64, fac * self.latent_size, device=device, dtype=dtype),
        )
        self.decoder = nn.Sequential(
            cls(fac * self.latent_size, fac * 64, device=device, dtype=dtype),
            nn.LeakyReLU(alpha),
            cls(fac * 64, fac * 128, device=device, dtype=dtype),
            nn.LeakyReLU(alpha),
            cls(fac * 128, fac * 256, device=device, dtype=dtype),
            nn.LeakyReLU(alpha),
            cls(fac * 256, fac * self.output_size, device=device, dtype=dtype),
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
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    epochs: int = 100,
    batch_size: int = 4096,
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
        outfile = filename.parent / f"{filename.stem}-ae-relu.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{device.type}"
    )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ train

    from torch import optim

    d = mat.shape[0]
    model = Autoencoder(
        d,
        d,
        use_complex_linear=False,
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
        Zmask = checkpoint["Zmask"]

        compiled_model = cast("Autoencoder", torch.compile(model))
    else:
        from commonlib import generate_training_dataset, torch_train

        # {{{ generate dataset

        X, Y, Zlast = generate_training_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            maxit=maxit,
            escape_radius=escape_radius,
        )
        Zmask = torch.sum(Zlast.real**2 + Zlast.imag**2, dim=-1) < escape_radius**2

        # }}}

        # {{{ train

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

    from commonlib import torch_predict
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

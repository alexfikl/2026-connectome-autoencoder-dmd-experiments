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

from nneuroutil.helpers import FuzzyChoiceAction, machine_info, module_logger
from nneuroutil.torch_extras import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)
from nneuroutil.visualization import figure

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains or runs inference with an autoencoder-like neural network with quadratic
activation functions.

Examples:

    > {SCRIPT_PATH.name} --device cuda train dataset.npz
    > {SCRIPT_PATH.name} --device cuda infer dataset-ae-quad.pkl
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


# {{{ helpers


def _plot_predict(
    model: Autoencoder,
    *,
    tag: str,
    basename: pathlib.Path,
    bbox: tuple[float, float, float, float],
    resolution: int,
    maxit: int,
    escape_radius: float,
    overwrite: bool,
) -> None:
    from commonlib import torch_predict

    Zpredict = torch_predict(
        model,
        bbox=bbox,
        resolution=2 * resolution,
        maxit=maxit,
    )
    mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

    with figure(
        basename.with_stem(f"{basename.stem}-predict-{tag}"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")


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
    loss_test: float,
    *,
    basename: pathlib.Path,
    overwrite: bool,
) -> None:
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


# {{{ train


def train(
    filename: pathlib.Path,
    *,
    weights: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    escape_radius: float | None = 100.0,
    resolution: int | None = None,
    maxit: int | None = None,
    epochs: int = 50,
    batch_size: int | None = None,
    split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    early_stop_patience: int = 0,
    overwrite: bool = False,
) -> int:
    from commonlib import (
        determine_escape_radius,
        generate_timeseries_dataset,
        split_timeseries_datasets,
        torch_train,
    )

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
        split = checkpoint["split"] if split is None else split
        seed = checkpoint["seed"] if seed is None else seed

        datakey = checkpoint["datakey"]
        escape_radius = checkpoint["escape_radius"]
    else:
        dataidx = 5 if dataidx is None else dataidx
        bbox = (-0.03, 0.03, -0.03, 0.03) if bbox is None else bbox
        resolution = 128 if resolution is None else resolution
        maxit = 128 if maxit is None else maxit
        batch_size = 4096 if batch_size is None else batch_size
        split = (0.8, 0.1, 0.1) if split is None else split
        if seed is None:
            seed = int(np.random.SeedSequence().entropy)  # ty: ignore[invalid-argument-type]

    # }}}

    # {{{ validate inputs

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

    mem.add_record("After data load")

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-ae-quad.npz"

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
    dtype = mat.dtype.to_real()

    # NOTE: when resuming, the stored architecture is used to rebuild the model
    model_config: dict[str, Any]
    if checkpoint is not None:
        model_config = checkpoint["model_config"]
    else:
        model_config = {
            "input_size": d,
            "encoder_layer_sizes": (d, d // 2),
            "decoder_layer_sizes": (d,),
            "alpha": 0.5,
        }

    model = Autoencoder(**model_config, device=device, dtype=dtype)
    criterion = nn.MSELoss()

    lr = 1.0e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.3, min_lr=1.0e-5, cooldown=2
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

    x, y, _, zmask = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
        split_timeseries_datasets(x, y, split, rng=np.random.default_rng(seed)),
    )

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    def plot_predict(epoch: int, m: nn.Module) -> None:
        # NOTE: plot 10 epochs and then only every 10th one to save time / memory
        if epoch > 10 and epoch % 10 != 0:
            return

        _plot_predict(
            m,  # ty: ignore[invalid-argument-type]
            tag=f"{epoch:04d}",
            basename=basename,
            bbox=bbox,
            resolution=resolution,
            maxit=maxit,
            escape_radius=escape_radius,
            overwrite=overwrite,
        )

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
        # callback=plot_predict,
    )

    mem.add_record("After training")

    model.eval()
    with torch.no_grad():
        x_test, y_test = xy_test
        predictions = model(x_test)
        loss_test = criterion(predictions, y_test) / target_scale

        loss_test = loss_test.cpu().item()
        log.info("[%03d/%03d] test %.5e", epochs, epochs, loss_test)

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
            "resolution": resolution,
            "maxit": maxit,
            "zmask": zmask,
            # hyperparameters
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "lr": lr,
            "batch_size": batch_size,
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
    _plot_loss(losses, loss_test, basename=basename, overwrite=overwrite)

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

    losses = checkpoint["losses"]
    loss_test = checkpoint["loss_test"]
    zmask = checkpoint["zmask"]

    if outfile is None:
        basename = weightsfile.with_suffix("")
    else:
        basename = outfile.with_suffix("")

    # {{{ load model

    model = Autoencoder(
        **checkpoint["model_config"], device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mem.add_record("After model load")

    # }}}

    # {{{ predict

    _plot_predict(
        model,
        tag="final",
        basename=basename,
        bbox=bbox,
        resolution=resolution,
        maxit=maxit,
        escape_radius=escape_radius,
        overwrite=overwrite,
    )

    mem.add_record("After prediction")

    # }}}

    # {{{ plot

    _plot_dataset(zmask, basename=basename, bbox=bbox, overwrite=overwrite)
    _plot_loss(losses, loss_test, basename=basename, overwrite=overwrite)

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
    subparsers = parser.add_subparsers(dest="command", required=True)

    # {{{ train

    subparser = subparsers.add_parser(
        "train",
        formatter_class=HelpFormatter,
        description="Train an autoencoder network on a dataset.",
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
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training loss does not improve for this many epochs (0 disables)",
    )

    # }}}

    # {{{ infer

    subparser = subparsers.add_parser(
        "infer",
        formatter_class=HelpFormatter,
        description="Run inference with a trained autoencoder network.",
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
                    early_stop_patience=args.early_stop_patience,
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

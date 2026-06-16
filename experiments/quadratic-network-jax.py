# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nneuroutil.flax_extras import available_device_names
from nneuroutil.helpers import FuzzyChoiceAction, module_logger, to_complex

log = module_logger("lib.quadratic_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a quadratic neural network: f(z) = (W z)^2 + b.

Example:

    > {SCRIPT_PATH.name} --device cpu dataset.npz
"""


# {{{ model


class MSELoss(nnx.Module):
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean((x - y) ** 2)


class QuadraticNetwork(nnx.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        *,
        dtype: Any = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)

        if output_size is None:
            output_size = input_size

        if dtype is None:
            dtype = np.dtype(np.float32)
        dtype = np.dtype(dtype)

        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype

        from nneuroutil.flax_extras import (
            Bias,
            ComplexLinear,
            ComplexQuadratic,
            quadratic_uniform,
        )

        self.model = nnx.Sequential(
            ComplexLinear(
                input_size,
                output_size,
                use_bias=False,
                kernel_init=quadratic_uniform(),
                dtype=dtype,
                rngs=rngs,
            ),
            ComplexQuadratic(),
            Bias(2 * output_size, dtype=dtype, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(x)


# }}}


# {{{ flax_train


def flax_train(
    model: QuadraticNetwork,
    criterion: nnx.Module,
    optimizer: nnx.Optimizer,
    X: jax.Array,
    Y: jax.Array,
    *,
    epochs: int = 50,
    batch_size: int = 4096,
) -> tuple[QuadraticNetwork, nnx.Optimizer, jax.Array]:
    from nneuroutil.helpers import TicTocTimer

    timer = TicTocTimer()

    num_batches = X.shape[0] // batch_size

    losses = np.empty(epochs)
    target_scale = jnp.mean(Y.conj() * Y).real

    @nnx.jit
    def train_step(
        model: QuadraticNetwork,
        optimizer: nnx.Optimizer,
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> jax.Array:
        def loss_fn(m: QuadraticNetwork) -> jax.Array:
            return criterion(m(batch_x), batch_y) / target_scale

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

        return loss

    from commonlib import iter_batches

    for epoch in range(epochs):
        loss_epoch = 0.0

        timer.tic()

        for batch_x, batch_y in iter_batches(X, Y, batch_size=batch_size):
            loss_val = train_step(model, optimizer, batch_x, batch_y)
            loss_epoch += loss_val

        losses[epoch] = loss_epoch / num_batches

        timer.toc()
        log.info("[%03d/%03d] loss %.5e (%s)", epoch + 1, epochs, loss_epoch, timer)

    return model, optimizer, jnp.array(losses)


# }}}


# {{{ flax_predict


def flax_predict(
    model: QuadraticNetwork,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
    device: Any = None,
) -> jax.Array:
    import array_api_compat

    from commonlib import generate_grid

    d = model.input_size
    dtype = model.dtype

    z = jnp.zeros((resolution, resolution, d), dtype=to_complex(dtype), device=device)
    c = generate_grid(
        *bbox,
        nx=resolution,
        ny=resolution,
        dtype=to_complex(dtype),
        xp=array_api_compat.array_namespace(z),
        device=device,
    )[:, :, None]

    for _ in range(1, maxit):
        Az = model(jnp.concat([z.real, z.imag], axis=-1))
        z = Az[:, :, :d] + 1j * Az[:, :, d:] + c

    return z


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
    overwrite: bool = False,
) -> int:
    from commonlib import determine_escape_radius

    (device,) = jax.devices(jax.default_backend())
    log.info("Device: %s", device)

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

    mat = jnp.array(matrices[dataidx], dtype=jnp.complex64).to_device(device)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-quad.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{jax.default_backend()}"
    )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ train

    import pickle

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, rngs=nnx.Rngs(0))
    criterion = MSELoss()
    optimizer = nnx.Optimizer(model, optax.adam(5.0e-3), wrt=nnx.Param)

    # FIXME: need a nice way to print this
    log.info("Network:")
    print(model)

    weightsfile = outfile.with_suffix(".pkl")
    if weightsfile.exists():
        with open(weightsfile, "rb") as f:
            checkpoint = pickle.load(f)

        nnx.update(
            model,
            jax.tree.map(
                lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
                checkpoint["model"],
            ),
        )
        losses = checkpoint["losses"]
        Zmask = checkpoint["Zmask"]
    else:
        from commonlib import generate_training_dataset

        # {{{ generate dataset

        X, Y, Zlast = generate_training_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            maxit=maxit,
            escape_radius=escape_radius,
        )
        Zmask = jnp.sum(Zlast.real**2 + Zlast.imag**2, axis=-1) < escape_radius**2

        # }}}

        # {{{ train

        model, _optimizer, losses = flax_train(
            model,
            criterion,
            optimizer,
            X,
            Y,
            epochs=epochs,
            batch_size=batch_size,
        )

        # }}}

        # {{{ checkpoint

        with open(weightsfile, "wb") as f:
            pickle.dump(
                {
                    # model
                    "model": nnx.state(model),
                    "epochs": epochs,
                    "losses": losses,
                    "Zmask": Zmask,
                    "Zlast": Zlast,
                    # parameters
                    "inputfile": filename,
                    "dataidx": dataidx,
                    "bbox": bbox,
                    "escape_radius": escape_radius,
                    "resolution": resolution,
                    "maxit": maxit,
                },
                f,
            )

        # }}}

    from nneuroutil.visualization import figure, make_colorbar_axes

    # }}}

    # {{{ plot fractal

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(np.array(Zmask), extent=bbox, cmap="binary")

    # }}}

    # {{{ plot loss

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        ax.semilogy(np.array(losses))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    # }}}

    # {{{ plot weight matrices

    # NOTE: the nn is trained to approximate the trajectory, not the matrix A,
    # even though the structure is the same (`(W x)^2 + b`). We look at the
    # linear layer to get a sense of what it managed to do here. However,
    #   1. we do not expect the matrices to match
    #   2. we do expect them to have similar singular values (i.e. the nn captured
    #      the general dynamics of the problem)

    # FIXME: this is pretty horrible. does it even work between versions?
    learned_A = model.model.layers[0].kernel.get_value()  # ty: ignore[unresolved-attribute]
    A = mat.astype(learned_A.dtype)

    with figure(
        basename.with_stem(f"{basename.stem}-weights"),
        nrows=2,
        ncols=2,
        overwrite=overwrite,
    ) as fig:
        ax11, ax12, ax21, ax22 = fig.axes

        # NOTE: some of the eig stuff does not work on GPU
        A_np = np.array(A).real
        learned_A_np = np.array(learned_A).real

        A_eigs = np.linalg.eigvals(A_np)
        learned_A_eigs = np.linalg.eigvals(learned_A_np)

        _, A_sigma, _ = np.linalg.svd(A_np)
        _, learned_A_sigma, _ = np.linalg.svd(learned_A_np)

        im = ax11.imshow(A_np, vmin=-1.0, vmax=1.0)
        ax11.set_title("$A$")
        fig.colorbar(im, cax=make_colorbar_axes(ax11))

        im = ax12.imshow(learned_A_np, vmin=-1.0, vmax=1.0)
        ax12.set_title("$A_{learned}$")
        fig.colorbar(im, cax=make_colorbar_axes(ax12))

        ax21.plot(A_eigs.real, A_eigs.imag, "o", label=r"$\lambda(A)$")
        ax21.plot(
            learned_A_eigs.real,
            learned_A_eigs.imag,
            "o",
            label=r"$\lambda(A_{learned})$",
        )
        ax21.set_xlabel(r"$\lambda_r$")
        ax21.set_ylabel(r"$\lambda_i$")
        ax21.set_title("EIG")
        ax21.legend()

        ax22.plot(A_sigma, "o", label=r"$\sigma(A)$")
        ax22.plot(learned_A_sigma, "o", label=r"$\sigma(A_{learned})$")
        ax22.set_ylabel(r"$\sigma$")
        ax22.set_title("SVD")
        ax22.legend()

    # }}}

    # {{{ plot prediction

    Zpredict = flax_predict(
        model,
        bbox=bbox,
        resolution=2 * resolution,
        maxit=maxit,
    )
    Zmask = jnp.sum(Zpredict.real**2 + Zpredict.imag**2, axis=-1) < escape_radius**2

    with figure(
        basename.with_stem(f"{basename.stem}-predict"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(np.array(Zmask), extent=bbox, cmap="binary")

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
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    from nneuroutil.flax_extras import jax_default_device, set_jax_config

    with jax_default_device(args.device):
        set_jax_config()

        raise SystemExit(
            main(
                args.filename,
                dataidx=args.dataidx,
                resolution=args.resolution,
                overwrite=args.force,
            )
        )

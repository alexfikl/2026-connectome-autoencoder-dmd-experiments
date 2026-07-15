# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from typing import Any, cast

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
    xy_train: tuple[jax.Array, jax.Array],
    xy_validate: tuple[jax.Array, jax.Array],
    *,
    epochs: int = 50,
    batch_size: int = 4096,
    target_scale: float = 1.0,
    callback: Callable[[int, QuadraticNetwork], None] | None = None,
    verbose: bool = False,
    patience_early_stop: int = 0,
) -> tuple[QuadraticNetwork, nnx.Optimizer, jax.Array]:
    # NOTE: an optax learning-rate schedule (e.g. reduce_on_plateau) can be
    # chained into *optimizer* instead of passing a scheduler object here.
    from nneuroutil.helpers import TicTocTimer

    x_train, y_train = xy_train
    x_validate, y_validate = xy_validate
    n_train = x_train.shape[0]

    timer = TicTocTimer()

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

    @nnx.jit
    def validate_step(model: QuadraticNetwork) -> jax.Array:
        return criterion(model(x_validate), y_validate) / target_scale

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    num_bad_epochs = 0

    losses = np.empty((epochs, 2, 1))
    key = jax.random.PRNGKey(0)

    for epoch in range(epochs):
        timer.tic()

        # {{{ training

        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)

        loss_epoch = 0.0
        num_batches = 0

        for i in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            batch_x = x_train[idx]
            batch_y = y_train[idx]

            loss_val = train_step(model, optimizer, batch_x, batch_y)
            loss_epoch += float(loss_val)
            num_batches += 1

            if verbose:
                log.info(
                    r"    \[batch %03d] loss %.5e", num_batches - 1, float(loss_val)
                )

        loss_train = loss_epoch / num_batches

        # }}}

        # {{{ validation

        loss_validate = float(validate_step(model))

        # }}}

        timer.toc()

        log.info(
            "[%03d/%03d] train %.5e validate %.5e (%s)",
            epoch + 1,
            epochs,
            loss_train,
            loss_validate,
            timer,
        )

        losses[epoch, 0, 0] = loss_train
        losses[epoch, 1, 0] = loss_validate

        # {{{ best model and early stopping

        if loss_validate < best_loss:
            best_loss = loss_validate
            best_state = nnx.state(model)
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
            callback(epoch, model)

    if best_state is not None:
        nnx.update(model, best_state)
        log.info(
            "Restored best model from epoch %03d (loss %.5e)", best_epoch, best_loss
        )

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
    assert isinstance(dtype, np.dtype)
    assert np.issubdtype(dtype, np.floating)

    z = jnp.zeros((resolution, resolution, 2 * d), dtype=dtype, device=device)
    c = generate_grid(
        *bbox,
        nx=resolution,
        ny=resolution,
        dtype=to_complex(dtype),
        xp=array_api_compat.array_namespace(z),
        device=device,
    )[:, :, None]

    for _ in range(1, maxit):
        Az = model(z)
        z = z.at[:, :, :d].set(Az[:, :, :d] + c.real)
        z = z.at[:, :, d:].set(Az[:, :, d:] + c.imag)

    return z[:, :, :d] + 1j * z[:, :, d:]


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
        from commonlib import determine_escape_radius

        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-quad.npz"

    outfile = outfile.with_stem(
        f"{outfile.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}-{jax.default_backend()}"
    )
    basename = outfile.with_suffix("")

    # }}}

    # {{{ plot predict

    import pickle

    from nneuroutil.visualization import figure, make_colorbar_axes

    def plot_predict(epoch: int, m: QuadraticNetwork) -> None:
        Zpredict = flax_predict(
            m,
            bbox=bbox,
            resolution=2 * resolution,
            maxit=maxit,
        )
        mask = jnp.sum(Zpredict.real**2 + Zpredict.imag**2, axis=-1) < escape_radius**2

        with figure(
            basename.with_stem(f"{basename.stem}-predict-{epoch:04d}"),
            overwrite=overwrite,
        ) as fig:
            ax = fig.gca()
            ax.imshow(np.array(mask), extent=bbox, cmap="binary")

    # }}}

    # {{{ train

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, dtype=mat.real.dtype, rngs=nnx.Rngs(0))
    criterion = MSELoss()
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=5.0e-3, weight_decay=1.0e-2),
        wrt=nnx.Param,
    )

    log.info("Network:\n%r", model, extra={"highlighter": None})

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
        loss_test = checkpoint.get("loss_test", 0.0)
        zmask = checkpoint["zmask"]

        plot_predict(epochs, model)
    else:
        from commonlib import (
            generate_timeseries_dataset,
            make_timeseries_datasets,
        )

        # {{{ generate dataset

        x, y, zmask = generate_timeseries_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            maxit=maxit,
            escape_radius=escape_radius,
        )
        xy_train, xy_validate, xy_test = cast(
            tuple[
                tuple[jax.Array, jax.Array],
                tuple[jax.Array, jax.Array],
                tuple[jax.Array, jax.Array],
            ],
            make_timeseries_datasets(
                x, y, (0.8, 0.1, 0.1), rng=np.random.default_rng(42)
            ),
        )

        # }}}

        # {{{ train

        _, y_train = xy_train
        target_scale = jnp.mean(y_train.conj() * y_train).real.item()

        model, _optimizer, losses = flax_train(
            model,
            criterion,
            optimizer,
            xy_train,
            xy_validate,
            epochs=epochs,
            batch_size=batch_size,
            target_scale=target_scale,
            patience_early_stop=early_stop_patience,
            callback=plot_predict,
        )

        # }}}

        # {{{ test loss

        x_test, y_test = xy_test
        predictions = model(x_test)
        loss_test = float(criterion(predictions, y_test) / target_scale)

        log.info("[%03d/%03d] test %.5e", epochs, epochs, loss_test)

        # }}}

        # {{{ checkpoint

        num_epochs_trained = len(losses)
        with open(weightsfile, "wb") as f:
            pickle.dump(
                {
                    # model
                    "model": nnx.state(model),
                    "epochs": epochs,
                    "num_epochs_trained": num_epochs_trained,
                    "losses": np.array(losses),
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
                f,
            )

        # }}}

    # }}}

    # {{{ plot fractal

    with figure(
        basename.with_stem(f"{basename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(np.array(zmask), extent=bbox, cmap="binary")

    # }}}

    # {{{ plot loss

    losses = np.asarray(losses)

    with figure(
        basename.with_stem(f"{basename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        ax.semilogy(losses[:, 0, 0], label="Train")
        ax.semilogy(losses[:, 1, 0], label="Validation")
        ax.axhline(loss_test, color="k", ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

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

    from nneuroutil.flax_extras import jax_default_device, set_jax_config

    with jax_default_device(args.device):
        set_jax_config()
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        jax.config.update("jax_default_matmul_precision", "high")

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

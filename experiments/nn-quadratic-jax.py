# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import pickle
from collections.abc import Callable, Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from commonlib import Config, make_argument_parser
from nneuroutil.array_api_extras import default_rng
from nneuroutil.helpers import machine_info, to_complex
from nneuroutil.interop.flax import JaxMemoryTracker, available_device_names
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.quadratic_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains or runs inference with a quadratic neural network: f(z) = (W z)^2 + b.

Examples:

    > {SCRIPT_PATH.name} --device cpu train dataset.npz
    > {SCRIPT_PATH.name} --device cpu infer dataset-quad.pkl
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

        from nneuroutil.interop.flax import (
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
                interleaved=True,
                kernel_init=quadratic_uniform(),
                dtype=dtype,
                rngs=rngs,
            ),
            ComplexQuadratic(interleaved=True),
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
        z = z.at[:, :, ::2].set(Az[:, :, ::2] + c.real)
        z = z.at[:, :, 1::2].set(Az[:, :, 1::2] + c.imag)

    return z[:, :, ::2] + 1j * z[:, :, 1::2]


# }}}


# {{{ helpers


def _as_jax(x: Any) -> Any:
    """Convert NumPy arrays back to JAX arrays (e.g. after unpickling)."""
    return jnp.asarray(x) if isinstance(x, np.ndarray) else x


def _plot_predict(
    filename: pathlib.Path,
    model: QuadraticNetwork,
    config: Config,
    *,
    tag: str,
    overwrite: bool,
) -> None:
    Zpredict = flax_predict(
        model,
        bbox=config.bbox,
        resolution=2 * config.resolution,
        maxit=config.maxit,
    )

    escape_radius = config.escape_radius
    mask = jnp.sum(Zpredict.real**2 + Zpredict.imag**2, axis=-1) < escape_radius**2

    with figure(
        filename.with_stem(f"{filename.stem}-predict-{tag}"),
        overwrite=overwrite,
    ) as fig:
        ax = fig.gca()
        ax.imshow(np.array(mask), extent=config.bbox, cmap="binary")


def _plot_dataset(
    filename: pathlib.Path,
    zmask: jax.Array,
    *,
    bbox: tuple[float, float, float, float],
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-dataset"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()
        ax.imshow(np.array(zmask), extent=bbox, cmap="binary")


def _plot_loss(
    filename: pathlib.Path,
    losses: jax.Array,
    loss_test: float,
    *,
    overwrite: bool,
) -> None:
    with figure(
        filename.with_stem(f"{filename.stem}-loss"), overwrite=overwrite
    ) as fig:
        ax = fig.gca()

        losses_np = np.asarray(losses)
        ax.semilogy(losses_np[:, 0, 0], label="Train")
        ax.semilogy(losses_np[:, 1, 0], label="Validation")
        ax.axhline(loss_test, color="k", ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()


def _plot_weights(
    filename: pathlib.Path,
    model: QuadraticNetwork,
    mat: jax.Array,
    *,
    overwrite: bool,
) -> None:
    # NOTE: the nn is trained to approximate the trajectory, not the matrix A,
    # even though the structure is the same (`(W x)^2 + b`). We look at the
    # linear layer to get a sense of what it managed to do here. However,
    #   1. we do not expect the matrices to match
    #   2. we do expect them to have similar singular values (i.e. the nn captured
    #      the general dynamics of the problem)

    # FIXME: this is pretty horrible. does it even work between versions?
    learned_A = model.model.layers[0].kernel.get_value()  # ty: ignore[missing-attribute]
    A = mat.astype(learned_A.dtype)

    with figure(
        filename.with_stem(f"{filename.stem}-weights"),
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


# {{{ train


def train(
    filename: pathlib.Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    weights: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    from commonlib import (
        generate_timeseries_dataset,
        load_matrix_datasets,
        split_timeseries_datasets,
    )

    machine = machine_info(SCRIPT_PATH)
    log.info("Machine:\n%s", machine.pretty())

    (device,) = jax.devices(jax.default_backend())
    log.info("Device: %s", device)

    mem = JaxMemoryTracker(device)
    mem.add_record("Start")

    provenance = {**machine._asdict(), "device": str(device)}

    # {{{ load parameters

    if overrides is None:
        overrides = {}

    checkpoint: dict[str, Any] | None = None
    if weights is not None:
        if not weights.exists():
            log.error("Weights file does not exist: '%s'.", weights)
            return 1

        with open(weights, "rb") as f:
            checkpoint = pickle.load(f)

        config = Config.from_sources(checkpoint["config"], overrides)
        mat = jnp.asarray(checkpoint["mat"])
    else:
        config = Config.from_sources(overrides)

        try:
            matrices = cast(
                jax.Array,
                load_matrix_datasets(
                    filename, key=config.datakey, device=device, xp=jnp
                ),
            )
            mat = matrices[config.dataidx].astype(jnp.complex64)
        except Exception as exc:
            log.error("Failed to load dataset '%s': %s", filename.name, exc)
            return 1

    mem.add_record("After data load")

    if outfile is None:
        outfile = filename.parent / f"{filename.stem}-quad.npz"

    outfile = outfile.with_stem(f"{outfile.stem}-{config.suffix}").with_suffix("")
    weightsfile = outfile.with_suffix(".pkl")

    if not overwrite and weightsfile.exists():
        log.error(
            "Weights file already exist (use --force to overwrite): '%s'", weightsfile
        )
        return 1

    # }}}

    # {{{ train

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, dtype=mat.real.dtype, rngs=nnx.Rngs(0))
    criterion = MSELoss()
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=config.lr, weight_decay=1.0e-2),
        wrt=nnx.Param,
    )

    log.info("Network:\n%r", model, extra={"highlighter": None})

    losses_previous: jax.Array | None = None
    if checkpoint is not None:
        nnx.update(model, jax.tree.map(_as_jax, checkpoint["model"]))
        nnx.update(optimizer, jax.tree.map(_as_jax, checkpoint["optimizer"]))

        losses_previous = checkpoint["losses"]
        log.info(
            "Resuming from '%s' (%d epochs trained)",
            weights,
            checkpoint["num_epochs_trained"],
        )

    # {{{ generate dataset

    x, y, _, zmask = cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        generate_timeseries_dataset(
            mat,
            bbox=config.bbox,
            resolution=config.resolution,
            maxit=config.maxit,
            escape_radius=config.escape_radius,
        ),
    )
    xy_train, xy_validate, xy_test = cast(
        tuple[
            tuple[jax.Array, jax.Array],
            tuple[jax.Array, jax.Array],
            tuple[jax.Array, jax.Array],
        ],
        split_timeseries_datasets(
            x, y, config.split, rng=default_rng(jax, config.seed)
        ),
    )

    mem.add_record("After data generation")

    # }}}

    # {{{ train

    def plot_predict(epoch: int, m: QuadraticNetwork) -> None:
        # NOTE: plot 10 epochs and then only every 10th one to save time / memory
        if epoch > 10 and epoch % 10 != 0:
            return

        _plot_predict(
            outfile,
            m,
            config,
            tag=f"{epoch:04d}",
            overwrite=overwrite,
        )

    _, y_train = xy_train
    target_scale = jnp.mean(y_train.conj() * y_train).real.item()

    model, _optimizer, losses = flax_train(
        model,
        criterion,
        optimizer,
        xy_train,
        xy_validate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        target_scale=target_scale,
        patience_early_stop=config.early_stop_patience,
        # callback=plot_predict,
    )

    mem.add_record("After training")

    # }}}

    # {{{ test loss

    x_test, y_test = xy_test
    predictions = model(x_test)
    loss_test = float(criterion(predictions, y_test) / target_scale)

    log.info("[%03d/%03d] test %.5e", config.epochs, config.epochs, loss_test)

    mem.add_record("After test loss")

    # }}}

    # {{{ checkpoint

    from dataclasses import asdict

    if losses_previous is not None:
        losses = jnp.concatenate((jnp.asarray(losses_previous), losses))

    num_epochs_trained = len(losses)
    with open(weightsfile, "wb") as f:
        pickle.dump(
            {
                # model
                "provenance": provenance,
                "model": nnx.state(model),
                "optimizer": nnx.state(optimizer),
                # loss
                "num_epochs_trained": num_epochs_trained,
                "losses": np.array(losses),
                "loss_test": loss_test,
                # dataset
                "inputfile": str(filename),
                "mat": np.asarray(mat),
                "zmask": np.asarray(zmask),
                # parameters
                "config": asdict(config),
            },
            f,
        )
    log.info("Saved checkpoint: '%s'.", weightsfile)

    mem.add_record("After checkpoint")

    # }}}

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, loss_test, overwrite=overwrite)
    _plot_weights(outfile, model, mat, overwrite=overwrite)

    # }}}

    mem.add_record("Finished")
    log.info("Memory:\n%s", mem)

    return 0


# }}}


# {{{ infer


def infer(
    weightsfile: pathlib.Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not weightsfile.exists():
        log.error("Weights file does not exist: '%s'.", weightsfile)
        return 1

    (device,) = jax.devices(jax.default_backend())
    log.info("Device: %s", device)

    mem = JaxMemoryTracker(device)
    mem.add_record("Start")

    # {{{ load parameters

    if overrides is None:
        overrides = {}

    with open(weightsfile, "rb") as f:
        checkpoint = pickle.load(f)
    config = Config.from_sources(checkpoint["config"], overrides)

    losses = checkpoint["losses"]
    loss_test = checkpoint["loss_test"]
    zmask = checkpoint["zmask"]
    mat = jnp.asarray(checkpoint["mat"])

    if outfile is None:
        outfile = weightsfile.with_suffix("")
    outfile = outfile.with_suffix("")

    # }}}

    # {{{ load model

    d = mat.shape[0]
    model = QuadraticNetwork(d, d, dtype=mat.real.dtype, rngs=nnx.Rngs(0))
    nnx.update(model, jax.tree.map(_as_jax, checkpoint["model"]))

    mem.add_record("After model load")

    # }}}

    # {{{ predict

    _plot_predict(
        outfile,
        model,
        config,
        tag="final",
        overwrite=overwrite,
    )

    mem.add_record("After prediction")

    # }}}

    # {{{ plot

    _plot_dataset(outfile, zmask, bbox=config.bbox, overwrite=overwrite)
    _plot_loss(outfile, losses, loss_test, overwrite=overwrite)
    _plot_weights(outfile, model, mat, overwrite=overwrite)

    # }}}

    mem.add_record("Finished")

    log.info("Memory:\n%s", mem)

    return 0


# }}}


if __name__ == "__main__":
    parser, _ = make_argument_parser(
        description=SCRIPT_LONG_HELP,
        devices=available_device_names(),
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    from nneuroutil.interop.flax import jax_default_device, set_jax_config

    with jax_default_device(args.device):
        set_jax_config()
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        jax.config.update("jax_default_matmul_precision", "high")

        if args.command == "train":
            raise SystemExit(
                train(
                    args.filename,
                    overrides=vars(args),
                    weights=args.weights,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        elif args.command == "infer":
            raise SystemExit(
                infer(
                    args.filename,
                    overrides=vars(args),
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise SystemExit(1)

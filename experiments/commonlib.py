# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from nneuroutil.dmd import DMD
from nneuroutil.helpers import TicTocTimer, module_logger, tictoc
from nneuroutil.typing import Array2D, Array3D, Array4D

if TYPE_CHECKING:
    import torch
    from torch import nn

log = module_logger("lib")

# {{{ matplotlib


def set_matplotlib_style() -> None:
    """Sets default plotting settings from ``scienceplots``.

    The defaults can be overwritten with a local ``default.mplstyle`` file.
    """
    import matplotlib.pyplot as mp

    from nneuroutil.visualization import load_scienceplots_styles

    # load scienceplots
    styles = load_scienceplots_styles()
    mp.style.use(styles.get(["science", "ieee"]))

    # load our overwrites
    mplstyle = pathlib.Path(__file__).parent / "default.mplstyle"
    if mplstyle.exists():
        mp.style.use(mplstyle)
    else:
        log.error("Could not find matplotlib style: '%s'.", mplstyle)

    ext = os.environ.get("PROJECT_FIGURE_FORMAT", "jpg")
    mp.rc("savefig", format=ext)


set_matplotlib_style()

# }}}


# {{{ escape radius


def determine_escape_radius(A: Array2D) -> float:
    xp = array_api_compat.array_namespace(A)

    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    sigma = xp.linalg.svdvals(A)

    return (2.0 * n**0.5 / xp.min(sigma) ** 2).item()


# }}}


# {{{ generate_grid


def generate_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    nx: int = 128,
    ny: int = 128,
    dtype: Any = None,
    xp: Any = None,
    device: Any = None,
) -> Array2D:
    if xp is None:
        xp = np

    x = xp.linspace(xmin, xmax, nx, device=device)
    y = xp.linspace(ymin, ymax, ny, device=device)

    result = x[None, :] + 1j * y[:, None]
    if dtype is not None and dtype != result.dtype:
        result = xp.astype(result, dtype)

    return result


# }}}


# {{{ generate dataset


def generate_dataset(
    A: Array2D,
    *,
    bbox: tuple[float, float, float, float],
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
) -> tuple[Array4D, Array2D]:
    xp = array_api_compat.array_namespace(A)

    d = A.shape[0]
    assert A.shape[0] == A.shape[1]

    # NOTE: allow a maximum element z[i] such that norm(z) ~ escape_radius
    max_z = (escape_radius + 1) / d**0.5
    escape_radius_sqr = escape_radius**2

    # generate grid
    c = generate_grid(
        *bbox,
        nx=resolution,
        ny=resolution,
        dtype=A.dtype,
        xp=xp,
        device=A.device,
    )
    c = c[:, :, None]

    # iterate
    result = [xp.zeros((resolution, resolution, d), dtype=A.dtype, device=A.device)]

    for i in range(1, maxit):
        # evolve
        z_prev = result[i - 1]
        z_new = xp.einsum("ij,klj->kli", A, z_prev) ** 2 + c

        # clip escaped values over escape radius
        escaped = xp.sum(z_new.real**2 + z_new.imag**2, axis=-1) > escape_radius_sqr
        z_new = xp.where(escaped[..., None], max_z, z_new)

        result.append(z_new)

    return xp.stack(result), c.squeeze(-1)


def generate_training_dataset(
    mat: Array2D,
    *,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    in_dmd: bool = False,
    xp: Any = None,
) -> tuple[Array2D, Array2D, Array2D]:
    if xp is None:
        xp = array_api_compat.array_namespace(mat)

    log.info("Generating dataset on %s", bbox)

    with tictoc("generate_dataset"):
        Z, c = generate_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            escape_radius=escape_radius,
            maxit=maxit,
        )

    X, Y = split_dataset(
        Z,
        C=c,
        escape_radius=escape_radius,
        in_dmd=in_dmd,
        split_real_imaginary=True,
    )
    log.info("Dataset: %s", X.shape)

    mask = xp.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, axis=-1) < escape_radius**2
    log.info("Points:")
    log.info("    Converged: %d/%d", xp.sum(mask), mask.size().numel())
    log.info("    Max: %.5e (radius %.5e)", xp.max(xp.abs(Z)), escape_radius)
    log.info("    Mean: %.5e", xp.mean(xp.abs(Z)))

    # NOTE: mask only returned for plotting
    return X, Y, Z[-1].clone()


# }}}


# {{{ split_dataset


def split_dataset(
    Z: Array4D,
    *,
    C: Array4D | None = None,
    escape_radius: float | None = None,
    in_dmd: bool = False,
    split_real_imaginary: bool = True,
    xp: Any = None,
) -> tuple[Array2D, Array2D]:
    if xp is None:
        xp = array_api_compat.array_namespace(Z, C)

    # NOTE: we have an array of shape (maxit, n, n, d) and we want to
    #   * make our input/output be (Z_n, Z_{n + 1}) so that we can learn the dynamics
    #   * take only the converged points (based on mask)
    #   * squish everything down to use d as the "features"

    # NOTE: we reshape the array in two different ways:
    #   1. if `in_dmd`: to a 3D array of shape (n * n, maxit, d)
    #   2. if not `in_dmd`: to a 2D array of shape (n * n * maxit, d)

    maxit, nx, ny, d = Z.shape

    # if provided, strip out points that escaped
    if escape_radius is not None:
        mask = xp.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, axis=-1) < escape_radius**2
    else:
        mask = xp.ones((nx, ny), dtype=xp.bool, device=Z.device)

    X = Z[:-1, mask, :]
    if in_dmd:  # noqa: SIM108
        X = xp.permute_dims(X, (1, 0, 2))
    else:
        X = xp.reshape(X, (-1, d))

    # if provided, strip the constant offset so we predict
    #   Y' = X
    #   where Y' = Z[1:] - c
    #         X = Z[:-1]
    if C is not None:
        C = xp.expand_dims(xp.expand_dims(C, axis=0), axis=-1)
        C = xp.broadcast_to(C, (maxit - 1, nx, ny, d))
        Y = Z[1:, mask, :] - C[:, mask, :]
    else:
        Y = Z[1:, mask, :]

    if in_dmd:  # noqa: SIM108
        Y = xp.permute_dims(Y, (1, 0, 2))
    else:
        Y = xp.reshape(Y, (-1, d))

    if split_real_imaginary:
        X = xp.concat([X.real, X.imag], axis=-1)
        Y = xp.concat([Y.real, Y.imag], axis=-1)

    return X, Y


# }}}


# {{{ dmd


def dmd_predict(
    model: DMD,
    Z: Array4D,
    *,
    bbox: tuple[float, float, float, float],
) -> Array3D:
    xp = array_api_compat.array_namespace(Z)
    device = Z.device

    maxit, nx, ny, d = Z.shape
    c = generate_grid(*bbox, nx=nx, ny=ny, dtype=model.dtype, xp=xp, device=device)
    c = c[None, :, :]

    # initialize the state to (z0, c)
    z = xp.zeros((d + 1, nx, ny), dtype=model.dtype, device=device)
    z[-1, :, :] = c

    # project the state back to the reduced state space
    zhat = xp.einsum("ij,j...->i...", model.U.conj().T, z)

    # iterate in the reduced state space for speeeeed!!!
    for _ in range(1, maxit):
        zhat = model(zhat)

    # NOTE: remove the last element because that corresponds to c theoretically
    return xp.einsum("ij,j...->i...", model.U, zhat)[:-1]


# }}}


# {{{ iter_batches


def iter_batches(
    X: Array2D,
    Y: Array2D,
    *,
    batch_size: int = 4096,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[Array2D, Array2D]]:
    xp = array_api_compat.array_namespace(X, Y)
    if rng is None:
        rng = np.random.default_rng()

    idx = xp.asarray(rng.permutation(len(X)), dtype=xp.int32, device=X.device)
    for start in range(0, len(X), batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], Y[sl]


# }}}


# {{{ torch_train


def torch_train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 4096,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.Tensor]:
    import torch

    timer = TicTocTimer()

    device = X.device
    compiled_model = torch.compile(model)

    num_batches = X.shape[0] // batch_size

    losses = torch.empty(epochs, device=device)
    target_scale = torch.mean(Y.conj() * Y).real

    for epoch in range(epochs):
        loss_epoch = torch.tensor(0.0, device=device)

        timer.tic()

        for batch_x, batch_y in iter_batches(X, Y, batch_size=batch_size):
            optimizer.zero_grad()

            # compute loss function
            predictions = compiled_model(batch_x)
            loss = criterion(predictions, batch_y) / target_scale

            # compute gradient
            loss.backward()

            # update weights
            optimizer.step()

            loss_epoch += loss.detach()

        losses[epoch] = loss_epoch.item() / num_batches
        scheduler.step(losses[epoch])  # ty: ignore[invalid-argument-type]

        timer.toc()
        log.info("[%03d/%03d] loss %.5e (%s)", epoch + 1, epochs, loss_epoch, timer)

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

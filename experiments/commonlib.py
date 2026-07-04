# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from nneuroutil.dmd import DMD
from nneuroutil.helpers import TicTocTimer, module_logger, tictoc
from nneuroutil.typing import Array2D, Array3D, Array4D

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, TensorDataset

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
    re_min: float,
    re_max: float,
    im_min: float,
    im_max: float,
    *,
    nx: int = 128,
    ny: int = 128,
    dtype: Any = None,
    xp: Any = None,
    device: Any = None,
) -> Array2D:
    """Generate a 2D complex grid in ``[re_min, re_max] x [im_min, im_max]``."""

    if xp is None:
        xp = np

    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid sizes (nx, ny) are not positive: ({nx}, {ny})")

    if re_min >= re_max or im_min >= im_max:
        raise ValueError(
            f"bounding box is not valid: [{re_min}, {re_max}] x [{im_min}, {im_max}]"
        )

    # convert to closest complex type -- should probably complain instead
    dtype = xp.result_type(dtype, xp.complex64)

    c_re = xp.linspace(re_min, re_max, nx, device=device)
    c_im = xp.linspace(im_min, im_max, ny, device=device)

    c = c_re[None, :] + 1j * c_im[:, None]
    if dtype is not None and dtype != c.dtype:
        c = xp.astype(c, dtype)

    return c


# }}}


# {{{ generate_dataset


def generate_dataset(
    A: Array2D,
    *,
    bbox: tuple[float, float, float, float],
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    xp: Any = None,
) -> tuple[Array4D, Array2D]:
    r"""Generate a full dataset based on the dynamical system:

    .. math::

        z_{n + 1} = (A z_n)^2 + c

    where :math:`z_0 = 0` and :math:`c \in \mathbb{C}` is generated on the grid
    described by *bbox* at a uniform resolution *resolution* in both the real
    and imaginary parts. The system is evolved for *maxit* iteration and all
    values that are larger than *escape_radius* are automatically clipped
    such that :math:`\|z_n\|_2 < R`.

    :returns: a tuple ``(z, c)`` where ``z`` is a complex tensor with shape
        ``(maxit, resolution, resolution, d)`` hat contains all the trajectories
        and ``c`` is a complex tensor of shape ``(resolution, resolution)``
        that contains the grid from which these trajectories were generated.
    """
    if escape_radius <= 0.0:
        raise ValueError(f"'escape_radius' must be positive: {escape_radius}")

    if maxit <= 0:
        raise ValueError(f"'maxit' must be positive: {maxit}")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"'A' must be a square matrix: {A.shape}")

    if xp is None:
        xp = array_api_compat.array_namespace(A)

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

    # NOTE: allow a maximum element z[i] such that norm(z) ~ escape_radius
    d = A.shape[0]
    max_z = (escape_radius + 1) / d**0.5
    escape_radius_sqr = escape_radius**2

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


def post_process_dataset(
    z: Array4D,
    escape_radius: float,
    *,
    stack_real_imaginary: bool = True,
    xp: Any = None,
) -> tuple[Array3D, Array2D]:
    """Post-process the dataset returned from :func:`generate_dataset`.

    * Mask out all the escaped points.
    * Permute the array so that it has shape ``(ntrajectories, maxit, d)``.
    * Stack into a real array of shape ``(ntrajectories, maxit, 2 d)``.

    :arg z: a complex tensor of shape ``(maxit, resolution, resolution, d)``.
    :arg c: a complex tensor of shape ``(resolution, resolution)``
    :returns: a tuple of ``(z, mask)`` of the post-processed dataset and the mask
        that was used to compute it.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(z)

    if escape_radius <= 0:
        raise ValueError(f"'escape_radius' must be positive: {escape_radius}")

    # 1. mask
    mask = xp.sum(z[-1].real ** 2 + z[-1].imag ** 2, axis=-1) < escape_radius**2
    z = z[:, mask, :]

    # 2. permute
    z = xp.permute_dims(z, (1, 0, 2))

    # 3. stack
    if stack_real_imaginary:
        z = xp.concat([z.real, z.imag], axis=-1)

    return z, mask


# }}}


# {{{{ generate_timeseries_dataset


def generate_timeseries_dataset(
    mat: Array2D,
    *,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    xp: Any = None,
) -> tuple[Array3D, Array3D, Array2D]:
    """
    :returns: a tuple ``(x, y, mask)``, where *x* and *y* are arrays of shape
        ``(ntrajectories, maxit - 1, 2d)`` and the *mask* is an array of shape
        ``(resolution, resolution)``. The two arrays represent consecutive
        time steps ``(z_n, z_{n + 1} - c)`` and the *mask* gives the converged
        points at the final time step over the whole *bbox*.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(mat)

    log.info("Generating dataset on %s", bbox)
    with tictoc("generate_dataset"):
        z, c = generate_dataset(
            mat,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
            xp=xp,
        )

    # only keep converged points
    z, mask = post_process_dataset(z, escape_radius, stack_real_imaginary=True, xp=xp)
    c = c[mask][:, None, None]

    # create datasets
    d, _ = mat.shape
    x = z[:, :-1, :]
    y = xp.concat([z[:, 1:, :d] - c.real, z[:, 1:, d:] - c.imag], axis=-1)
    assert x.shape == y.shape

    zabs = xp.sqrt(z[:, :, :d] ** 2 + z[:, :, d:] ** 2)
    log.info("Dataset:")
    log.info("    Converged:    %d/%d", xp.sum(mask), mask.size().numel())
    log.info("    Size:         %s", z.shape)
    log.info("    Max:          %.5e (radius %.5e)", xp.max(zabs), escape_radius)
    log.info("    Spread:       %.5e ± %.5e", xp.mean(zabs), xp.std(zabs))

    # NOTE: mask only returned for plotting
    return x, y, mask


def torch_flatten(ds: Dataset) -> TensorDataset:
    import torch
    from torch.utils.data import Subset, TensorDataset

    if isinstance(ds, Subset):
        idx = ds.indices
        tensors = tuple(tensor[idx] for tensor in ds.dataset.tensors)  # ty: ignore[unresolved-attribute]
    elif isinstance(ds, TensorDataset):
        tensors = ds.tensors
    else:
        raise TypeError(f"unsupported dataset type: {type(ds)}")

    tensors = tuple(torch.reshape(tensor, (-1, tensor.shape[-1])) for tensor in tensors)
    return TensorDataset(*tensors)


def make_torch_timeseries_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: tuple[float, float, float] | None = None,
    *,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    import torch
    from torch.utils.data import TensorDataset, random_split

    if lengths is None:
        lengths = (0.8, 0.1, 0.1)

    # NOTE: this is what random_split seems to also be using
    if not math.isclose(sum(lengths), 1):
        raise ValueError(f"'lengths' must sum to 1: {lengths}")

    dataset = TensorDataset(x, y)
    generator = torch.Generator(x.device).manual_seed(seed)
    training, validation, testing = random_split(dataset, lengths, generator=generator)

    training = torch_flatten(training)
    validation = torch_flatten(validation)
    testing = torch_flatten(testing)

    log.info("Split:")
    log.info("    Training:     %d", len(training))
    log.info("    Validation:   %d", len(validation))
    log.info("    Testing:      %d", len(testing))

    return training, validation, testing


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


# {{{ torch_train


def torch_train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    xy_train: TensorDataset,
    xy_validate: TensorDataset,
    *,
    epochs: int = 50,
    batch_size: int = 4096,
    target_scale: float = 1.0,
    callback: Callable[[int, nn.Module], None] | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.Tensor]:
    import torch
    from torch.utils.data import DataLoader

    # FIXME: not very nice to just split these?
    x_train, _ = xy_train.tensors
    x_validate, y_validate = xy_validate.tensors
    device = x_train.device

    compiled_model = torch.compile(model)
    train_loader = DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device),
    )

    losses = torch.empty((epochs, 2, 1), device=device)

    timer = TicTocTimer()

    for epoch in range(epochs):
        # {{{ training

        loss_epoch = torch.tensor(0.0, device=device)
        num_batches = 0

        timer.tic()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            predictions = compiled_model(batch_x)
            loss = criterion(predictions, batch_y) / target_scale

            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach()
            num_batches += 1

        loss_train = loss_epoch / num_batches

        # }}}

        # {{{ validation

        with torch.no_grad():
            predictions = compiled_model(x_validate)
            loss_validate = criterion(predictions, y_validate) / target_scale

        # }}}

        scheduler.step(loss_validate)

        timer.toc()

        log.info(
            "[%03d/%03d] train %.5e validate %.5e (%s)",
            epoch + 1,
            epochs,
            loss_train,
            loss_validate,
            timer,
        )

        losses[epoch, 0] = loss_train.item()
        losses[epoch, 1] = loss_validate.item()

        if callback is not None:
            callback(epoch, compiled_model)  # ty: ignore[invalid-argument-type]

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

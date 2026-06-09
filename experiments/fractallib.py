# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.dmd import DMD
from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array2D, Array3D, Array4D

log = module_logger("fractallib")

# {{{ matplotlib


def set_plotting_defaults() -> None:
    """Sets default plotting settings from ``scienceplots``.

    The defaults can be overwritten with a local ``default.mplstyle`` file.
    """
    import matplotlib.pyplot as mp

    # NOTE: since v1.1.0 an import is required to import the styles
    import scienceplots  # noqa: F401

    dirname = pathlib.Path(__file__).parent
    mp.style.use(["science", "ieee"])

    mplstyle = dirname / "default.mplstyle"
    if mplstyle.exists():
        mp.style.use(mplstyle)
    else:
        log.error("Could not find matplotlib style: '%s'.", mplstyle)

    import os

    ext = os.environ.get("DMDAELIB_FIGURE_FORMAT", "jpg")
    mp.rcParams["savefig.format"] = ext


set_plotting_defaults()

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
) -> Array2D:
    if xp is None:
        xp = np

    x = xp.linspace(xmin, xmax, nx)
    y = xp.linspace(ymin, ymax, ny)

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

    assert A.shape[0] == A.shape[1]

    # NOTE: promoting A to complex because einsum doesn't handle heterogeneous dtypes
    A = xp.astype(A, xp.result_type(A.dtype, xp.complex64))
    d = A.shape[0]

    # NOTE: allow a maximum element z[i] such that norm(z) ~ escape_radius
    max_z = (escape_radius + 1) / d**0.5

    # generate grid
    c = generate_grid(*bbox, nx=resolution, ny=resolution, dtype=A.dtype, xp=xp)
    c = c[:, :, None]

    # iterate
    escape_radius_sqr = escape_radius**2
    result = xp.zeros((maxit, resolution, resolution, d), dtype=A.dtype)

    for i in range(1, maxit):
        # evolve
        z_prev = result[i - 1]
        z_new = xp.einsum("ij,klj->kli", A, z_prev) ** 2 + c

        # clip escaped values over escape radius
        # NOTE: we set it to `R + 1 / sqrt(d)` so that `norm2(z) == R + 1`
        escaped = xp.sum(z_new.real**2 + z_new.imag**2, axis=-1) > escape_radius_sqr
        z_new = xp.where(escaped[..., None], max_z, z_new)

        result[i] = z_new

    return result, c.squeeze(-1)


# }}}


# {{{ dmd


def dmd_predict(
    model: DMD,
    Z: Array4D,
    *,
    bbox: tuple[float, float, float, float],
) -> Array3D:
    xp = array_api_compat.array_namespace(Z)

    maxit, nx, ny, d = Z.shape
    c = generate_grid(*bbox, nx=nx, ny=ny, dtype=model.dtype, xp=xp)
    c = c[None, :, :]

    # initialize the state to (z0, c)
    z = xp.zeros((d + 1, nx, ny), dtype=model.dtype)
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
) -> Iterator[tuple[Array2D, Array2D]]:
    xp = array_api_compat.array_namespace(X, Y)

    idx = xp.randperm(len(X))
    for start in range(0, len(X), batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], Y[sl]


# }}}

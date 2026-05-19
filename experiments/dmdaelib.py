# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import array_api_compat
import numpy as np
import rich.logging

# {{{ typing

# FIXME: not clear if there's a nice way to do this with the array_api
Array1D: TypeAlias = Any
Array2D: TypeAlias = Any
Array3D: TypeAlias = Any
Array4D: TypeAlias = Any
Matrix: TypeAlias = Any
DTypeLike: TypeAlias = np.typing.DTypeLike

# }}}

# {{{ logging


def make_default_logger(module: str, level: int | None = None) -> logging.Logger:
    if level is None:
        level = logging.ERROR

    path = pathlib.Path(module)
    if path.exists():
        module = path.stem.replace("-", "_")

    log = logging.getLogger(f"dmdaelib.{module}")
    log.setLevel(level)
    log.addHandler(rich.logging.RichHandler())

    return log


log = make_default_logger(__name__, level=logging.INFO)

# }}}


# {{{ matplotlib


def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (10, 10),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "text": {"usetex": True},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 5},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "xtick": {"labelsize": 20, "direction": "inout"},
        "ytick": {"labelsize": 20, "direction": "inout"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    from contextlib import suppress

    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    for group, params in defaults.items():
        mp.rc(group, **params)


set_recommended_matplotlib()


@contextmanager
def axis(filename: pathlib.Path) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    if not filename.suffix:
        ext = mp.rcParams["savefig.format"]
        filename = filename.with_suffix(f".{ext}")

    fig = mp.figure(num=1)
    ax = fig.gca()

    try:
        yield ax
    finally:
        log.info("Saving figure in '%s'.", filename)
        fig.savefig(filename)
        mp.close(fig)


# }}}


# {{{ tictoc


@dataclass
class TicTocTimer:
    t_wall_start: float = field(default=0.0, init=False)
    t_wall: float = field(default=0.0, init=False)

    def tic(self) -> None:
        import time

        self.t_wall = 0.0
        self.t_wall_start = time.perf_counter()

    def toc(self) -> float:
        import time

        self.t_wall = time.perf_counter() - self.t_wall_start
        return self.t_wall

    def __str__(self) -> str:
        return f"walltime {self.t_wall:.5f} seconds."


@contextmanager
def timeit(text: str) -> Iterator[None]:
    tt = TicTocTimer()
    tt.tic()

    try:
        yield None
    finally:
        tt.toc()
        log.info("%s (%s)", text, tt)


# }}}


# {{{ escape radius


def determine_escape_radius(A: Matrix) -> float:
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
    A: Matrix,
    *,
    bbox: tuple[float, float, float, float],
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
) -> tuple[Array4D, Array2D]:
    xp = array_api_compat.array_namespace(A)

    assert A.shape[0] == A.shape[1]

    # NOTE: promoting A to complex because einsum doesn't handle heterogeneous dtypes
    A = A.to(A.dtype.to_complex())
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
        escaped = xp.sum(z_new.real**2 + z_new.imag**2, dim=-1) > escape_radius_sqr
        z_new = xp.where(escaped[..., None], max_z, z_new)

        result[i] = z_new

    return result, c.squeeze(-1)


# }}}


# {{{ dmd


@dataclass(frozen=True)
class DMD:
    Ahat: Array2D

    U: Array2D
    S: Array1D
    V: Array2D

    @property
    def dtype(self) -> np.dtype:
        return self.Ahat.dtype

    @property
    def input_size(self) -> int:
        return self.U.shape[0]

    @property
    def rank(self) -> int:
        return self.Ahat.shape[0]

    def __call__(self, x: Array1D) -> Array1D:
        xp = array_api_compat.array_namespace(x, self.Ahat, self.U, self.S, self.V)

        assert x.shape[0] == self.rank
        x = xp.einsum("ij,j...->i...", self.Ahat, x)

        return x


def dmd_build(
    X1: Array2D,
    X2: Array2D,
    *,
    rank: int | None = None,
    eps: float | None = None,
) -> DMD:
    xp = array_api_compat.array_namespace(X1, X2)

    U, S, Vh = xp.linalg.svd(X1, full_matrices=False)
    S = S.to(X1.dtype)

    if rank is not None:
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    if eps is None:
        eps = 10 * xp.finfo(X1.dtype).eps

    mask = xp.abs(S) > eps
    U, Vh = U[:, mask], Vh[mask, :]

    # construct reduced order model
    Ahat = U.conj().T @ X2 @ Vh.conj().T @ xp.diag(1 / S[mask])
    assert Ahat.ndim == 2
    assert Ahat.shape[0] == Ahat.shape[1]

    return DMD(Ahat, U, S, Vh)


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

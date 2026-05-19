# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import array_api_compat
import numpy as np
import rich.logging

if TYPE_CHECKING:
    import matplotlib.pyplot as mp

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
        mp.style.use(dirname / "default.mplstyle")

    import os

    ext = os.environ.get("DMDAELIB_FIGURE_FORMAT", "jpg")
    mp.rcParams["savefig.format"] = ext


set_plotting_defaults()


@contextmanager
def figure(
    filename: pathlib.Path,
    nrows: int = 1,
    ncols: int = 1,
    *,
    constrained_layout: bool = True,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Iterator[mp.Figure]:
    import matplotlib.pyplot as mp

    fig = mp.figure(constrained_layout=constrained_layout)

    for i in range(nrows * ncols):
        fig.add_subplot(nrows, ncols, i + 1)

    # FIXME: get size of one figure
    if figsize is None:
        width, height = mp.rcParams["figure.figsize"]
        figsize = (width * ncols, height * nrows)
    fig.set_size_inches(*figsize)

    try:
        yield fig
    finally:
        if not constrained_layout:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.3, hspace=0.1)

        savefig(fig, filename, **kwargs)
        mp.close(fig)


def savefig(
    fig: mp.Figure,
    filename: pathlib.Path,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0,
    normalize: bool = False,
    facecolor: str = "white",
    transparent: bool = False,
    overwrite: bool = True,
    **kwargs: Any,
) -> None:
    import matplotlib.pyplot as mp

    ext = mp.rcParams["savefig.format"]
    filename = pathlib.Path(filename)

    if not filename.suffix:
        filename = filename.with_suffix(f".{ext}").resolve()

    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    bbox_extra_artists = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            bbox_extra_artists.append(legend)

    log.info("Saving '%s'", filename)
    fig.savefig(
        filename,
        bbox_extra_artists=tuple(bbox_extra_artists),
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor=facecolor,
        transparent=transparent,
        **kwargs,
    )


def make_colorbar_axes(
    ax: mp.Axes,
    *,
    position: str = "right",
    size: str = "5%",
    pad: float = 0.1,
) -> mp.Axes:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)

    return cax


def scale_axis(ax: mp.Axes, scale: float) -> mp.Axes:
    return ax.inset_axes(((1.0 - scale) / 2, (1.0 - scale) / 2, scale, scale))


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

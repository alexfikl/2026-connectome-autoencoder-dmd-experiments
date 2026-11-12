# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib
import secrets
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import array_api_compat
import numpy as np

import nneuroutil.array_api_extras as xpx
from nneuroutil.helpers import FuzzyChoiceAction, dc_stringify, set_up_logging, tictoc
from nneuroutil.typing import Array1D, Array2D, Array3D, Array4D

if TYPE_CHECKING:
    import torch

set_up_logging("INFO")
log = logging.getLogger("lib")

# NOTE: ignore torch warnings about array_api_compat usage
warnings.filterwarnings(
    "ignore",
    message=r"Dynamo detected a call to a .functools\.lru_cache.-wrapped function",
    category=UserWarning,
)

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


# {{{ gpu_array_namespace


def gpu_array_namespace() -> tuple[Any, str]:
    xp = array_api_compat.array_namespace(np.empty(0))
    device = "cpu"

    try:
        import torch
    except ImportError:
        return xp, device

    from nneuroutil.interop.torch import available_device_names

    if any(dev.startswith("cuda") for dev in available_device_names()):
        xp = array_api_compat.array_namespace(torch.empty(0))
        device = "cuda"

    return xp, device


# }}}


# {{{ load_feature_datasets


class FeatureDataset(NamedTuple):
    x: Array3D[np.floating[Any]]
    """Feature tensor of shape ``(nsamples, nclasses, nfeatures)``."""
    labels: Array3D[np.integer[Any]]
    """Class index for each entry of :attr:`x` of shape ``(nsamples, nclasses, 1)``. """
    class_names: tuple[str, ...]
    """Class name for each entry along the class axis of :attr:`x`."""
    starts: Array1D[np.integer[Any]]
    """Feature group boundaries of length ``ngroups + 1``."""
    descrs: tuple[str, ...]
    """Description of each feature group."""


def make_class_labels(
    x: Array3D[np.floating[Any]],
    *,
    xp: Any = None,
) -> Array3D[np.integer[Any]]:
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    nsamples, nclasses, _ = x.shape
    indices = xp.arange(nclasses, dtype=xp.int64, device=array_api_compat.device(x))
    return xp.tile(indices[None, :, None], (nsamples, 1, 1))


def load_feature_datasets(
    filename: pathlib.Path,
    *,
    exclude: tuple[str, ...] | None = None,
    datakey: str = "datasets",
    device: Any = None,
    xp: Any = None,
) -> FeatureDataset:
    """Load a feature file written by ``make-features.py``.

    :arg datakey: name of the variable holding the class dictionary.
    """
    if not filename.exists():
        raise FileNotFoundError(f"file does not exist: '{filename}'")

    try:
        data = np.load(filename, allow_pickle=True)
    except Exception:
        raise RuntimeError(f"failed to load data from file: '{filename}'") from None

    if datakey not in data:
        raise ValueError(f"file does not contain '{datakey}' dataset: '{filename}'")

    datasets = data[datakey]
    if datasets.ndim == 0:
        datasets = datasets[()]

    if not isinstance(datasets, dict):
        raise ValueError(f"expected a dictionary in '{datakey}': '{filename}'")

    if len(datasets) <= 1:
        raise ValueError(f"dataset does not have sufficient classes: {len(datasets)}")

    if exclude is None:
        exclude = ()
    excludes = set(exclude)

    descrs = data["descrs"]
    if descrs.ndim == 0:
        descrs = descrs[()]
    descrs = tuple(str(d) for d in descrs)

    if xp is None:
        xp = array_api_compat.array_namespace(np.array([0]))

    classes = []
    xs: list[Array2D[np.floating[Any]]] = []
    for name, x in datasets.items():
        if name in excludes:
            continue

        if x.ndim != 2:
            raise ValueError(f"expected a 2d array for class '{name}': {x.shape}")

        if xs and x.shape[1] != xs[-1].shape[1]:
            raise ValueError(f"class '{name}' has inconsistent feature size: {x.shape}")

        if xs and x.shape[0] != xs[-1].shape[0]:
            raise ValueError(f"class '{name}' has inconsistent sample size: {x.shape}")

        if not np.all(np.isfinite(x)):
            raise ValueError(f"class '{name}' contains NaN or inf values")

        xs.append(x)
        classes.append(name)

    x = xp.asarray(np.stack(xs, axis=1), device=device)
    return FeatureDataset(
        x=x,
        labels=make_class_labels(x),
        class_names=tuple(classes),
        starts=xp.asarray(data["starts"], device=device),
        descrs=tuple(descrs),
    )


def load_matrix_datasets(
    filename: pathlib.Path,
    *,
    key: str = "matrices",
    device: Any = None,
    xp: Any = None,
) -> Array3D[np.number[Any]]:
    if not filename.exists():
        raise FileNotFoundError(filename)

    try:
        data = np.load(filename)
    except Exception as exc:
        raise RuntimeError(f"could not load file '{filename}'") from exc

    if key not in data:
        raise ValueError(f"'{filename}' does not contain '{key}' dataset")

    matrices = data[key]
    if matrices.ndim != 3:
        raise ValueError(f"expected a 3 dimensional array: {matrices.shape}")

    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    matrices = xp.asarray(matrices)
    if device is not None:
        matrices = array_api_compat.to_device(matrices, device)

    return matrices


# }}}


# {{{ escape radius


def determine_escape_radius(A: Array2D[np.number[Any]]) -> float:
    xp = array_api_compat.array_namespace(A)

    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    sigma = xp.linalg.svdvals(A)

    return (2.0 * n**0.5 / xp.min(sigma) ** 2).item()


# }}}


# {{{ configuration


@dataclass(frozen=True, slots=True)
class Config:
    # optimizer
    epochs: int = 50
    lr: float = 5.0e-3

    # cqn dataset
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03)
    resolution: int = 128
    maxit: int = 128
    escape_radius: float = 100.0

    # train + infer
    dataidx: int = 5
    datakey: str = "matrices"
    batch_size: int = 4096
    seed: int | None = None
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    early_stop_patience: int = 0

    @property
    def suffix(self) -> str:
        import hashlib
        import json

        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        digest = hashlib.sha256(payload).hexdigest()[:12]

        return f"m{self.dataidx:02d}-r{self.resolution}-n{self.maxit}-{digest}"

    @classmethod
    def from_sources(cls, *sources: Mapping[str, Any] | None) -> Self:
        values: dict[str, Any] = {f.name: f.default for f in fields(cls)}

        for source in sources:
            if source is None:
                continue

            values.update({
                key: value
                for key, value in source.items()
                if key in values and value is not None
            })

        if values["seed"] is None:
            values["seed"] = secrets.randbits(64)

        return cls(**values)

    def __str__(self) -> str:
        return dc_stringify(self)


def make_argument_parser(
    *,
    description: str = "",
    devices: frozenset[str] = frozenset(),
) -> tuple[
    argparse.ArgumentParser, argparse._SubParsersAction[argparse.ArgumentParser]
]:
    """Make the argument parser with the common options and the ``train`` and
    ``infer`` commands.

    The sub-parsers are also returned, so that scripts can add their own commands.
    """

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=description,
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(devices),
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
        "-r",
        "--resolution",
        type=int,
        default=None,
        help="Resolution (in the c complex plane) of the CQN grid",
    )
    parser.add_argument(
        "--maxit",
        type=int,
        default=None,
        help="Maximum number of iterations for the CQN",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Bounding box in the complex plane for the CQN",
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
    subparser = subparsers.add_parser(
        "train",
        formatter_class=HelpFormatter,
        description="Train a neural network on a CQN dataset.",
    )

    subparser.add_argument("filename", type=pathlib.Path)
    subparser.add_argument(
        "-w",
        "--weights",
        type=pathlib.Path,
        default=None,
        help="Checkpoint to resume training from (starts from scratch if not given)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--escape-radius",
        type=float,
        default=None,
        help="Escape radius used to clip trajectories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (a random one is drawn if not given)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Epochs to wait for an improvement before stopping (0 disables)",
    )
    subparser.add_argument(
        "--dataidx",
        type=int,
        default=None,
        help="Index of the matrix to train on",
    )

    subparser = subparsers.add_parser(
        "infer",
        formatter_class=HelpFormatter,
        description="Run inference with a trained neural network.",
    )
    subparser.add_argument("filename", type=pathlib.Path)

    return parser, subparsers


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
) -> Array2D[np.complexfloating[Any]]:
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


def generate_timeseries(
    A: Array2D[np.number[Any]],
    *,
    bbox: tuple[float, float, float, float],
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    xp: Any = None,
) -> tuple[Array4D[np.complexfloating[Any]], Array2D[np.complexfloating[Any]]]:
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

    # convert to closest complex type -- should probably complain instead
    dtype = xp.result_type(A.dtype, xp.complex64)

    # generate grid
    c = generate_grid(
        *bbox,
        nx=resolution,
        ny=resolution,
        dtype=dtype,
        xp=xp,
        device=A.device,
    )
    c = c[:, :, None]

    # NOTE: allow a maximum element z[i] such that norm(z) ~ escape_radius
    d = A.shape[0]
    max_z = (escape_radius + 1) / d**0.5
    escape_radius_sqr = escape_radius**2

    # iterate
    c_flat = xp.reshape(c, (-1, 1))
    if array_api_compat.is_writeable_array(c):  # spell: disable
        z = xp.zeros((maxit, resolution * resolution, d), dtype=dtype, device=A.device)
        for i in range(1, maxit):
            # evolve
            z_new = (z[i - 1] @ A.T) ** 2 + c_flat  # ty: ignore[missing-attribute]

            # clip escaped values over escape radius
            escaped = (
                xp.sum(z_new.real**2 + z_new.imag**2, axis=-1, keepdims=True)
                > escape_radius_sqr
            )
            z[i] = xp.where(escaped, max_z, z_new)

        z = xp.reshape(z, (maxit, resolution, resolution, d))
    else:
        z_init = xp.zeros((resolution * resolution, d), dtype=dtype, device=A.device)
        result = [z_init]
        for i in range(1, maxit):
            # evolve
            z_prev = result[i - 1]
            z_new = (z_prev @ A.T) ** 2 + c_flat  # ty: ignore[missing-attribute]

            # clip escaped values over escape radius
            escaped = (
                xp.sum(z_new.real**2 + z_new.imag**2, axis=-1, keepdims=True)
                > escape_radius_sqr
            )
            z_new = xp.where(escaped, max_z, z_new)

            result.append(z_new)

        z = xp.reshape(xp.stack(result), (maxit, resolution, resolution, d))

    return z, c.squeeze(-1)


def compute_escaped_mask(
    z: Array4D[np.complexfloating[Any]],
    escape_radius: float,
) -> Array3D[np.bool]:
    """Compute a boolean mask of the pinned (escaped) states of *z*.

    This uses the same escape criterion as :func:`generate_timeseries`.

    :arg z: trajectories of shape ``(n_levels, nx, ny, d)``.
    """
    xp = array_api_compat.array_namespace(z)
    sqr_norm = xp.sum(xp.real(z) ** 2 + xp.imag(z) ** 2, axis=-1)

    return xp.astype(
        sqr_norm > escape_radius**2,
        xp.bool,
    )


def post_process_dataset(
    z: Array4D[np.complexfloating[Any]],
    escape_radius: float,
    *,
    interleaved: bool = True,
    min_escape_iters: int | None = None,
    xp: Any = None,
) -> tuple[Array3D[np.floating[Any]], Array2D[np.bool]]:
    """Post-process the dataset returned from :func:`generate_dataset`.

    * Mask out points that escape before *min_escape_iters*.
    * Permute the array so that it has shape ``(ntrajectories, maxit, d)``.
    * Stack/Interleave into a real array of shape ``(ntrajectories, maxit, 2 d)``.

    :arg z: a complex tensor of shape ``(maxit, resolution, resolution, d)``.
    :arg min_escape_iters: minimum iterations before escape.  ``None`` keeps
        only fully converged points (escaped at the final timestep).
    :arg c: a complex tensor of shape ``(resolution, resolution)``
    :returns: a tuple of ``(z, mask)`` of the post-processed dataset and the mask
        that was used to compute it.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(z)

    if escape_radius <= 0:
        raise ValueError(f"'escape_radius' must be positive: {escape_radius}")

    maxit = z.shape[0]

    if min_escape_iters is None:
        min_escape_iters = maxit

    # 1. mask
    sqr_norm = xp.sum(xp.real(z) ** 2 + xp.imag(z) ** 2, axis=-1)
    escaped = sqr_norm > escape_radius**2
    never_escaped = xp.all(~escaped, axis=0)
    first_escaped = xp.argmax(xp.astype(escaped, xp.int32), axis=0)
    escape_time = xp.where(never_escaped, maxit, first_escaped)

    mask = escape_time >= min_escape_iters
    z = z[:, mask, :]

    # 2. permute
    z = xp.permute_dims(z, (1, 0, 2))

    # 3. interleave
    if interleaved:
        z = xpx.view_as_real(z, copy=True)
    else:
        z = xp.concat([z.real, z.imag], axis=-1)

    return z, mask


# }}}


# {{{ generate_timeseries_dataset


def generate_timeseries_dataset(
    mat: Array2D[np.number[Any]],
    *,
    bbox: tuple[float, float, float, float] = (-0.03, 0.03, -0.03, 0.03),
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    min_escape_iters: int | None = None,
    xp: Any = None,
) -> tuple[
    Array3D[np.floating[Any]],
    Array3D[np.floating[Any]],
    Array2D[np.bool],
    Array2D[np.bool],
]:
    """
    :arg min_escape_iters: Minimum iterations before a trajectory is allowed
        to escape.  Points escaping earlier are excluded.  ``None`` keeps
        only fully converged points (escaped at the final timestep).

    :returns: a tuple ``(x, y, mask, in_set)``, where *x* and *y* are arrays of
        shape ``(ntrajectories, maxit - 1, 2d)`` and the *mask* and *inset* are an
        arrays of shape ``(resolution, resolution)``. The two arrays represent
        consecutive time steps ``(z_n, z_{n + 1} - c)``, the *mask* gives the
        chosen points in the dataset, and *in_set* gives the converged points
        at the final time step over the whole *bbox*.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(mat)

    log.info("Generating dataset on %s", bbox)
    with tictoc("generate_dataset"):
        z, c = generate_timeseries(
            mat,
            bbox=bbox,
            escape_radius=escape_radius,
            resolution=resolution,
            maxit=maxit,
            xp=xp,
        )

    # converged points over the whole grid (never escaped, i.e. still below the
    # escape radius at the final time step)
    in_set = (
        xp.sum(xp.real(z[-1]) ** 2 + xp.imag(z[-1]) ** 2, axis=-1) < escape_radius**2
    )

    # only keep points that survive long enough
    z, mask = post_process_dataset(
        z,
        escape_radius,
        interleaved=True,
        min_escape_iters=min_escape_iters,
        xp=xp,
    )
    c = c[mask][:, None, None]

    # create datasets
    x = z[:, :-1, :]
    y_real = z[:, 1:, ::2] - c.real
    y_imag = z[:, 1:, 1::2] - c.imag
    y = xp.reshape(
        xp.stack([y_real, y_imag], axis=-1),
        (*y_real.shape[:-1], 2 * y_real.shape[-1]),
    )
    assert x.shape == y.shape

    del c

    zabs = xp.sqrt(z[:, :, ::2] ** 2 + z[:, :, 1::2] ** 2)
    log.info("Dataset:")
    log.info("    Converged:    %d/%d", xp.sum(in_set), math.prod(in_set.shape))
    log.info("    Picked:       %d/%d", xp.sum(mask), math.prod(mask.shape))
    log.info("    Size:         %s", z.shape)
    log.info("    Max:          %.5e (radius %.5e)", xp.max(zabs), escape_radius)
    log.info("    Spread:       %.5e ± %.5e", xp.mean(zabs), xp.std(zabs))

    del zabs

    # NOTE: mask only returned for plotting
    return x, y, mask, in_set


def split_timeseries_datasets(
    x: Array3D[np.complexfloating[Any]],
    y: Array3D[np.complexfloating[Any]],
    lengths: tuple[float, float, float] | None = None,
    *,
    mask: Array2D[np.bool] | None = None,
    rng: xpx.RandomGenerator | None = None,
    xp: Any = None,
) -> tuple[
    tuple[Array2D[np.floating[Any]], Array2D[np.floating[Any]]],
    tuple[Array2D[np.floating[Any]], Array2D[np.floating[Any]]],
    tuple[Array2D[np.floating[Any]], Array2D[np.floating[Any]]],
]:
    from nneuroutil.data import random_split

    if lengths is None:
        lengths = (0.8, 0.1, 0.1)

    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    def flatten_pairs(
        xs: Array3D, ys: Array3D, ms: Array2D | None
    ) -> tuple[Array2D, Array2D]:
        xs = xp.reshape(xs, (-1, xs.shape[-1]))
        ys = xp.reshape(ys, (-1, ys.shape[-1]))

        # NOTE: when given, `mask` drops the (state, next-state) pairs that
        # carry no dynamics (see `compute_trajectory_mask`)
        if ms is None:
            return xs, ys

        ms = xp.reshape(ms, (-1,))
        return xs[ms], ys[ms]

    if mask is None:
        training, validation, testing = (
            flatten_pairs(xs, ys, None)
            for xs, ys in random_split((x, y), lengths, rng=rng, xp=xp)
        )
    else:
        training, validation, testing = (
            flatten_pairs(xs, ys, ms)
            for xs, ys, ms in random_split((x, y, mask), lengths, rng=rng, xp=xp)
        )

    log.info("Split:")
    log.info("    Training:     %d", len(training[0]))
    log.info("    Validation:   %d", len(validation[0]))
    log.info("    Testing:      %d", len(testing[0]))

    return training, validation, testing


# }}}


# {{{ mask_dataset


class TrajectoryMasks(NamedTuple):
    state: torch.Tensor
    transition: torch.Tensor


def compute_trajectory_mask(
    x: Array3D[np.floating[Any]],
    *,
    escape_radius: float | None = None,
    rtol: float | None = None,
    atol: float = 1.0e-8,
    keep_escape_transition: bool = False,
    xp: Any = None,
) -> TrajectoryMasks:
    """Compute the masks of usable states and transitions in ``x``.

    Two kinds of states carry no dynamics and are masked out:

    * *pinned* (clipped) states: their squared norm exceeds the escape radius
      (see ``commonlib.generate_dataset``). If *keep_escape_transition* is
      *False* (default), transitions into the pinned state are also dropped,
      preventing constant clipping artifacts from corrupting linear fits.
    * *settled* states: the trajectory has converged to a fixed point and only
      moves by a relative step ``|z_{n + 1} - z_n| / (|z_n| + atol)`` below *rtol*.

    With both *escape_radius* and *rtol* unset nothing is masked.
    """
    from nneuroutil.array_api_extras import view_as_complex

    if xp is None:
        xp = array_api_compat.array_namespace(x)

    batch_size, window_size = x.shape[:2]
    device = array_api_compat.device(x)

    zx = view_as_complex(x)
    zn = xp.sum(xp.real(zx) ** 2 + xp.imag(zx) ** 2, axis=-1)

    if escape_radius is None:
        active = xp.ones_like(zn, dtype=xp.bool)
    else:
        active = zn <= escape_radius**2

    # a transition is dynamic if the trajectory still moves along it
    if rtol is None:
        moving = xp.ones((batch_size, window_size - 1), dtype=xp.bool, device=device)
    else:
        dz = xp.linalg.vector_norm(zx[:, 1:, :] - zx[:, :-1, :], axis=-1)
        zn = xp.sqrt(zn[:, :-1])
        moving = dz > rtol * (zn + atol)

    # a state is transient if the trajectory moves into or out of it
    empty = xp.zeros((batch_size, 1), dtype=xp.bool, device=device)
    moving_in = xp.concat([empty, moving], axis=1)
    moving_out = xp.concat([moving, empty], axis=1)

    trans_active = (
        active[:, :-1] if keep_escape_transition else (active[:, :-1] & active[:, 1:])
    )

    return TrajectoryMasks(
        state=active & (moving_in | moving_out),
        transition=trans_active & moving,
    )


# }}}

# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Literal, cast

import numpy as np
import torch

from nneuroutil.helpers import FuzzyChoiceAction, tictoc
from nneuroutil.interop.torch import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)
from nneuroutil.visualization import figure, make_colorbar_axes

log = logging.getLogger("lib.dmd")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Uses a classic DMD to learn the dynamics of a quadratic iteration

    z <- (A z)^2 + c

Example:

    > {SCRIPT_PATH.name} dataset.npz
"""


def complexify(R: torch.Tensor) -> torch.Tensor:
    """Best complex M with R ≈ real_embedding(M)."""
    p = 0.5 * (R[0::2, 0::2] + R[1::2, 1::2])  # Re part
    q = 0.5 * (R[0::2, 1::2] - R[1::2, 0::2])  # Im part
    return (p + 1j * q).T


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int = 5,
    method: Literal["dmd", "edmd"] = "dmd",
    bbox: tuple[float, float, float, float] = (-0.04, 0.04, -0.04, 0.04),
    escape_radius: float | None = 100.0,
    resolution: int = 64,
    maxit: int = 192,
    overwrite: bool = False,
) -> int:
    from commonlib import determine_escape_radius

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

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

    mat = torch.from_numpy(matrices[dataidx]).to(torch.complex64).to(device)
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = filename.parent / (
            f"{filename.stem}-{method}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}.npz"
        )

    basename = outfile.with_suffix("")
    log.info("DMD")

    # }}}

    # {{{ generate dataset

    mem.add_record("Before data generation")

    d = mat.shape[0]
    with tictoc(f"Generated data on {bbox}"):
        from commonlib import generate_timeseries

        Z, c = cast(
            tuple[torch.Tensor, torch.Tensor],
            generate_timeseries(
                mat,
                bbox=bbox,
                resolution=resolution,
                escape_radius=escape_radius,
                maxit=maxit,
            ),
        )

        # NOTE: keep initial/final states for convergence tracking
        Z_init = Z[0].clone()
        Z_final = Z[-1].clone()

        mask = torch.sum(Z_final.real**2 + Z_final.imag**2, dim=-1) < escape_radius**2

    mem.add_record("After data generation")

    log.info("Points:")
    log.info("Converged: %d", torch.sum(mask))
    log.info(
        "Max: %.5e (radius %.5e)",
        torch.max(torch.abs(Z_final)),
        escape_radius,
    )
    log.info("Mean: %.5e", torch.mean(torch.abs(Z_final)))

    with figure(basename.with_stem(f"{basename.stem}-dataset")) as fig:
        ax = fig.gca()
        ax.imshow(mask.cpu().numpy(), extent=bbox, cmap="binary")

    # }}}

    # {{{ build dataset

    # NOTE: To build the dataset we
    #   1. Interleave complex coordinates into real states (num_traj, maxit, 2d)
    #   2. Keep only the trajectories that converged (same `mask` as before)
    #   3. Treat stacked [Re(c), Im(c)] as control inputs U
    #   4. Use all transitions of the selected trajectories

    with tictoc("Build dataset"):
        from nneuroutil.array_api_extras import view_as_real

        # NOTE: Z has shape (maxit, resolution, resolution, d) and `mask` is
        # (resolution, resolution). Keep only the converged trajectories and
        # drop the full grid (`Z_init` and `Z_final` are already clones).
        Z_conv = Z[:, mask, :]
        X_real = view_as_real(Z_conv.permute(1, 0, 2))
        c_real = torch.stack([c.real, c.imag], dim=-1)[mask]

        X1 = X_real[:, :-1, :].reshape(-1, 2 * d)
        X2 = X_real[:, 1:, :].reshape(-1, 2 * d)
        U = c_real.unsqueeze(1).expand(-1, maxit - 1, 2).reshape(-1, 2)
        del Z, X_real

        log.info(
            "Dataset: X1 = %s, U = %s, X2 = %s (transitions: %d/%d)",
            X1.shape,
            U.shape,
            X2.shape,
            X1.shape[0],
            int(torch.sum(mask)) * (maxit - 1),
        )

    mem.add_record("Built dataset")

    # }}}

    # {{{ train

    from nneuroutil.dmdc import build_dense_dmdc, build_dense_extended_dmdc

    with tictoc("Constructed DMD model"):
        if method == "dmd":
            dmd = build_dense_dmdc(X1, U, X2, method="ridge")
        elif method == "edmd":
            # NOTE: we know our system, so we choose some nice observables:
            #   x, (A x)^2, (A x)^4
            # This might be missing some cross terms, but it is what it is.
            from nneuroutil.array_api_extras import view_as_complex

            dmd = build_dense_extended_dmdc(
                [
                    lambda x: x,
                    lambda x, A=mat: view_as_real((view_as_complex(x) @ A.T) ** 2),
                    lambda x, A=mat: view_as_real((view_as_complex(x) @ A.T) ** 4),
                ],
                X1,
                U,
                X2,
                method="ridge",
            )
        else:
            raise ValueError(f"unknown method: {method!r}")

    mem.add_record("Constructed DMD model")
    A_ref = mat.cpu()
    A_dmd = complexify(dmd.A).cpu()  # ty: ignore[invalid-argument-type]

    with figure(basename.with_stem(f"{basename.stem}-sigma")) as fig:
        ax = fig.gca()

        # NOTE: none of these will match or anything, but plotting for visualization
        S = torch.linalg.svdvals(A_ref)
        ax.semilogy(S.cpu().numpy(), label=r"$\sigma_A$")

        S = torch.linalg.svdvals(A_dmd)
        ax.semilogy(S.cpu().numpy(), label=r"$\sigma_{{DMD}}$")

        S = torch.linalg.svdvals(X1.mH @ X1).sqrt()
        ax.semilogy(S.cpu().numpy(), label=r"$\sigma_{{X}}$")

        ax.set_xlabel("n")
        ax.legend()

    with figure(basename.with_stem(f"{basename.stem}-lambda")) as fig:
        ax = fig.gca()

        theta = np.linspace(0, 2.0 * np.pi, 512)
        ax.plot(np.sin(theta), np.cos(theta), "k-", lw=1)

        # NOTE: none of these will match or anything, but plotting for visualization
        S = torch.linalg.eigvals(A_ref).numpy()
        ax.plot(S.real, S.imag, "o", label=r"$\lambda_A$")

        S = torch.linalg.eigvals(A_dmd).numpy()
        ax.plot(S.real, S.imag, "o", label=r"$\lambda_{{DMD}}$")

        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")
        ax.set_xlim((-3, 3))
        ax.set_aspect("equal")
        ax.legend()

    # NOTE: this should really be < 1 if we want a stable rollout. If it's > 1,
    # then slowly all the trajectories will explode regardless of Z0
    log.info("max |lambda_dmd| = %.8e", np.max(np.abs(S)))

    # }}}

    # {{{ predict

    with tictoc(f"Predicting using {method.upper()}"):
        # TODO: would be nice to predict at 2x resolution to see if we're
        # able to do anything here. That would need to also evolve at 2x
        # resolution to be able to compare the two.
        Zpredict = torch.empty(
            (maxit, resolution, resolution, d),
            dtype=Z_init.dtype,
            device=device,
        )
        Zpredict[0] = Z_init
        c_grid_real = torch.stack([c.real, c.imag], dim=-1)
        Xinit = view_as_real(Z_init)
        Xpredict = dmd.encode(Xinit)
        del Z_init

        for i in range(1, maxit):
            Xpredict = dmd.evolve(Xpredict, c_grid_real)
            Xdecoded = cast("torch.Tensor", dmd.decode(Xpredict))
            Zpredict[i] = torch.view_as_complex(
                Xdecoded.reshape(*Xdecoded.shape[:-1], -1, 2)
            )

        Zpredict_final = Zpredict[-1].clone()
        Zpredict_converged = Zpredict[:, mask, :]
        del Zpredict

    mem.add_record("After predicting")

    # }}}

    # {{{ plot predict error

    from nneuroutil.dmdc import relative_forecast_error

    with figure(basename.with_stem(f"{basename.stem}-forecast")) as fig:
        ax = fig.gca()

        Upredict = c[mask, None].unsqueeze(0).expand(maxit, -1, -1)
        error = relative_forecast_error(dmd, Z_conv, Upredict, Zpredict_converged)
        del Z_conv, Zpredict_converged, Upredict

        error = cast("torch.Tensor", error).cpu().numpy()
        assert error.shape == (maxit, torch.sum(mask))

        median = np.median(error, axis=1)
        p25 = np.percentile(error, 25, axis=1)
        p75 = np.percentile(error, 75, axis=1)

        ax.plot(median, label="Median")
        ax.fill_between(np.arange(maxit), p25, p75, alpha=0.3)
        ax.set_xlabel("$n$")
        ax.set_ylabel("Error")
        ax.legend()

    mem.add_record("Freed trajectories")

    # }}}

    # {{{ observable error

    if method == "dmd":
        obs_labels = (r"$z$",)
        obs_slices = (slice(0, 2 * d),)
    elif method == "edmd":
        obs_labels = (r"$z$", r"$(A z)^2$", r"$(A z)^4$")
        obs_slices = (
            slice(0, 2 * d),
            slice(2 * d, 4 * d),
            slice(4 * d, 6 * d),
        )
    else:
        raise AssertionError

    X1_lift = dmd.encode(X1)
    X1_rec = dmd.decode(X1_lift)
    res_rec = (torch.linalg.norm(X1 - X1_rec) / torch.linalg.norm(X1)).item()
    del X1, X1_rec

    X2_lift_hat = dmd.evolve(X1_lift, U)
    del X1_lift, U

    X2_lift = dmd.encode(X2)

    obs_residuals: list[float] = []
    for label, s in zip(obs_labels, obs_slices, strict=True):
        denom = torch.linalg.norm(X2_lift[:, s])
        res_val = torch.linalg.norm(X2_lift[:, s] - X2_lift_hat[:, s]) / denom
        obs_residuals.append(res_val.item())

        log.info("Observable %s 1-step residual: %.8e", label, res_val.item())

    res_lift = (
        torch.linalg.norm(X2_lift - X2_lift_hat) / torch.linalg.norm(X2_lift)
    ).item()
    del X2_lift

    X2_phys_hat = dmd.decode(X2_lift_hat)
    del X2_lift_hat

    res_phys = (torch.linalg.norm(X2 - X2_phys_hat) / torch.linalg.norm(X2)).item()
    del X2, X2_phys_hat

    mem.add_record("Computed residuals & errors")

    with figure(basename.with_stem(f"{basename.stem}-observables")) as fig:
        ax = fig.gca()

        ax.bar(obs_labels, obs_residuals)
        ax.set_yscale("log")
        ax.set_xlabel("Observable")
        ax.set_ylabel("Residual")

    # }}}

    # {{{ fit residuals (lifted vs physical vs decoder reconstruction)

    log.info("Residual (lifted space): %.8e", res_lift)
    log.info("Residual (physical decoded): %.8e", res_phys)
    log.info("Residual (decoder consistency): %.8e", res_rec)

    with figure(basename.with_stem(f"{basename.stem}-residuals")) as fig:
        ax = fig.gca()

        ax.bar(
            ["lift. fit", "phys. dec.", "dec. recon."],
            [res_lift, res_phys, res_rec],
        )
        ax.set_yscale("log")
        ax.set_ylabel("Relative Error")

    # }}}

    # {{{ plot final error

    from nneuroutil.vision import as_grayscale, otsu_threshold_from_image

    # NOTE: this is a linear operator, so we expect everybody to converge
    escaped = (
        torch.sum(
            Zpredict_final.real**2 + Zpredict_final.imag**2,
            dim=-1,
        )
        < escape_radius**2
    )
    total_escaped = torch.sum(escaped)
    log.info("Escaped %d (points %d)", total_escaped, resolution * resolution)

    # NOTE: try to tell who escaped by looking at the error
    error = torch.linalg.norm(Z_final - Zpredict_final, dim=-1)

    # binarize error in a nice way
    gray = as_grayscale(error, dtype=torch.uint8)
    thresh = otsu_threshold_from_image(gray)

    with figure(basename.with_stem(f"{basename.stem}-error")) as fig:
        ax = fig.gca()

        im = ax.imshow(torch.log10(error).cpu().numpy(), extent=bbox)
        fig.colorbar(im, cax=make_colorbar_axes(ax))

    with figure(basename.with_stem(f"{basename.stem}-predict")) as fig:
        ax = fig.gca()

        ax.imshow((gray < thresh).cpu().numpy(), extent=bbox, cmap="binary")  # ty: ignore[unsupported-operator]

    mem.add_record("Finished")

    # }}}

    log.info("Memory:\n%s", mem)

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
        "--method",
        choices=("dmd", "edmd"),
        default="dmd",
        help="DMD variant to use for learning the dynamical system",
    )
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
        default=64,
        help="Resolution (in the c complex plane) of the images",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default deviced used by the framework",
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

    with torch_default_device(args.device):
        # NOTE: use TF32 for matrix multiplication (should be faster, similar precision)
        torch.set_float32_matmul_precision("high")

        raise SystemExit(
            main(
                args.filename,
                dataidx=args.dataidx,
                resolution=args.resolution,
                method=args.method,
                overwrite=args.force,
            )
        )

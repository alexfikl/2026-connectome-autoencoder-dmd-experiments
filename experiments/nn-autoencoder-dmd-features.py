# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch

import dmdaelib
from nneuroutil.dmdc import DenseDMDc
from nneuroutil.helpers import FuzzyChoiceAction
from nneuroutil.interop.torch import (
    TorchMemoryTracker,
    available_device_names,
    torch_default_device,
)

log = logging.getLogger("lib.autoencoder_features")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Compute classification features from trained DMD autoencoder checkpoints and
merge per-condition feature files for downstream benchmarks.

Examples:

    > {SCRIPT_PATH.name} --device cuda compute data/ae-dmd/Rest_LR_Task/*.pkl
    > {SCRIPT_PATH.name} merge data/ae-dmd/*
"""


# {{{ features


def compute_features(
    model: dmdaelib.DMDAutoencoder,
    dmdc: DenseDMDc[Any],
    *,
    bbox: tuple[float, float, float, float],
    resolution: int,
    maxit: int,
) -> dict[str, np.ndarray]:
    r"""Compute features of the trained model for classification.

    The latent basis is arbitrary (every matrix has its own encoder), so the
    features are either invariant to it or are lifted to the state space, where
    the node coordinates are shared between matrices.

    * ``eigvals``: the eigenvalues of the operator ``dmdc.A``, sorted by decreasing
      modulus.
    * ``jac_map``, ``jac_stats``: the gradient norm
      :math:`|\nabla_c \log |z_N||` of the decoded final log-norm with respect to
      the (physical) control, on the grid, and its (mean, std, median, p90, max).
    * ``inset_fraction``: fraction of the grid predicted in the set (decoded final
      norm below the escape radius). The following describe the decoded final states
      of these points, which approximate the attractors in the node space.
    * ``attractor_node_magnitude``: ``(d,)`` mean over the points of the per-node
      magnitude ``|z_i|`` of the final state.
    * ``attractor_node_phase``: ``(2, d)`` mean over the points of the unit phasor
      ``z_i / |z_i| * conj(c / |c|)``, i.e. the phase of each node relative to the
      control, as its (cos, sin) components. The modulus of the mean is the
      coherence of the phase over the points.
    * ``attractor_energy``: the first 8 normalized squared singular values of the
      final states (uncentered), i.e. how many directions the attractors span.
    """
    scale = max(abs(b) for b in bbox)

    # {{{ spectrum

    lam = torch.linalg.eigvals(dmdc.A.detach().cpu().to(torch.float64))  # ty: ignore[missing-attribute]
    lam = lam[torch.argsort(lam.abs(), descending=True)]

    # }}}

    # {{{ sensitivity to the control

    c = dmdaelib.grid_control(
        bbox,
        resolution,
        dtype=model.dtype,
        device=next(model.parameters()).device,
    )
    c = c.detach().clone().requires_grad_()
    g_final = model.rollout(dmdaelib.encode_origin(model, c), dmdc, maxit)[:, -1]
    _, logn_final = model.decode(g_final)
    (grad,) = torch.autograd.grad(logn_final.sum(), c)

    # NOTE: `c` is normalized by `scale` and the log-norm by `dmdaelib.LOG_SCALE`
    jac = torch.linalg.vector_norm(grad, dim=-1) * (dmdaelib.LOG_SCALE / scale)
    jac = jac.detach().cpu()
    jac_stats = torch.stack([
        jac.mean(),
        jac.std(),
        jac.median(),
        torch.quantile(jac, 0.9),
        jac.max(),
    ])

    # }}}

    # {{{ attractors

    # NOTE: states of the in-set points at the end of the rollout, rebuilt from
    # the decoded direction (unit, in the orthonormal POD basis) and norm
    with torch.no_grad():
        g_last = model.rollout(dmdaelib.encode_origin(model, c.detach()), dmdc, maxit)
        direction, logn = model.decode(g_last[:, -1])
        norm = (torch.exp(logn * dmdaelib.LOG_SCALE) - dmdaelib.LOG_EPS).clamp(min=0.0)
        states = norm[:, None] * (direction @ cast(torch.Tensor, model.basis).T)

        inset = norm < model.escape_radius
        final = states[inset].cpu().to(torch.float64)
        c_inset = c.detach()[inset].cpu().to(torch.float64)

    nenergy = 8
    inset_fraction = torch.tensor(float(inset.float().mean()), device="cpu")
    nnodes = model.input_size
    attractor_node_magnitude = torch.zeros(nnodes, dtype=torch.float64, device="cpu")
    attractor_node_phase = torch.zeros((2, nnodes), dtype=torch.float64, device="cpu")
    attractor_energy = torch.zeros(nenergy, dtype=torch.float64, device="cpu")

    if final.shape[0] > 1:
        z = torch.view_as_complex(final.reshape(final.shape[0], -1, 2).contiguous())
        zc = torch.view_as_complex(c_inset.contiguous())

        attractor_node_magnitude = z.abs().mean(dim=0)

        phasor = z / z.abs().clamp(min=1.0e-12)
        phasor = phasor * (zc / zc.abs().clamp(min=1.0e-12)).conj()[:, None]
        phasor = phasor.mean(dim=0)
        attractor_node_phase = torch.stack([phasor.real, phasor.imag])

        energy = torch.linalg.svdvals(final) ** 2
        energy = energy[:nenergy] / energy.sum()
        attractor_energy[: energy.shape[0]] = energy

    # }}}

    return {
        "eigvals": lam.numpy(),
        "jac_map": jac.reshape(resolution, resolution).numpy(),
        "jac_stats": jac_stats.numpy(),
        "inset_fraction": inset_fraction.numpy(),
        "attractor_node_magnitude": attractor_node_magnitude.numpy(),
        "attractor_node_phase": attractor_node_phase.numpy(),
        "attractor_energy": attractor_energy.numpy(),
    }


def make_feature_vector(
    feats: Mapping[str, np.ndarray], *, neig: int = 3, nenergy: int = 3
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Gather the useful features of :func:`compute_features` in a single vector.

    The format matches ``make-features.py``: the feature vector, the boundaries of
    the feature groups (``ngroups + 1``) and a description of each group. The node
    magnitudes carry most of the information; the small groups are secondary. The
    node phases and ``inset_fraction`` are left out, since they are nearly constant
    and a shape quantity, respectively.
    """
    # NOTE: the operator is real, so the eigenvalues come in conjugate pairs that
    # have the same modulus; we only keep one of each
    eigvals = feats["eigvals"]
    moduli = np.sort(np.abs(eigvals[eigvals.imag >= 0]))[::-1][:neig]

    groups = {
        "Leading eigenvalue moduli": moduli,
        "Control sensitivity (log)": np.log(feats["jac_stats"]),
        "Attractor node magnitudes": feats["attractor_node_magnitude"],
        "Attractor energy": feats["attractor_energy"][:nenergy],
    }

    x = np.concatenate([np.ravel(v) for v in groups.values()]).astype(np.float64)
    starts = np.cumsum([0, *(np.size(v) for v in groups.values())])
    if not np.all(np.isfinite(x)):
        raise ValueError("features are not finite")

    return x, starts, tuple(groups)


# }}}


# {{{ compute


def _compute_one(
    weightsfile: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not weightsfile.exists():
        log.error("Weights file does not exist: '%s'.", weightsfile)
        return 1

    device = torch.get_default_device()
    checkpoint = torch.load(weightsfile, map_location=device, weights_only=True)

    inputfile = checkpoint.get("inputfile")
    if inputfile is None:
        log.error("Checkpoint does not contain an 'inputfile': '%s'.", weightsfile)
        return 1

    inputfile = pathlib.Path(inputfile)
    if not inputfile.exists():
        log.error("Input file does not exist: '%s'.", inputfile)
        return 1

    with np.load(inputfile) as data:
        if "scan_type" not in data:
            log.error("Input file does not contain 'scan_type': '%s'.", inputfile)
            return 1

        scan_type = str(data["scan_type"])

    config = dmdaelib.DMDConfig.from_sources(checkpoint["config"])
    dmdc = dmdaelib.load_latent_dmd(checkpoint)

    stem = (outfile if outfile is not None else weightsfile).with_suffix("")
    featfile = stem.with_name(f"{stem.name}-features.npz")
    if not overwrite and featfile.exists():
        log.error("Features file exists (use --force to overwrite): '%s'", featfile)
        return 1

    model = dmdaelib.make_dmd_autoencoder(
        config,
        checkpoint["mat"].shape[0],
        device=device,
        dtype=torch.float32,
    )
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        log.error(
            "Cannot load '%s': the model architecture does not match the "
            "checkpoint.\n%s",
            weightsfile,
            exc,
        )
        return 1
    model.eval()

    feats = compute_features(
        model,
        dmdc,
        bbox=config.bbox,
        resolution=config.resolution,
        maxit=config.maxit,
    )
    feats["dmdc_A"] = dmdc.A.detach().cpu().numpy()  # ty: ignore[missing-attribute]
    feats["dmdc_B"] = dmdc.B.detach().cpu().numpy()  # ty: ignore[missing-attribute]
    feats["x"], feats["starts"], descrs = make_feature_vector(feats)
    feats["descrs"] = np.array(descrs)
    feats["scan_type"] = np.array(scan_type)
    if "loss_test" in checkpoint:
        feats["loss_test"] = checkpoint["loss_test"].detach().cpu().numpy()

    eigvals = feats["eigvals"]
    log.info(
        "%s |lambda| (top 8): %s",
        weightsfile.name,
        np.array2string(np.abs(eigvals[:8]), precision=3),
    )
    log.info(
        "%s jacobian (mean, std, median, p90, max): %s",
        weightsfile.name,
        feats["jac_stats"],
    )

    np.savez(featfile, allow_pickle=True, **feats)
    log.info("Saved features: '%s'.", featfile)
    return 0


def compute(
    files: Sequence[pathlib.Path],
    *,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if outfile is not None and len(files) > 1:
        log.error("--outfile can only be used with a single input checkpoint.")
        return 1

    device = torch.get_default_device()
    log.info("Device: %s", device)

    mem = TorchMemoryTracker(device)
    mem.add_record("Start")

    for weightsfile in files:
        rc = _compute_one(weightsfile, outfile=outfile, overwrite=overwrite)
        if rc != 0:
            return rc

    mem.add_record("Finished")
    log.info("Memory:\n%s", mem)
    return 0


# }}}


# {{{ merge


def _natural_sort_key(name: str) -> str:
    return re.sub(r"(\d+)", lambda match: match.group(1).zfill(20), name)


def merge(
    indirs: Sequence[pathlib.Path],
    *,
    pattern: str = "*-features.npz",
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not indirs:
        log.warning("No directories given.")
        return 0

    if outfile is None:
        outfile = indirs[0].parent / "dataset-features-ae-dmd.npz"
    outfile = outfile.with_suffix(".npz")

    if not overwrite and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
        return 1

    datasets = {}
    scan_types: dict[str, str] = {}
    starts = None
    descrs = None

    for indir in indirs:
        if not indir.is_dir():
            log.error("Not a directory: '%s'.", indir)
            return 1

        name = indir.name
        if name in datasets:
            log.error("Duplicate dataset name: '%s'.", name)
            return 1

        found = sorted(
            indir.glob(pattern), key=lambda path: _natural_sort_key(path.name)
        )
        if not found:
            log.error("No files matching '%s' in '%s'.", pattern, indir)
            return 1

        xs = []
        scan_type: str | None = None
        for filename in found:
            with np.load(filename) as data:
                if starts is None:
                    starts, descrs = data["starts"], data["descrs"]

                if not (
                    np.array_equal(data["starts"], starts)
                    and np.array_equal(data["descrs"], descrs)
                ):
                    log.error("Inconsistent feature groups: '%s'.", filename)
                    return 1

                if "scan_type" not in data:
                    log.error("File does not contain 'scan_type': '%s'.", filename)
                    return 1

                filename_scan_type = str(data["scan_type"])
                if scan_type is None:
                    scan_type = filename_scan_type
                elif filename_scan_type != scan_type:
                    log.error(
                        "Inconsistent scan types in '%s': '%s' != '%s'.",
                        indir,
                        scan_type,
                        filename_scan_type,
                    )
                    return 1

                xs.append(data["x"])

        assert scan_type is not None
        datasets[name] = np.stack(xs)
        scan_types[name] = scan_type
        log.info("Gathered %d matrices for '%s'.", len(xs), name)

    np.savez(
        outfile,
        datasets=datasets,
        scan_types=scan_types,
        starts=starts,
        descrs=descrs,
    )
    log.info("Saved merged features: '%s'.", outfile)

    return 0


# }}}


if __name__ == "__main__":

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default device used by PyTorch",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only log warnings and errors",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparser = subparsers.add_parser(
        "compute",
        formatter_class=parser.formatter_class,
        description=(
            "Compute classification features from one or more trained DMD "
            "autoencoder checkpoints and save each to '{basename}-features.npz'."
        ),
    )
    subparser.add_argument(
        "files",
        nargs="+",
        type=pathlib.Path,
        help="Trained checkpoint files (*.pkl)",
    )

    subparser = subparsers.add_parser(
        "merge",
        formatter_class=parser.formatter_class,
        description=(
            "Merge the features written by 'compute' into a single file for "
            "make-features.py consumers. Each given directory is treated as one "
            "dataset (class), named after the directory, and all its matching files "
            "are stacked together."
        ),
    )
    subparser.add_argument(
        "indirs",
        nargs="+",
        type=pathlib.Path,
        help="Directory containing the features of a single dataset",
    )
    subparser.add_argument(
        "--pattern",
        default="*-features.npz",
        help="Glob pattern used to find the feature files in each directory",
    )

    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    with torch_default_device(args.device):
        torch.set_float32_matmul_precision("highest")

        if args.command == "compute":
            raise SystemExit(
                compute(
                    args.files,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        elif args.command == "merge":
            raise SystemExit(
                merge(
                    args.indirs,
                    pattern=args.pattern,
                    outfile=args.outfile,
                    overwrite=args.force,
                )
            )
        else:
            raise SystemExit(1)

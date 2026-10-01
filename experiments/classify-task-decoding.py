# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import cast

import numpy as np
import torch

from nneuroutil.helpers import FuzzyChoiceAction, module_logger
from nneuroutil.torch_extras import available_device_names, torch_default_device

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Classify structural and functional MRI data using standard methods

Example:

    > {SCRIPT_PATH.name} dataset1.npz dataset2.npz dataset3.npz
"""


def main(
    filenames: tuple[pathlib.Path, ...],
    *,
    datakey: str = "matrices",
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if len(filenames) < 2:
        log.error("Need two or more datasets to do classification.")
        return 1

    ret = 0
    device = torch.get_default_device()
    log.info("Device: %s", device)

    from nneuroutil.data import random_split

    x_train: list[torch.Tensor] = []
    x_test: list[torch.Tensor] = []
    y_train: list[int] = []
    y_test: list[int] = []

    for label, filename in enumerate(filenames):
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)

        try:
            data = np.load(filename)
        except Exception:
            log.error("Failed to load data from file: '%s'.", filename)
            ret += 1
            continue

        if datakey not in data:
            log.error("File does not contain '%s' dataset: '%s'.", datakey, filename)
            ret += 1
            continue

        matrices = torch.from_numpy(data[datakey]).to(device)
        if matrices.ndim != 3:
            log.error("Expected a 3 dimensional array: %s.", matrices.shape)
            ret += 1
            continue

        errsym = torch.linalg.norm(matrices - torch.transpose(matrices, 1, 2))
        if errsym > 1.0e-8:
            log.error("Matrices do not seem to be symmetric.")
            ret += 1
            continue

        # NOTE: matrices are symmetric with zero diagonl, so we use only the
        # lower triangular values as features for this example
        _, n, _ = matrices.shape
        rows, cols = torch.tril_indices(n, n, offset=-1)
        x = matrices[:, rows, cols]

        # split each dataset for training / testing as is tradition
        (x_train_i,), (x_test_i,) = cast(
            tuple[tuple[torch.Tensor], tuple[torch.Tensor]],
            random_split((x,), (0.6, 0.4)),
        )

        x_train.append(x_train_i)
        y_train.append(label)

        x_test.append(x_test_i)
        y_test.append(label)

    from nneuroutil.classifier import classify_linear_discriminant_analysis

    lda = classify_linear_discriminant_analysis(x_train, y_train)

    for x_test_i, y_test_i in zip(x_test, y_test, strict=True):
        y_test_lda = lda.predict(x_test_i, metric="l2")
        matched = torch.sum(torch.eq(y_test_lda, y_test_i))  # ty: ignore[no-matching-overload]

        log.info(
            "Match %d out of %d (%.2f%%)",
            matched,
            x_test_i.shape[0],
            matched / x_test_i.shape[0] * 100,
        )

    return ret


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
    parser.add_argument("filenames", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default deviced used by the framework",
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
                args.filenames,
                outfile=args.outfile,
                overwrite=args.force,
            )
        )

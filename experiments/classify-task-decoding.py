# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Literal, cast, get_args

import torch

from commonlib import load_feature_datasets
from nneuroutil.helpers import FuzzyChoiceAction, module_logger
from nneuroutil.torch_extras import available_device_names, torch_default_device

log = module_logger("lib.autoencoder_network")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Classify tasks using features computed by make-features.py

Example:

    > {SCRIPT_PATH.name} --method lda dataset-features-matrix.npz
"""

ClassificationMethod = Literal["lda", "svm", "mlp"]


def main(
    filename: pathlib.Path,
    *,
    method: ClassificationMethod = "svm",
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    ret = 0
    device = torch.get_default_device()
    log.info("Device: %s", device)

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    try:
        ds = load_feature_datasets(filename)
    except Exception as exc:
        log.error("Failed to load features from '%s': %s", filename, exc)
        return 1

    from nneuroutil.data import random_split

    x_train: list[torch.Tensor] = []
    y_train: list[int] = []

    x_test: list[torch.Tensor] = []
    y_test: list[int] = []

    idx_train: torch.Tensor | None = None
    idx_test: torch.Tensor | None = None

    for label in range(len(ds.class_names)):
        x = torch.from_numpy(ds.x[:, label, :]).to(device)

        # split each dataset for training / testing as is tradition
        # NOTE: we only do this once because the various datasets seem to be the
        # same subjects in the same order, so this doesn't contaminate
        if idx_train is None and idx_test is None:
            (idx_train,), (idx_test,) = cast(
                tuple[tuple[torch.Tensor], tuple[torch.Tensor]],
                random_split((torch.arange(x.shape[0]),), (0.6, 0.4)),
            )

        x_train.append(x[idx_train])
        y_train.append(label)

        x_test.append(x[idx_test])
        y_test.append(label)

    if method == "lda":
        from nneuroutil.classifier import classify_linear_discriminant_analysis

        lda = classify_linear_discriminant_analysis(x_train, y_train)

        for i, (x_test_i, y_test_i) in enumerate(zip(x_test, y_test, strict=True)):
            y_test_lda = lda.predict(x_test_i, metric="l2")
            matched = torch.sum(torch.eq(y_test_lda, y_test_i))  # ty: ignore[no-matching-overload]

            log.info(
                "%20s: Match %d out of %d (%.2f%%)",
                ds.class_names[i],
                matched,
                x_test_i.shape[0],
                matched / x_test_i.shape[0] * 100,
            )
    elif method == "svm":
        from nneuroutil.classifier import classify_support_vector_machine

        svm = classify_support_vector_machine(x_train, y_train)

        for i, (x_test_i, y_test_i) in enumerate(zip(x_test, y_test, strict=True)):
            y_test_lda = svm.predict(x_test_i)
            matched = torch.sum(torch.eq(y_test_lda, y_test_i))  # ty: ignore[no-matching-overload]

            log.info(
                "%20s: Match %d out of %d (%.2f%%)",
                ds.class_names[i],
                matched,
                x_test_i.shape[0],
                matched / x_test_i.shape[0] * 100,
            )
    elif method == "mlp":
        from torchlib import classify_multilayer_perceptron

        mlp = classify_multilayer_perceptron(
            x_train, y_train, learning_rate_init=1.0e-2, max_iter=400
        )

        for i, (x_test_i, y_test_i) in enumerate(zip(x_test, y_test, strict=True)):
            y_test_mlp = mlp.predict(x_test_i)
            matched = torch.sum(torch.eq(y_test_mlp, y_test_i))

            log.info(
                "%20s: Match %d out of %d (%.2f%%)",
                ds.class_names[i],
                matched,
                x_test_i.shape[0],
                matched / x_test_i.shape[0] * 100,
            )
    else:
        raise ValueError(f"unknown method: {method!r}")

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
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "--device",
        action=FuzzyChoiceAction,
        choices=sorted(available_device_names()),
        default="cpu",
        help="The default deviced used by the framework",
    )
    parser.add_argument(
        "--method",
        choices=get_args(ClassificationMethod),
        default="lda",
        help="The method used to do the classification",
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
                args.filename,
                outfile=args.outfile,
                method=args.method,
                overwrite=args.force,
            )
        )

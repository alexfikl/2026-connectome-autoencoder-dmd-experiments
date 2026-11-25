# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging
import pathlib
import re
from typing import Any

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("summarize_training")

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Summarize training metrics and test losses across the condition directories in
an ae-dmd output folder.

Example:

    > {SCRIPT_PATH.name} ../data/ae-dmd -o summary.tex
"""


def _natural_sort_key(name: str) -> str:
    return re.sub(r"(\d+)", lambda match: match.group(1).zfill(20), name)


def _format_condition(name: str) -> str:
    label = name.removesuffix("_Task").removesuffix("_LR")
    return label.replace("_", " ")


def _format_mean_std(
    values: np.ndarray,
    *,
    scale: float = 1.0,
    decimals: int = 3,
    highlight: bool = False,
) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "--"

    scaled = finite * scale
    mean = float(np.mean(scaled))
    std = float(np.std(scaled, ddof=1)) if scaled.size > 1 else 0.0
    text = f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"
    if highlight:
        return f"\\underline{{{text}}}"
    return text


def _load_condition_metrics(cond_dir: pathlib.Path) -> dict[str, np.ndarray] | None:
    pkl_files = sorted(
        cond_dir.glob("*.pkl"), key=lambda path: _natural_sort_key(path.name)
    )
    if not pkl_files:
        return None

    records: dict[str, list[float]] = {
        "iou": [],
        "dice": [],
        "a_radius": [],
        "train_total": [],
        "val_total": [],
        "test_total": [],
        "test_rec": [],
        "test_lin": [],
        "test_pre": [],
    }

    for path in pkl_files:
        ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)

        final_err = ckpt["final_error_metrics"]
        records["iou"].append(float(final_err["iou"]))
        records["dice"].append(float(final_err["dice"]))
        records["a_radius"].append(float(final_err["a_radius"]))

        losses = ckpt["losses"]
        records["train_total"].append(float(losses[-1, 0, 0]))
        records["val_total"].append(float(losses[-1, 1, 0]))

        loss_test = ckpt["loss_test"]
        records["test_total"].append(float(loss_test[0]))
        records["test_rec"].append(float(loss_test[1]))
        records["test_lin"].append(float(loss_test[2]))
        records["test_pre"].append(float(loss_test[3]))

    return {k: np.asarray(v, dtype=np.float64) for k, v in records.items()}


def _loss_exponent(summaries: dict[str, dict[str, np.ndarray]]) -> int:
    means = []
    for metrics in summaries.values():
        for key in ("train_total", "val_total", "test_total"):
            vals = metrics[key]
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                m = float(np.mean(finite))
                if m > 0:
                    means.append(m)

    if not means:
        return 0
    return int(np.floor(np.log10(float(np.median(means)))))


def format_latex_table(
    summaries: dict[str, dict[str, np.ndarray]],
) -> str:
    exp = _loss_exponent(summaries)
    loss_scale = 10.0 ** (-exp)
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"& \textbf{IoU} & "
            r"\textbf{Train} & \textbf{Validation} & \textbf{Test} & "
            r"\textbf{$\rho(\matr{A}_{\mathrm{dmd}})$} \\"
        ),
    ]
    if exp != 0:
        unit = f"($\\times 10^{{{exp}}}$)"
        lines.append(f" & & {unit} & {unit} & {unit} & \\\\")
    lines.append(r"\midrule")

    best_cond: dict[str, str] = {}
    for key, reducer in (
        ("iou", max),
        ("train_total", min),
        ("val_total", min),
        ("test_total", min),
    ):
        cond_means = {
            name: float(np.mean(m[key][np.isfinite(m[key])]))
            for name, m in summaries.items()
            if np.any(np.isfinite(m[key]))
        }
        if cond_means:
            target = reducer(cond_means.values())
            for name, val in cond_means.items():
                if val == target:
                    best_cond[f"{key}:{name}"] = name

    for cond_name, metrics in summaries.items():
        row = " & ".join([
            _format_condition(cond_name),
            _format_mean_std(metrics["iou"], highlight=f"iou:{cond_name}" in best_cond),
            _format_mean_std(
                metrics["train_total"],
                scale=loss_scale,
                decimals=2,
                highlight=f"train_total:{cond_name}" in best_cond,
            ),
            _format_mean_std(
                metrics["val_total"],
                scale=loss_scale,
                decimals=2,
                highlight=f"val_total:{cond_name}" in best_cond,
            ),
            _format_mean_std(
                metrics["test_total"],
                scale=loss_scale,
                decimals=2,
                highlight=f"test_total:{cond_name}" in best_cond,
            ),
            _format_mean_std(metrics["a_radius"]),
        ])
        lines.append(f"{row} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines)


def main(
    indir: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
) -> int:
    if not indir.is_dir():
        log.error("Input directory does not exist: '%s'.", indir)
        return 1

    cond_dirs = sorted(
        (p for p in indir.iterdir() if p.is_dir()),
        key=lambda path: _natural_sort_key(path.name),
    )

    summaries: dict[str, dict[str, np.ndarray]] = {}
    for cond_dir in cond_dirs:
        metrics = _load_condition_metrics(cond_dir)
        if metrics is None:
            log.warning("No .pkl checkpoints found in '%s'.", cond_dir)
            continue

        summaries[cond_dir.name] = metrics

    if not summaries:
        log.error("No checkpoints found under '%s'.", indir)
        return 1

    latex = format_latex_table(summaries)
    if outfile is not None:
        outfile.write_text(f"{latex}\n", encoding="utf-8")
        log.info("Wrote LaTeX table to '%s'.", outfile)
    else:
        print(latex)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=SCRIPT_LONG_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "indir",
        type=pathlib.Path,
        help="Path to the ae-dmd directory containing per-condition subdirectories",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Optional output .tex file (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    raise SystemExit(main(args.indir, outfile=args.outfile))

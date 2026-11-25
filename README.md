# Koopman Autoencoder Reconstruction of Fractal Connectome Geometry

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)

This code accompanies the paper by the same name. It contains the scripts used to
generate the feature files, figures and tables in the paper.

# Dependencies

The Python dependencies for the project are listed in the `requirements.txt` file.
All dependencies are pinned to the latest version known to work (i.e. as published
with the paper). If you want to reproduce the results exactly, it is highly
recommended to use these dependencies. However, if the results do not reproduce
with newer versions, this is likely a bug!

PyTorch is an optional dependency and is not part of `requirements.txt`. It is
required for training the autoencoder and for all the classification experiments,
so it has to be installed separately (see below).

Besides this, the user is expected to have

* A CUDA-capable GPU. The `train`, `table*`, `figure8` and `figure9` recipes pass
  `--device cuda` to the scripts and will not run as-is on a CPU-only machine.
* [just](https://github.com/casey/just) to run the recipes in the `justfile`.
* [netbrot](https://github.com/alexfikl/netbrot) to render the equi-M sets. A
  prebuilt Windows executable is available on the releases page, otherwise build
  it with `cargo build --release`.
* [Optional] An up to date (>=2025) LaTeX installation. This dependency can be
  removed by setting `text.usetex: False` in `experiments/default.mplstyle`, as it
  is only required for plotting.

# Installation

It is recommended to install the (Python) dependencies in a [virtual
environment](https://docs.python.org/3/library/venv.html) to avoid conflicts
with other packages. The dependencies can then be installed using
[pip](https://pip.pypa.io/en/stable/user_guide/)
```bash
pip install -r requirements.txt
```
or
[uv](https://docs.astral.sh/uv/pip/packages/#installing-packages)
```bash
uv pip sync requirements.txt
```

Then install the PyTorch build matching your GPU and driver (see
[pytorch.org](https://pytorch.org/get-started/locally/) for the right index URL)
```bash
pip install torch
```

# Reproducing the results

## Data

### Human Connectome Project data

The `.mat` files used in this study are derived from the 48-subject subsample of
the HCP S1200 Q4 release analyzed by Rădulescu et al. (2025). They are **not**
redistributed with this repository. To obtain them, please **ask us** (contact
Alexandru Fikl at <alexandru.fikl@e-uvt.ro>, or the corresponding author) and we
will share the same preprocessed files. Users of these datasets must comply with
the [HCP Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms).

Put the files in the `data/` directory with exactly these names
```text
data/Emotion_LR_Task.mat
data/Gambling_LR_Task.mat
data/Language_LR_Task.mat
data/Relational_LR_Task.mat
data/Rest_LR_Task.mat
data/Social_LR_Task.mat
data/WM_LR_Task.mat
data/Structural_LR.mat
```
and convert them to the `.npz` format used by the scripts
```bash
just dataset-hcp
```

### Equi-M set geometry

The equi-M sets are rendered with [netbrot](https://github.com/alexfikl/netbrot).
Each connectome is first converted to a netbrot JSON exhibit (containing the
matrix, the escape radius, and the bounding box) and then rendered to an image.
The companion `2025-fractal-connectomes-paper-experiments` repository contains a
`scripts/generate-exhibits.py` that performs this conversion for the HCP `.mat`
files; the JSON format itself is documented in the netbrot README.

The paper uses the parameters in the following table.

| Parameter | Functional | Structural |
|---|---|---|
| Bounding box | `[-0.03, 0.03] x [-0.03, 0.03]` | `[-1.0, 0.3] x [-0.65, 0.65]` |
| Resolution | `512` | `512` |
| Maximum iterations | `256` | `256` |
| Escape radius | `100` | `100` |

For example
```bash
netbrot --render mandelbrot \
    --resolution 512 --maxit 256 \
    --outfile data/task-emotion/exhibit-000.png \
    exhibit-000.json
```

The rendered images have to be grouped by condition into the directories
```text
data/task-emotion/exhibit-*.png
data/task-gambling/exhibit-*.png
data/task-language/exhibit-*.png
data/task-relational/exhibit-*.png
data/task-rest/exhibit-*.png
data/task-social/exhibit-*.png
data/task-wm/exhibit-*.png
data/structural/exhibit-*.png
```
and converted to the `.npz` format with
```bash
just dataset-equim data
```

This reads the bounding box that netbrot stores in each image's EXIF metadata,
so all images of a condition must use the same bounding box and resolution.

## Features

The feature files are the most expensive part of the pipeline. The autoencoder
features are by far the slowest to compute, so start them first. The full
pipeline is
```bash
# 1. Convert the HCP .mat files (required by everything else)
just dataset-hcp

# 2. Render and convert the equi-M sets (can run while `just train` runs)
just dataset-equim data

# 3. Train an autoencoder for every connectome and compute its features.
#    This trains 8 conditions x 48 matrices and takes the longest.
just train

# 4. Assemble all the feature files used in the paper
just features
```

Note that `just train` consumes the `.npz` files produced by `just dataset-hcp`,
so `just dataset-hcp` has to be run before it. The equi-M conversion in step 2 is
independent of training, so both can run at the same time.

The `just features` target is composed of the per-family invocations below, each
writing `data/dataset-features-<name>.npz`. The input files for each family are
supplied by `just features`; run that target for the complete set.

| Family target | Family | Description |
|---|---|---|
| `just make_features scale` | — | Frobenius norm and spectral radius of `A` (regression target in Figure 7) |
| `just make_features matrix` | A | The 19,900 strictly lower-triangular entries of `A` |
| `just make_features eig` | B | The 200 real eigenvalues of `A` |
| `just make_features global` | C | 24 global graph measures (8 measures x 3 sign variants) |
| `just make_features local` | D | 3,600 node-level graph measures (6 measures x 3 sign variants x 200 nodes) |
| `just make_features image` | E | 1,024 features from the `32 x 32` block-averaged equi-M image |
| `just make_features fourier` | F | 20 boundary Fourier modes of the equi-M contour |
| `just make_features landmarks` | G | 11 cusp-invariant shape descriptors of the equi-M set |
| `just make_features dmd` | H | 400 DMDc descriptors (eigenvalues and mode strengths) |
| `just make_ae_features data/ae-dmd` | I | 211 Koopman autoencoder descriptors (latent eigenvalues, control sensitivity, terminal node magnitudes, attractor energy) |

The `dmd` family is computed directly from the connectome matrices, while the
`ae-dmd` family requires the checkpoints written by `just train` under
`data/ae-dmd/`. The `image`, `landmarks` and `fourier` families require the
equi-M `.npz` files from `just dataset-equim`.

## Figures and tables

Once the feature files exist, the figures and tables can be generated with

```bash
# Figure 2: effective rank of the trajectories
just figure2

# Figures 4, 5 and 6: autoencoder training diagnostics, multi-resolution
# reconstruction and held-out rollouts (requires a trained checkpoint)
just figure456

# Figure 7: scale regression
just figure7

# Figures 8 and 9: classification and mechanism analysis
just table6      # Table 6: structural vs functional classification
just table7a     # Table 7: task decoding
just table7b     # Table 7: subject fingerprinting
just figure8
just figure9

# Table 8: degree-moment regression (printed by the figure9 run)
```

Table 5 (the per-condition autoencoder reconstruction errors) is summarized
directly from the training checkpoints with
```bash
python experiments/summarize-training.py data/ae-dmd -o summary.tex
```

Figure 3 (the autoencoder architecture) is a schematic and is not generated by
these scripts. Figure 8 depends on `table7a` and `table7b`, and Figure 9 also
prints the values reported in Table 8. The `figure456` recipe plots a fixed
representative checkpoint (Emotion, `dataidx 45`); if you trained your own
models, update the checkpoint path in the recipe to the corresponding file under
`data/ae-dmd/Emotion_LR_Task/`.

# License

The MATLAB mat files from the `data` folder are derived from MRI and fMRI
datasets from the [Human Connectome Project](https://www.humanconnectome.org/).
Users of these datasets must comply with the HCP Data Use Terms, as appropriate.
They are licensed under the CC-BY-NC-4.0 (Non-commercial) license.

The code itself and other files in this repository are provided under the
standard MIT license.

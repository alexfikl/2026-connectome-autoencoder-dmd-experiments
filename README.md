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

The pinned PyTorch wheel is the default PyPI build, which is CUDA-enabled on
Linux. If your GPU needs a different CUDA toolkit, or you want a CPU-only build,
install the matching wheel from
[pytorch.org](https://pytorch.org/get-started/locally/) after syncing; this
replaces the pinned build.

# Reproducing the results

## Data

### Human Connectome Project data

The `.mat` files used in this study are derived from the HCP S1200 Q4 release
and are **not** redistributed with this repository. To obtain them, please
**ask us** and we will share the same preprocessed files. Users of these
datasets must comply with the [HCP Data Use
Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms).

Put the files in the `data/` directory with the provided names and convert them
to the `.npz` format used by the scripts
```bash
just dataset-hcp
```

### Equi-M set geometry

The equi-M sets are rendered with [netbrot](https://github.com/alexfikl/netbrot).
Each connectome is first converted to a netbrot JSON exhibit (containing the
matrix, the escape radius, and the bounding box) and then rendered to an image.
The `scripts/generate-exhibits.py` script performs the conversion from the HCP `.mat`
files to the JSON format.

For example
```bash
netbrot \
    --render mandelbrot \
    --resolution 512 --maxit 256 \
    --outfile data/task-emotion/exhibit-000.png \
    exhibit-000.json
```

The resulting images are converted to the `.npz` format with
```bash
just dataset-equim data
```

## Features

The feature files are the most expensive part of the pipeline. The autoencoder
features are by far the slowest to compute, so start them first. The full
pipeline is
```bash
# 1. Convert the HCP .mat files
just dataset-hcp

# 2. Convert the equi-M set images
just dataset-equim data

# 3. Train an autoencoder for every connectome and compute its features.
#    This trains 8 conditions x 48 matrices and takes the longest.
just train

# 4. Assemble all the feature files used in the paper
just features
```

## Figures and tables

Once the feature files exist, the figures and tables from the paper can be
generated with

```bash
# Figure 2: effective rank of the trajectories
just figure2

# Figures 4, 5 and 6: autoencoder training diagnostics, multi-resolution
# reconstruction and held-out rollouts (requires a trained checkpoint)
just table5
just figure456

# Figure 7: scale regression
just figure7

# Figures 8 and 9: classification and mechanism analysis
just table6      # Table 6: structural vs functional classification
just table7a     # Table 7: task decoding
just table7b     # Table 7: subject fingerprinting
just figure8
just figure9
```

# License

The MATLAB mat files from the `data` folder are derived from MRI and fMRI
datasets from the [Human Connectome Project](https://www.humanconnectome.org/).
Users of these datasets must comply with the HCP Data Use Terms, as appropriate.
They are licensed under the CC-BY-NC-4.0 (Non-commercial) license.

The code itself and other files in this repository are provided under the
standard MIT license.

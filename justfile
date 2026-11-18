PYTHON := "nice python"
EXT := ".jpg"

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc("Reformat all source code")]
format: isort black pyproject justfmt

[doc("Run ruff isort fixes over the source code")]
isort:
    ruff check --fix --select=I experiments/*.py scripts/*.py
    ruff check --fix --select=RUF022 experiments/*.py scripts/*.py
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc("Run ruff format over the source code")]
black:
    ruff format experiments/*.py scripts/*.py
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc("Run pyproject-fmt over the configuration")]
pyproject:
    {{ PYTHON }} -m pyproject_fmt \
        --indent 4 --max-supported-python "3.14" \
        pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc("Run just --fmt over the justfile")]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}

# {{{ linting

[doc("Run all linting checks over the source code")]
lint: typos reuse ruff ty

[doc("Run typos over the source code and documentation")]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc('Check REUSE license compliance')]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc("Run ruff checks over the source code")]
ruff:
    ruff check experiments/*.py scripts/*.py
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run ty checks over the source code")]
ty:
    ty check experiments/*.py scripts/*.py
    @echo -e "\e[1;32mty clean!\e[0m"

# }}}

# {{{ develop

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

[doc("Remove all generated files")]
purge:
    rm -rf *.png *.jpg *.pdf
    rm -rf .ruff_cache __pycache__

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version '3.11' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt

# }}}

# {{{ datasets

[private]
mat2npz scan_type filename *ARGS:
    {{ PYTHON }} scripts/mat2npz.py --force \
        --variable-name 'w:matrices' \
        --transpose \
        --scan-type {{ scan_type }} \
        {{ ARGS }} \
        {{ filename }}

[doc("Convert HCP datasets")]
dataset-hcp:
    @just mat2npz fmri data/Emotion_LR_Task.mat
    @just mat2npz fmri data/Gambling_LR_Task.mat
    @just mat2npz fmri data/Language_LR_Task.mat
    @just mat2npz fmri data/Relational_LR_Task.mat
    @just mat2npz fmri data/Rest_LR_Task.mat
    @just mat2npz fmri data/Social_LR_Task.mat
    @just mat2npz fmri data/WM_LR_Task.mat
    @just mat2npz mri data/Structural_LR.mat \
        --normalize 'max' --bbox '-0.8' '0.3' '-0.55' '0.55'

[private]
img2npz scan_type root *ARGS:
    {{ PYTHON }} scripts/img2npz.py --force \
        --outfile data/$(basename {{ root }})-fractal.npz \
        --scan-type {{ scan_type }} \
        {{ ARGS }} \
        {{ root }}/exhibit-*.png

[doc("Convert rendered equi-M images to datasets")]
dataset-equim root:
    @just img2npz fmri {{ root }}/task-emotion
    @just img2npz fmri {{ root }}/task-gambling
    @just img2npz fmri {{ root }}/task-language
    @just img2npz fmri {{ root }}/task-relational
    @just img2npz fmri {{ root }}/task-rest
    @just img2npz fmri {{ root }}/task-social
    @just img2npz fmri {{ root }}/task-wm
    @just img2npz mri {{ root }}/structural

[private]
make_features name *FILES:
    {{ PYTHON }} experiments/make-features.py --force \
        --outfile data/dataset-features-{{ name }}.npz \
        --features {{ name }} \
        {{ FILES }} || true

[doc("Construct DMD autoencoder features from trained models")]
make_ae_features dirname:
    {{ PYTHON }} experiments/nn-autoencoder-dmd-features.py --force \
        --outfile data/dataset-features-ae-dmd.npz \
        merge {{ dirname }}/*

[doc("Construct all features")]
features:
    # @just make_features matrix data/*_LR_Task.npz data/Structural_LR.npz
    # @just make_features eig data/*_LR_Task.npz data/Structural_LR.npz
    @just make_features local data/*_LR_Task.npz data/Structural_LR.npz
    # @just make_features global data/*_LR_Task.npz data/Structural_LR.npz
    # @just make_features dmd data/*_LR_Task.npz data/Structural_LR.npz
    # @just make_features geometry data/task-*-fractal.npz data/structural-fractal.npz
    # @just make_ae_features data/ae-dmd

# }}}

# {{{ autoencoder-dmd

[private]
train_autoencoder_dmd filename *ARGS:
    stem=$(basename {{ filename }} .npz); \
    mkdir -p data/ae-dmd/$stem; \
    for idx in $(seq -w 0 47); do \
        out=data/ae-dmd/$stem/$stem-m$idx; \
        {{ PYTHON }} experiments/nn-autoencoder-dmd.py \
            --device cuda --force -o $out.npz \
            --seed 0 \
            --resolution 64 --maxit 32 \
            --epochs 100 --batch-size 1024 \
            {{ ARGS }} \
            train --dataidx $idx {{ filename }} || exit 1; \
    done; \
    {{ PYTHON }} experiments/nn-autoencoder-dmd-features.py \
        --device cuda --force \
        compute data/ae-dmd/$stem/*.pkl

[doc("Train a DMD autoencoder and compute its features for each of the 48 matrices")]
train:
    @just train_autoencoder_dmd data/Emotion_LR_Task.npz
    @just train_autoencoder_dmd data/Gambling_LR_Task.npz
    @just train_autoencoder_dmd data/Language_LR_Task.npz
    @just train_autoencoder_dmd data/Relational_LR_Task.npz
    @just train_autoencoder_dmd data/Rest_LR_Task.npz
    @just train_autoencoder_dmd data/Social_LR_Task.npz
    @just train_autoencoder_dmd data/WM_LR_Task.npz
    @just train_autoencoder_dmd data/Structural_LR.npz \
        --resolution 128 \
        --bbox '-0.8' '0.3' '-0.55' '0.55'

[doc("Run or resume ablation studies for a connectome matrix (default: dataidx 0)")]
ablation filename dataidx="0" *ARGS:
    {{ PYTHON }} experiments/nn-autoencoder-dmd-sweep.py \
        --device cuda \
        --resolution 64 --maxit 32 --epochs 60 --batch-size 1024 --seed 0 \
        --study all \
        {{ ARGS }} \
        --dataidx {{ dataidx }} {{ filename }}

# }}}

# {{{ figures

[private]
figure2_run filename *ARGS:
    {{ PYTHON }} experiments/visualize-effective-rank.py \
        --maxit 32 --resolution 64 --escape-radius 100 {{ ARGS }} \
        {{ filename }}

[doc("Generate plots for Figure 2")]
figure2:
    @just figure2_run data/Rest_LR_Task.npz --bbox '-0.03' '0.03' '-0.03' '0.03'
    @just figure2_run data/Emotion_LR_Task.npz --bbox '-0.03' '0.03' '-0.03' '0.03'
    @just figure2_run data/Structural_LR.npz --bbox '-0.8' '0.3' '-0.55' '0.55'

# }}}

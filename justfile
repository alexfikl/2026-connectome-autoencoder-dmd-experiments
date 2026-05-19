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
    uv pip compile --upgrade --universal --python-version '3.10' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt

# }}}
# {{{ datasets
DATASET_URL := 'https://raw.githubusercontent.com/alexfikl/2025-fractal-connectomes-paper-experiments/refs/heads/main/data'

[private]
download_dataset name:
    curl -o "data/{{ name }}.mat" "{{ DATASET_URL }}/{{ name }}.mat"
    {{ PYTHON }} scripts/mat2npz.py \
        --variable-name 'w:matrices' \
        --transpose \
        "data/{{ name }}.mat"

[doc("Download datasets")]
dataset:
    mkdir -p data
    @just download_dataset Emotion_LR_Task
    @just download_dataset Rest_LR_Task

# }}}

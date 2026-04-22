# Setting Up the CUDA Environment

To follow along this CUDA branch, you will need a Linux or WSL2 environment with an NVIDIA GPU. We manage the codebase with pdm.

This branch is being migrated from the original MLX course. The CUDA implementation is not complete yet, so some commands are marked as TODO.

## Install pdm

Please follow the [official guide](https://pdm-project.org/en/latest/) to install pdm.

You can also install pdm in a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install pdm ninja
```

## Clone the Repository

```bash
git clone https://github.com/fayechen/tiny-llm.git
cd tiny-llm
```

The repository is organized as follows:

```
src/tiny_llm -- your implementation
src/tiny_llm_ref -- reference implementation from the original MLX course
src/tiny_llm_torch_ref -- migrated Torch/CUDA reference implementation
tests/ -- unit tests for your implementation
tests_refsol/ -- unit tests for the reference implementation
tests_torch_ref/ -- Torch/CUDA tests for migrated chapters
book/ -- the book
```

We keep the original MLX reference implementation while the Torch/CUDA version is built out.

## Install Dependencies

First check the CUDA environment:

```bash
nvidia-smi
nvcc --version
```

Install PyTorch with CUDA support:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining Python dependencies:

```bash
python -m pip install transformers accelerate safetensors tokenizers sentencepiece huggingface-hub numpy pytest pytest-benchmark ruff torchao torchtune
```

The current `pyproject.toml` still contains MLX dependencies from the original project. We will split the CUDA dependencies later.

## Check the Installation

```bash
pdm run check-cuda-installation
```

This is only an environment check. It imports PyTorch, runs a CPU tensor add, runs a CUDA tensor add, and prints the CUDA device.

## Run Unit Tests

For the migrated CUDA/Torch chapters, run the Torch reference tests directly:

```bash
.venv/bin/pytest -q tests_torch_ref/test_week_1_day_1.py
.venv/bin/pytest -q tests_torch_ref/test_week_1_day_2.py
```

The original book workflow command is still available:

```bash
pdm run test
```

That command still follows the original book workflow and uses `tests/` plus `tests_refsol`. For the CUDA branch, use `tests_torch_ref` for migrated chapters.

## Download the Model Parameters

For the CUDA branch, use the regular Hugging Face model weights instead of the MLX weights.

Follow the guide of [this page](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to install the Hugging Face CLI (`hf`).

The model parameters are hosted on Hugging Face. Once you authenticated your CLI with the credentials, you can download them with:

```bash
hf auth login
hf download Qwen/Qwen2-0.5B-Instruct
```

The original MLX course used models such as `Qwen/Qwen2-0.5B-Instruct-MLX`. Those are not the default model weights for this CUDA branch.

## Run the Model

The original command was:

```bash
pdm run main --solution ref --loader week1
```

That command still uses `mlx_lm`, MLX model weights, and the original MLX reference implementation. It is not the CUDA branch smoke test.

The CUDA model-running path is not wired up yet in this branch.

In week 2, we will write some kernels in C++/CUDA. The original course used C++/Metal for MLX; this CUDA branch will replace that path with CUDA-oriented tooling.

{{#include copyright.md}}

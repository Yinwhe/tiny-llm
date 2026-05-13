# Setting Up the CUDA Environment

To follow this course, you will need a Linux or WSL2 environment with an NVIDIA GPU. We manage the codebase with pdm.

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
src/tiny_llm -- your Torch/CUDA learner implementation
src/tiny_llm_ref -- Torch/CUDA reference implementation
tests/ -- unit tests for your implementation
tests_refsol/ -- Torch/CUDA reference tests
book/ -- the book
```

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

## Check the Installation

```bash
pdm run check-installation
```

This checks PyTorch on CPU and CUDA, and prints the detected CUDA device.

## Run Unit Tests

Use the standard `pdm` workflow throughout this repository:

```bash
pdm run test --week 1 --day 1
pdm run test --week 1 --day 2
```

To run a specific task with a pytest filter:

```bash
pdm run test --week 1 --day 1 -- -k task_1
```

To run the Torch/CUDA reference tests directly:

```bash
pdm run test-refsol --week 1 --day 1
```

The `pdm run test` command copies the corresponding file from `tests_refsol/` into `tests/` and runs it against `src/tiny_llm`.

## Download the Model Parameters

Use the Hugging Face model weights for this course.

Follow the guide of [this page](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to install the Hugging Face CLI (`hf`).

The model parameters are hosted on Hugging Face. Once you authenticated your CLI with the credentials, you can download them with:

```bash
hf auth login
hf download Qwen/Qwen2-0.5B-Instruct
```

## Run the Model

Run the reference implementation:

```bash
pdm run main --solution ref --loader week1 --model qwen2-0.5b --device gpu
```

Run your own implementation once you have completed the corresponding chapter:

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen2-0.5b --device gpu
```

In week 2, we will write some kernels in C++/CUDA.

{{#include copyright.md}}

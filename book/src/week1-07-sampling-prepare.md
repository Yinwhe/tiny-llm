# Week 1 Day 7: Sampling and Preparing for Week 2

In day 7, we will implement various sampling strategies. And we will get you prepared for week 2.

## Task 1: Sampling

We implemented the default greedy sampling strategy in the previous day. In this task, we will implement the temperature,
top-k, and top-p (nucleus) sampling strategies.

```
src/tiny_llm/sampler.py
```

- 📚 [PyTorch categorical distribution](https://pytorch.org/docs/stable/distributions.html#categorical)

**Temperature Sampling**

The first sampling strategy is the temperature sampling. When `temp=0`, we use the default greedy strategy. When it is
larger than 0, we will randomly select the next token based on the logprobs. The temperature parameter scales the distribution.
When the value is larger, the distribution will be more uniform, making the lower probability token more likely to be
selected, and therefore making the model more creative.

To implement temperature sampling, simply divide the logprobs by the temperature and use `torch.distributions.Categorical` to
randomly select the next token.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen2-0.5b \
  --device gpu --sampler-temp 0.5
```

**Top-k Sampling**

In top-k sampling, we will only keep the top-k tokens with the highest probabilities before sampling the probabilities.
This is done before the final temperature scaling.

You can use `torch.topk` to get the indices of the top-k elements, and then,
mask those logprobs outside the top-k with `-torch.inf`. After that, do temperature sampling.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen2-0.5b \
  --device gpu --sampler-temp 0.5 --sampler-top-k 10
```

**Top-p (Nucleus) Sampling**

In top-p (nucleus) sampling, we will only keep the top-p tokens with the highest cumulative probabilities before sampling
the probabilities. This is done before the final temperature scaling.

There are multiple ways of implementing it. One way is to first use `torch.sort` to sort the logprobs (from highest
probability to lowest), and then, do a `torch.cumsum` over the probabilities to get the cumulative probabilities. Then,
mask those logprobs outside the top-p with `-torch.inf`. After that, do temperature sampling.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen2-0.5b \
  --device gpu --sampler-temp 0.5 --sampler-top-p 0.9
```

## Task 2: Prepare for Week 2

In week 2, we will optimize the serving infrastructure of the Qwen2 model. We will write some C++/CUDA extension code
to make some operations run faster. You will need a working CUDA toolchain and CMake to compile the extensions.

1.  **Install CUDA Toolkit:**
    Install the CUDA toolkit that matches your NVIDIA driver and PyTorch CUDA build.
2.  **Install a C++ compiler toolchain:**
    Ensure `gcc`/`g++` are available in your environment.
3.  **Check CUDA compiler:**
    Open your Terminal and run:
    ```bash
    nvcc --version
    ```
4.  **Install CMake:**
    Install CMake using your system package manager.
5.  **Check GPU availability in PyTorch:**
    ```bash
    pdm run check-installation
    ```

You can test your installation by compiling the code in `src/extensions_torch`:

```bash
pdm run build-ext-torch
pdm run build-ext-torch-ref
```

Both commands should complete without build errors.

If you are not familiar with C++ or CUDA programming, we also suggest doing some small exercises to get familiar with
them. You can implement some element-wise operations like `exp`, `sin`, `cos` and replace the PyTorch ones in your model
implementation.

That's all for week 1! We have implemented all the components to serve the Qwen2 model. Now we are ready to start week 2,
where we will optimize the serving infrastructure and make it run faster on your CUDA device.

{{#include copyright.md}}

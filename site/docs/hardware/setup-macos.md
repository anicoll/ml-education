---
sidebar_position: 6
title: "Setup: macOS on Apple Silicon"
---

# Setup: macOS on Apple Silicon

There's no driver to install on a Mac: Metal support is built in. Read [Apple Silicon](./apple-silicon.md) first for what to expect.

## Step 1: Command-line tools and Homebrew

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

## Step 2: Python environment

Make sure you use a native **arm64** Python, not an Intel one running under Rosetta. uv installs the right one automatically:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/code/ml-education
uv venv --python 3.12
source .venv/bin/activate
python -c "import platform; print(platform.machine())"   # should print arm64
```

## Step 3: PyTorch (with MPS)

The standard macOS wheels include MPS support. No special index URL needed:

```bash
uv pip install torch torchvision
```

## Step 4: MLX

```bash
uv pip install mlx mlx-lm
```

## Step 5: Verify

```python
import time, torch
print(torch.__version__)
print(torch.backends.mps.is_available())     # True

x = torch.randn(4096, 4096, device="mps")
torch.mps.synchronize()
t = time.time()
for _ in range(50):
    y = x @ x
torch.mps.synchronize()
print(f"{50 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOPS fp32")
```

And MLX:

```python
import mlx.core as mx
a = mx.random.normal((4096, 4096))
b = a @ a
mx.eval(b)            # MLX is lazy: work only happens when you evaluate
print(mx.default_device())   # Device(gpu, 0)
```

If you also have an NVIDIA PC, run the same matmul benchmark there (from the [Ubuntu setup](./setup-linux.md)) and compare. It's a quick, concrete feel for the two machines.

## Step 6: The rest of the stack

Install everything in `notebooks/requirements.txt` **except bitsandbytes**, which needs CUDA:

```bash
grep -v bitsandbytes notebooks/requirements.txt > /tmp/requirements-mac.txt
uv pip install -r /tmp/requirements-mac.txt
```

Add this to your shell profile (`~/.zshrc`) so missing MPS operations fall back to the CPU instead of erroring:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Step 7: Ollama or LM Studio

Install [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai). Both use the Mac GPU automatically. LM Studio can also run MLX-format models.

## Common problems

| Symptom | Fix |
| --- | --- |
| `torch.backends.mps.is_available()` is `False` | You're probably on an x86 Python under Rosetta. Check `platform.machine()` prints `arm64`, then recreate the environment. |
| `NotImplementedError` for an operation on MPS | Set `PYTORCH_ENABLE_MPS_FALLBACK=1`. |
| Errors mentioning `bitsandbytes` or CUDA | That code path is CUDA-only. Use MLX for quantised models and QLoRA, or run that notebook on an NVIDIA machine. |
| A big model loads but generation crawls | Memory pressure is high and macOS is swapping. Close apps, shorten the context, or use a smaller quantisation. |
| Model too big for the default GPU memory limit | Raise it with `sudo sysctl iogpu.wired_limit_mb=...` (see [Apple Silicon](./apple-silicon.md)). |

:::tip[Try it]
Save the outputs of both verification scripts to `notes/setup.md`, along with your macOS, PyTorch and MLX versions.
:::

## Check your understanding

<MultipleChoice
  id="macos-1"
  question="`torch.backends.mps.is_available()` returns False on an Apple Silicon Mac. What's the most likely cause?"
  options={["MPS needs a driver install", "You're running an Intel (x86) Python under Rosetta; check that platform.machine() prints arm64", "The Mac has too little memory", "PyTorch doesn't support macOS"]}
  answer={1}
  explanation="Recreate the environment with a native arm64 Python (uv does this automatically)."
/>

<MultipleChoice
  id="macos-2"
  question="What does `PYTORCH_ENABLE_MPS_FALLBACK=1` do?"
  options={["Runs everything on the CPU", "Runs operations that MPS doesn't support yet on the CPU instead of raising an error", "Enables CUDA on the Mac", "Speeds up training"]}
  answer={1}
  explanation="Most operations still run on the GPU; only the missing ones fall back."
/>

<OpenQuestion
  id="macos-3"
  question="In the MLX check, why is `mx.eval(b)` needed before the work actually happens?">

MLX is **lazy**: when you write `b = a @ a`, it only records that `b` should be the product of `a` with itself, without computing it. The actual work happens when the result is needed, for example when you print it, convert it to NumPy, or explicitly call `mx.eval(b)`.

That's why benchmarks in MLX must call `mx.eval(...)` inside the timed section; otherwise you'd only be timing how long it takes to *describe* the computation. Laziness lets MLX fuse and optimise operations before running them.

</OpenQuestion>

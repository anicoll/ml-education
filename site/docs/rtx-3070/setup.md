---
sidebar_position: 3
title: Setting up your machine
---

# Setting up your machine

The goal: a Python environment where `torch.cuda.is_available()` returns `True`. Pick the path that matches your OS.

## Which path?

| Setup | Pros | Cons |
| --- | --- | --- |
| **Linux (native)** | Everything just works; best performance. | Need a Linux install. |
| **Windows + WSL2** | Linux tooling on a Windows PC. Supports bitsandbytes, `torch.compile`, vLLM, Unsloth. | One-time setup; files are best kept inside the Linux filesystem. |
| **Windows (native)** | No extra setup for basic PyTorch. | Some LLM tooling is Linux-first and may be rough or unsupported. |

**Recommendation:** if you're on Windows, use **WSL2** for the LLM parts of this course. Native Windows is fine for Parts 1–4.

## Step 1: NVIDIA driver

Install the latest **Game Ready** or **Studio** driver from NVIDIA for your 3070. On Windows, install it in Windows only, even if you'll use WSL2: the Windows driver provides CUDA to WSL automatically. **Do not** install a Linux NVIDIA driver inside WSL.

Check it:

```bash
nvidia-smi
```

You should see your RTX 3070, the driver version, and the highest CUDA version the driver supports.

## Step 2 (Windows only): WSL2

In an administrator PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot, open "Ubuntu" from the Start menu, create a user, then check that `nvidia-smi` works inside WSL as well.

Keep your projects inside the Linux filesystem (for example `~/code/ml-education`), not under `/mnt/c/...`; file access across the boundary is much slower. VS Code's "WSL" extension lets you edit them from Windows seamlessly.

## Step 3: Python environment

Use **uv** (fast and simple) or **conda/miniforge**. With uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/code/ml-education
uv venv --python 3.12
source .venv/bin/activate
```

## Step 4: PyTorch with CUDA

Get the exact install command for your setup from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). It will look like:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

PyTorch wheels bundle their own CUDA runtime, so you **don't need to install the CUDA toolkit separately** for normal use. You only need it if you compile custom CUDA code.

## Step 5: Verify

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())             # True
print(torch.cuda.get_device_name(0))         # NVIDIA GeForce RTX 3070
print(torch.cuda.is_bf16_supported())        # True on Ampere

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
import time; t = time.time()
for _ in range(50):
    y = x @ x
torch.cuda.synchronize()
print(f"{50 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOPS fp32")
```

## Step 6: The rest of the stack

```bash
uv pip install numpy pandas matplotlib scikit-learn jupyterlab tensorboard
uv pip install transformers datasets peft trl accelerate bitsandbytes
```

Then check bitsandbytes can see your GPU:

```bash
python -m bitsandbytes
```

## Jupyter

```bash
jupyter lab
```

Or open the `notebooks/` folder in VS Code with the Python and Jupyter extensions and select the `.venv` interpreter.

## Common problems

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` is `False` | You installed the CPU-only wheel. Reinstall with the CUDA index URL from pytorch.org. |
| `nvidia-smi` not found in WSL | Update the Windows NVIDIA driver and run `wsl --update`. |
| bitsandbytes errors on Windows native | Use WSL2. |
| Out of memory immediately | Something else is using the GPU. Check `nvidia-smi` for other processes. |
| Very slow data loading in WSL | Move the data from `/mnt/c/...` into the Linux home directory. |

:::tip Try it
Get the verification script above running and save its output to `notes/setup.md`, along with the exact versions of your driver, PyTorch and Python. Future you will thank you when something breaks after an update.
:::

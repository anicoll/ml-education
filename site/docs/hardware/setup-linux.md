---
sidebar_position: 5
title: "Setup: native Ubuntu"
---

# Setup: native Ubuntu (NVIDIA GPU)

Native Linux is the smoothest platform for CUDA work: every library in this course supports it. These steps are for Ubuntu 24.04 or newer. If you're using WSL2 on Windows, start with [Windows + WSL2](./setup-windows-wsl.md), which sends you back here for the shared steps.

The goal: a Python environment where `torch.cuda.is_available()` returns `True`.

## Step 1: NVIDIA driver (native Linux only, skip in WSL)

Let Ubuntu pick the recommended proprietary driver:

```bash
sudo apt update
ubuntu-drivers devices          # shows the recommended driver
sudo ubuntu-drivers install     # installs it
sudo reboot
```

If Secure Boot is on, the installer asks you to set a password, and after the reboot a blue "MOK management" screen asks for it again to enrol the driver's signing key. If you skip that screen, the driver won't load.

Check it:

```bash
nvidia-smi
```

You should see your GPU's name, the driver version, and the highest CUDA version the driver supports.

## Step 2: Build tools

```bash
sudo apt install -y build-essential git curl
```

## Step 3: Python environment

Use **uv** (fast and simple) or **conda/miniforge**. With uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/code/ml-education
uv venv --python 3.12
source .venv/bin/activate
```

## Step 4: PyTorch with CUDA

Get the exact install command for your setup from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). It will look something like:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

PyTorch wheels bundle their own CUDA runtime, so you **don't need to install the CUDA toolkit separately** for normal use. You only need it if you compile custom CUDA code. The driver just has to support at least the CUDA version of the wheel (the top right of `nvidia-smi`).

## Step 5: Verify

```python
import time, torch
print(torch.__version__)
print(torch.cuda.is_available())             # True
print(torch.cuda.get_device_name(0))         # e.g. NVIDIA GeForce RTX 4060
print(torch.cuda.is_bf16_supported())        # True on RTX 30-series (Ampere) and newer

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
t = time.time()
for _ in range(50):
    y = x @ x
torch.cuda.synchronize()
print(f"{50 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOPS fp32")
```

## Step 6: The rest of the stack

```bash
uv pip install -r notebooks/requirements.txt
python -m bitsandbytes      # checks bitsandbytes can see the GPU
```

Then start Jupyter with `jupyter lab`, or open the repo in VS Code with the Python and Jupyter extensions and pick the `.venv` interpreter.

## Common problems

| Symptom | Fix |
| --- | --- |
| `nvidia-smi` says it can't communicate with the driver | Usually Secure Boot blocked the driver. Re-run the install and complete the MOK enrolment, or disable Secure Boot. |
| `torch.cuda.is_available()` is `False` | You installed the CPU-only wheel. Reinstall with the CUDA index URL from pytorch.org. |
| Driver broke after a kernel update | `sudo ubuntu-drivers install` again, then reboot. |
| Out of memory immediately | Something else is using the GPU. Check `nvidia-smi` for other processes, including your desktop. |

:::tip[Try it]
Get the verification script running and save its output to `notes/setup.md`, along with your driver, PyTorch and Python versions. Future you will thank you when something breaks after an update.
:::

## Check your understanding

<MultipleChoice
  id="linux-1"
  question="`torch.cuda.is_available()` returns False, but `nvidia-smi` works. What's the most likely cause?"
  options={["The GPU is broken", "You installed the CPU-only PyTorch wheel; reinstall using the CUDA index URL from pytorch.org", "Secure Boot", "Not enough RAM"]}
  answer={1}
  explanation="If nvidia-smi works, the driver is fine. The PyTorch build is the problem."
/>

<MultipleChoice
  id="linux-2"
  question="Do you need to install the CUDA toolkit separately to use PyTorch on the GPU?"
  options={["Yes, always", "No: PyTorch wheels bundle their own CUDA runtime; you only need a recent enough driver", "Only on Ubuntu", "Only for inference"]}
  answer={1}
  explanation="You only need the full toolkit if you compile custom CUDA code."
/>

<OpenQuestion
  id="linux-3"
  question="After a reboot, `nvidia-smi` says it can't communicate with the driver. What do you check, and in what order?">

1. **Secure Boot / MOK enrolment.** The most common cause. If you skipped the blue "MOK management" screen after installing the driver, the kernel won't load the unsigned driver. Re-run `sudo ubuntu-drivers install`, reboot, and complete the enrolment with the password you set (or disable Secure Boot in the BIOS).
2. **Kernel update.** If the system updated its kernel, the driver module may not have rebuilt. Run `sudo ubuntu-drivers install` again and reboot.
3. **Check what's installed:** `ubuntu-drivers devices` shows the recommended driver; make sure that's the one installed.

Once `nvidia-smi` shows your GPU again, re-run the verification script from Step 5 to confirm PyTorch can see it.

</OpenQuestion>

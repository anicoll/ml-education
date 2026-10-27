---
sidebar_position: 4
title: "Setup: Windows + WSL2"
---

# Setup: Windows + WSL2 (RTX 3070)

WSL2 runs a real Ubuntu inside Windows, with full access to your NVIDIA GPU. It gives you the Linux ML tooling (bitsandbytes, `torch.compile`, vLLM, Unsloth) without dual-booting. This is the recommended way to use the 3070 from Windows.

| Setup | Pros | Cons |
| --- | --- | --- |
| **Windows + WSL2** | Linux tooling on a Windows PC. Everything in the course works. | Files are best kept inside the Linux filesystem. |
| **Windows (native)** | No extra setup for basic PyTorch. | Some LLM tooling is Linux-first and may be rough or unsupported. Fine for Parts 1–4. |

## Step 1: NVIDIA driver on Windows

Install the latest **Game Ready** or **Studio** driver from NVIDIA for your 3070, **in Windows only**. The Windows driver provides CUDA to WSL automatically. **Do not** install a Linux NVIDIA driver inside WSL; it will break GPU access.

## Step 2: Install WSL2 and Ubuntu

Skip this if you already have it. In an administrator PowerShell:

```powershell
wsl --install -d Ubuntu
wsl --update
```

Reboot, open "Ubuntu" from the Start menu and create a user. Check the GPU is visible inside WSL:

```bash
nvidia-smi
```

## Step 3: Keep your files in Linux

Clone the repo inside the Linux filesystem, for example `~/code/ml-education`, not under `/mnt/c/...`. File access across the Windows/Linux boundary is much slower, which hurts data loading badly. VS Code's **WSL** extension lets you edit these files from Windows seamlessly (run `code .` from the WSL terminal).

## Step 4: Give WSL enough memory

By default WSL2 can use about half your PC's RAM. For data processing or llama.cpp CPU offload you may want more. Create `C:\Users\<you>\.wslconfig`:

```text
[wsl2]
memory=24GB
```

Then run `wsl --shutdown` in PowerShell and reopen Ubuntu.

## Step 5: Everything else

From here, it's the same as native Linux. Follow [Native Ubuntu](./setup-linux.md) from **Step 2 (build tools)** onwards, skipping the driver step.

## WSL-specific problems

| Symptom | Fix |
| --- | --- |
| `nvidia-smi` not found or fails in WSL | Update the Windows NVIDIA driver and run `wsl --update`. Make sure you didn't install a Linux driver inside WSL. |
| Very slow data loading | Move the data from `/mnt/c/...` into your Linux home directory. |
| WSL runs out of memory or gets killed | Raise `memory` in `.wslconfig`. |
| Disk filling up | WSL's virtual disk grows but doesn't shrink on its own. Clean up `~/.cache/huggingface` and old checkpoints. |

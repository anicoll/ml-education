---
sidebar_position: 4
title: "Setup: Windows + WSL2"
---

# Setup: Windows + WSL2 (NVIDIA GPU)

WSL2 runs a real Ubuntu inside Windows, with full access to your NVIDIA GPU. It gives you the Linux ML tooling (bitsandbytes, `torch.compile`, vLLM, Unsloth) without dual-booting. This is the recommended way to use an NVIDIA GPU for ML from Windows.

| Setup | Pros | Cons |
| --- | --- | --- |
| **Windows + WSL2** | Linux tooling on a Windows PC. Everything in the course works. | Files are best kept inside the Linux filesystem. |
| **Windows (native)** | No extra setup for basic PyTorch. | Some LLM tooling is Linux-first and may be rough or unsupported. Fine for Parts 1–4. |

## Step 1: NVIDIA driver on Windows

Install the latest **Game Ready** or **Studio** driver from NVIDIA for your card, **in Windows only**. The Windows driver provides CUDA to WSL automatically. **Do not** install a Linux NVIDIA driver inside WSL; it will break GPU access.

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

By default WSL2 can use about half your PC's RAM. For data processing or llama.cpp CPU offload you may want more. Create `.wslconfig` in your Windows user folder (`%UserProfile%\.wslconfig`), giving WSL as much as your PC can spare:

```text
[wsl2]
memory=24GB
```

Then run `wsl --shutdown` in PowerShell and reopen Ubuntu.

## Step 5: Everything else

From here, it's the same as native Linux. Follow [Native Ubuntu](./setup-linux.md) from **Step 2 (build tools)** onwards, skipping the driver step.

## A native Windows environment (for Trackmania)

Most of the course runs in WSL. The exception is the [Trackmania capstone](../reinforcement-learning/capstone-trackmania.md): the game only runs on Windows, and the agent has to capture its window and send controller input, which a WSL environment can't do. For that project, set up a separate Python environment directly in Windows:

```powershell
winget install astral-sh.uv
mkdir C:\code\trackmania; cd C:\code\trackmania
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

Use the exact PyTorch command from [pytorch.org](https://pytorch.org/get-started/locally/) for Windows. The Windows NVIDIA driver you installed in Step 1 serves both environments, so nothing else needs installing.

## WSL-specific problems

| Symptom | Fix |
| --- | --- |
| `nvidia-smi` not found or fails in WSL | Update the Windows NVIDIA driver and run `wsl --update`. Make sure you didn't install a Linux driver inside WSL. |
| Very slow data loading | Move the data from `/mnt/c/...` into your Linux home directory. |
| WSL runs out of memory or gets killed | Raise `memory` in `.wslconfig`. |
| Disk filling up | WSL's virtual disk grows but doesn't shrink on its own. Clean up `~/.cache/huggingface` and old checkpoints. |

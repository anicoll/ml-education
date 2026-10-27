---
sidebar_position: 1
title: Choosing your machine
---

# Choosing your machine

This course works on three kinds of setup. You can use more than one: many people prototype on a Mac and run CUDA-only jobs on a PC.

| | RTX 3070 PC (Windows + WSL2, or native Ubuntu) | Mac with Apple Silicon (M4 Pro / Max or newer) |
| --- | --- | --- |
| Memory for models | 8 GB VRAM | Unified memory shared by CPU and GPU. With 48 GB, roughly 36 GB is usable by the GPU by default. |
| GPU backend | CUDA | Metal, via PyTorch **MPS** or Apple's **MLX** |
| Ecosystem support | Everything. Almost all ML tooling targets CUDA first. | Good and improving. Some CUDA-only libraries (bitsandbytes, Unsloth, vLLM, FlashAttention) don't work. |
| Strength | Fast training for models that fit; every tutorial just works | Runs and fine-tunes **much bigger models** than 8 GB allows |
| Weakness | 8 GB is a hard ceiling | Slower per step for training; some tutorials need adapting |

## Which machine for which part of the course

| Part | Best on | Notes |
| --- | --- | --- |
| 1–2. Foundations, classical ML | Either | Mostly NumPy and scikit-learn, CPU is fine. |
| 3–4. Neural networks, architectures | Either | PyTorch runs on CUDA and MPS. The 3070 is usually faster for small models. |
| 5. Training from scratch | 3070 first | `torch.compile` and bf16 autocast are most mature on CUDA. The Mac works too, more slowly. |
| 6. Running models locally | **Mac** | 48 GB of unified memory runs 30B-class models comfortably, and 70B at 4-bit if you raise the GPU memory limit. |
| 6. QLoRA walkthrough | 3070 for the Hugging Face + bitsandbytes route as written | On the Mac, use **MLX** for LoRA/QLoRA instead. See [Apple Silicon](./apple-silicon.md). |
| 6. Fine-tuning 7B+ models | **Mac** | LoRA on 7–8B and QLoRA on 14B+ fit easily in 48 GB. |

## Writing code that runs on both

Pick the device once at the top of every script:

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

Then use `.to(device)` everywhere instead of `.cuda()`. Most of the lesson code uses `"cuda"`; swap in `device` when running on the Mac.

## Setup guides

- [Windows + WSL2](./setup-windows-wsl.md) for the 3070 under Windows (recommended if you're on Windows).
- [Native Ubuntu](./setup-linux.md) for the 3070 on a Linux install.
- [macOS on Apple Silicon](./setup-macos.md).

## Hardware references

- [RTX 3070 (8 GB, CUDA)](./rtx-3070.md): what fits and habits for an 8 GB card.
- [Apple Silicon (unified memory)](./apple-silicon.md): MPS vs MLX and what fits in 48 GB.
- [VRAM maths](./vram-math.md): estimate memory for any model on any machine.
- [When to rent a GPU](./cloud-gpus.md).

---
sidebar_position: 1
title: Choosing your machine
---

# Choosing your machine

This course works on two broad kinds of machine. You can use more than one: many people prototype on a Mac and run CUDA-only jobs on a PC.

| | PC with an NVIDIA GPU (Windows + WSL2, or native Ubuntu) | Mac with Apple Silicon (M-series, ideally a Pro or Max chip) |
| --- | --- | --- |
| Memory for models | The card's own VRAM. Common sizes are 8, 12, 16 and 24 GB. This course uses **8 GB** as its baseline because it's the most common, and the most limiting. | Unified memory shared by CPU and GPU. By default the GPU can use roughly two-thirds to three-quarters of it (about 24 GB on a 36 GB Mac, 36 GB on a 48 GB Mac). |
| GPU backend | CUDA | Metal, via PyTorch **MPS** or Apple's **MLX** |
| Ecosystem support | Everything. Almost all ML tooling targets CUDA first. | Good and improving. Some CUDA-only libraries (bitsandbytes, Unsloth, vLLM, FlashAttention) don't work. |
| Strength | Fast training for models that fit; every tutorial just works | With 32 GB or more, runs and fine-tunes **much bigger models** than an 8–12 GB card allows |
| Weakness | VRAM is a hard ceiling | Slower per step for training; some tutorials need adapting |

You don't need a powerful machine for Parts 1–4: any laptop from the last few years will do, and the free GPUs in Google Colab or Kaggle cover the rest. See [When to rent a GPU](./cloud-gpus.md).

## Which machine for which part of the course

| Part | Best on | Notes |
| --- | --- | --- |
| 1–2. Foundations, classical ML | Anything | Mostly NumPy and scikit-learn, CPU is fine. |
| 3–4. Neural networks, architectures | Either | PyTorch runs on CUDA and MPS. A mid-range NVIDIA card is usually faster for small models. |
| 5. Training from scratch | NVIDIA first | `torch.compile` and bf16 autocast are most mature on CUDA. A Mac works too, more slowly. |
| 6. Running models locally | Whichever has more memory | An 8 GB card runs 7–8B models at 4-bit. A Mac with 32 GB or more runs 30B-class models comfortably. |
| 6. QLoRA walkthrough | NVIDIA for the Hugging Face + bitsandbytes route as written | On a Mac, use **MLX** for LoRA/QLoRA instead. See [Apple Silicon](./apple-silicon.md). |
| 6. Fine-tuning 7B+ models | Mac with 32 GB+, or an NVIDIA card with 16 GB+ | QLoRA on an 8B model only just fits in 8 GB. |

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

Then use `.to(device)` everywhere instead of `.cuda()`. Most of the lesson code uses `"cuda"`; swap in `device` when running on a Mac.

## Setup guides

- [Windows + WSL2](./setup-windows-wsl.md) for an NVIDIA GPU under Windows (recommended if you're on Windows).
- [Native Ubuntu](./setup-linux.md) for an NVIDIA GPU on a Linux install.
- [macOS on Apple Silicon](./setup-macos.md).

## Hardware references

- [NVIDIA GPUs (8 GB and up)](./nvidia-gpu.md): what fits and habits for a memory-limited card.
- [Apple Silicon (unified memory)](./apple-silicon.md): MPS vs MLX and what fits at each memory size.
- [VRAM maths](./vram-math.md): estimate memory for any model on any machine.
- [When to rent a GPU](./cloud-gpus.md).

## Check your understanding

<MultipleChoice
  id="hw-overview-1"
  question="What is the main practical limit of a typical consumer NVIDIA card for ML work?"
  options={[
    "Its compute speed: it's too slow for anything but toy models",
    "Its VRAM: models, activations and optimiser state must all fit in it",
    "It can only run CUDA code written in C++",
    "It can't use bf16",
  ]}
  answer={1}
  explanation="Consumer NVIDIA cards are fast. What stops you is memory: if the weights, gradients, optimiser state and activations don't fit in VRAM, the run fails or crawls."
/>

<MultipleChoice
  id="hw-overview-2"
  question="Why does a Mac with 48 GB of unified memory let you run bigger models than a PC with an 8 GB GPU and 64 GB of system RAM?"
  options={[
    "Macs compress model weights automatically",
    "On the Mac, the GPU can use most of the machine's RAM directly; on the PC the GPU can only use its own 8 GB of VRAM at full speed",
    "MLX models are smaller than PyTorch models",
    "The Mac's GPU is faster than any NVIDIA card",
  ]}
  answer={1}
  explanation="Unified memory means one pool shared by CPU and GPU. The PC's 64 GB of system RAM isn't usable by the GPU at full speed; layers that spill there run much more slowly."
/>

<OpenQuestion
  id="hw-overview-3"
  question="You write `model.cuda()` in a script and then try to run it on a Mac. What happens, and how should you write it instead?">

It fails, because there is no CUDA device on a Mac. Pick the device once at the top of the script and move things to it:

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = model.to(device)
x = x.to(device)
```

Now the same script runs on an NVIDIA PC, a Mac, or a plain CPU.

</OpenQuestion>

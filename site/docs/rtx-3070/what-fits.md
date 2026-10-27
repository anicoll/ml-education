---
sidebar_position: 1
title: What fits on 8 GB
---

# What fits on an RTX 3070

Your RTX 3070 is a genuinely useful ML card. Its limit is memory, not speed. This page is the quick reference; [VRAM maths](./vram-math.md) explains how to work out the numbers yourself.

## The card

| Spec | RTX 3070 | Why it matters |
| --- | --- | --- |
| VRAM | 8 GB GDDR6 | The hard limit on model and batch size. |
| Architecture | Ampere (compute capability 8.6) | Supports **bf16** and TF32, FlashAttention 2, and bitsandbytes 4-bit. |
| Tensor cores | 3rd gen | Fast mixed-precision matrix maths. |
| Memory bandwidth | ~448 GB/s | Determines inference speed for LLMs (generation is memory-bound). |

Practically, your desktop and browser also use some VRAM, so plan on about **7–7.5 GB usable**. Closing other GPU-heavy apps helps.

## Cheat sheet

### Training from scratch

| Project | Fits? |
| --- | --- |
| MLPs and CNNs on MNIST / CIFAR-10 | Easily |
| Fine-tuning ResNet / ViT-small on your images | Easily |
| GPT with 10–30M parameters | Easily |
| GPT with ~100M parameters (GPT-2 small size) | Yes, with bf16, small batches plus gradient accumulation; slow to train to a good loss |
| Anything over ~300M parameters from scratch | Fits in memory with effort, but too slow to train meaningfully. Rent. |

### Running LLMs (inference)

| Model | Format | Fits? |
| --- | --- | --- |
| Up to ~3B | bf16 | Yes |
| 7–8B | 4-bit (Q4_K_M / NF4) | Yes, with a few thousand tokens of context |
| 7–8B | 8-bit | Just over; partially on CPU |
| 13–14B | 4-bit | Partially on CPU, slower |
| 30B+ | 4-bit | Mostly CPU; slow but possible with lots of system RAM |

### Fine-tuning LLMs

| Model | Method | Fits? |
| --- | --- | --- |
| Up to ~0.5B | Full fine-tuning | Yes |
| 0.5–1.5B | LoRA on bf16 base | Yes |
| 1.5–3B | QLoRA | Yes, comfortably. **The sweet spot.** |
| 7–8B | QLoRA | Yes, tightly: batch size 1, sequences of 512–1024, gradient checkpointing. Unsloth helps a lot. |
| 13B+ | Anything | No. Rent a GPU. |

### Other things that work well

- **Image generation:** Stable Diffusion 1.5 and SDXL run well. Training LoRAs for them works too.
- **Speech-to-text:** Whisper (up to large, with some care) runs fine.
- **Embeddings and RAG:** sentence-embedding models are tiny; a local RAG system with an 8B model is very doable.

## Habits for an 8 GB card

1. **Always use bf16** (or 4-bit for frozen weights).
2. **Small batch + gradient accumulation** instead of big batches.
3. **Gradient checkpointing** on whenever memory is tight.
4. **Shorter sequences** are the biggest single lever for LLM fine-tuning memory.
5. **Measure, don't guess:** `torch.cuda.max_memory_allocated()` after a few steps, and `nvidia-smi`.
6. **Know when to rent.** If a run won't fit, an hour on a 24–80 GB cloud GPU costs less than an evening fighting memory. See [When to rent a GPU](./cloud-gpus.md).

---
sidebar_position: 6
title: LoRA and QLoRA
---

# LoRA and QLoRA

LoRA is the technique that makes fine-tuning large models possible on hardware like yours. QLoRA combines it with 4-bit quantisation to stretch it even further.

## LoRA: Low-Rank Adaptation

A weight matrix $W$ in a transformer might be $4096 \times 4096$: about 16.8 million numbers. Full fine-tuning learns an update $\Delta W$ of the same size.

The LoRA insight (Hu et al., 2021): the update needed for fine-tuning is usually **low-rank**. It can be well approximated as the product of two thin matrices:

$$
\Delta W = B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll d
$$

With rank $r = 16$, $B$ and $A$ together have $4096 \times 16 + 16 \times 4096 \approx 131{,}000$ numbers. That's **under 1%** of the full matrix.

The layer's output becomes:

$$
h = W x + \frac{\alpha}{r} B A x
$$

- $W$ is **frozen**: no gradients, no optimiser state.
- Only $A$ and $B$ are trained.
- $B$ starts at zero, so at the start of training the model behaves exactly like the original.
- $\alpha$ is a scaling factor that controls how strongly the adapter affects the output.

```mermaid
flowchart LR
  x[x] --> W["W (frozen)"]
  x --> A["A (r × k, trained)"]
  A --> B["B (d × r, trained)"]
  W --> plus(("+"))
  B -->|"× α/r"| plus
  plus --> h[h]
```

## Why it saves so much memory

Recall full fine-tuning costs about 16 bytes per parameter. With LoRA, the frozen weights cost just their storage (2 bytes in bf16), and the 16-bytes-per-parameter cost applies only to the tiny adapter. For a 1.5B model:

- Full fine-tune: about 24 GB plus activations. Doesn't fit.
- LoRA: about 3 GB for frozen weights plus a few hundred MB for adapters and optimiser state, plus activations. Fits.

Bonus: the trained adapter is a small file (tens of MB), so you can keep many adapters for one base model and swap them.

## QLoRA: LoRA on a 4-bit base

**QLoRA** (Dettmers et al., 2023) goes one step further: store the frozen base model in **4-bit NF4**, and train 16-bit LoRA adapters on top. Gradients flow *through* the dequantised 4-bit weights into the adapters.

That cuts the frozen weights' cost from 2 bytes to about 0.5 bytes per parameter. For an 8B model: about 5 GB instead of 16 GB, which is what makes 7–8B fine-tuning possible on an 8 GB card (only just; see below).

QLoRA also introduced **paged optimisers**, which move optimiser state to CPU RAM when the GPU runs out, avoiding crashes from memory spikes.

## The knobs

| Hyperparameter | Typical values | Effect |
| --- | --- | --- |
| `r` (rank) | 8–64 | Adapter capacity. 16 is a good default. Higher for bigger behaviour changes. |
| `lora_alpha` | Often `r` or `2r` | Scale of the update. Effective strength is alpha divided by r. |
| `target_modules` | Attention projections (q, k, v, o) or `"all-linear"` | Which layers get adapters. All linear layers usually works best. |
| `lora_dropout` | 0–0.1 | Regularisation on the adapter. |
| Learning rate | 1e-4 to 2e-4 | Higher than full fine-tuning, because adapters start from zero. |
| Epochs | 1–3 | More risks overfitting a small dataset. |

## What fits on an RTX 3070 (8 GB)

Approximate, with gradient checkpointing, bf16 compute and batch size 1–2 plus gradient accumulation:

| Base model | Method | Max comfortable sequence length | Verdict |
| --- | --- | --- | --- |
| 0.5–1.5B | LoRA (bf16 base) | 2048+ | Comfortable. Great first project. |
| 3B | QLoRA | 1024–2048 | Comfortable. The sweet spot for real projects. |
| 7–8B | QLoRA | 512–1024 | Tight but works. Expect slow steps and to tune memory carefully. |
| 13B+ | QLoRA | — | Doesn't fit. Rent a GPU. |

**Unsloth** is a library that rewrites the key kernels for LoRA/QLoRA training and typically cuts memory use substantially and roughly doubles speed. On an 8 GB card it can make the difference between a 7–8B run fitting or not. Worth using once you understand the plain `peft` version.

## What fits on a 48 GB Apple Silicon Mac

With about 36–40 GB usable by the GPU, using MLX (`mlx_lm.lora`):

| Base model | Method | Verdict |
| --- | --- | --- |
| 1.5–3B | LoRA (bf16 base) | Easy, long sequences fine. Good for fast iteration. |
| 7–8B | LoRA (bf16 base) | Comfortable. No quantisation needed. |
| 14B | QLoRA (4-bit base) | Comfortable. |
| 27–32B | QLoRA (4-bit base) | Possible with short sequences and fewer LoRA layers. |
| 70B | — | Doesn't fit. Rent a GPU. |

Steps are slower than on an NVIDIA card of similar size, but the Mac can fine-tune much larger models than the 3070. A good workflow is to debug your data and settings on a small model on either machine, then do the 7B+ run on the Mac. See [Apple Silicon](../hardware/apple-silicon.md) for the commands.

## After training

You have two choices:

1. **Keep the adapter separate.** Load the base model, then load the adapter on top. Easy to swap adapters.
2. **Merge it in.** Compute $W + \frac{\alpha}{r}BA$ once and save a normal model. No extra inference cost, and you can then convert it to GGUF and run it in Ollama.

:::info Key idea
LoRA freezes the big model and trains two small matrices per layer whose product is the update. QLoRA does the same with the frozen model stored in 4-bit. Together they cut fine-tuning memory by around 10× or more.
:::

:::tip Try it
Before training anything, calculate on paper: for a 3B model with 28 layers, hidden size 3072, and LoRA rank 16 on just the q and v projections (each $3072 \times 3072$), how many trainable parameters are there? What fraction of the model is that? Then check your answer with `model.print_trainable_parameters()` in the walkthrough on the next page.
:::

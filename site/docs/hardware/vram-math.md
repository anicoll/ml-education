---
sidebar_position: 7
title: VRAM maths
---

# VRAM maths

You can estimate whether something will fit before you try it. GPU memory during ML goes to four things.

## 1. Weights

$$
\text{weights memory} = \text{parameters} \times \text{bytes per parameter}
$$

| Precision | Bytes |
| --- | --- |
| fp32 | 4 |
| bf16 / fp16 | 2 |
| int8 | 1 |
| 4-bit | ~0.5 (plus a little for scale factors) |

An 8B model: 16 GB in bf16, about 4.5–5 GB in 4-bit.

## 2. Gradients

One gradient per **trainable** parameter, usually in the same precision as the weights (2 bytes in bf16). Zero for frozen weights, which is why LoRA saves so much.

## 3. Optimiser state

AdamW keeps two values per trainable parameter, normally in fp32: **8 bytes per trainable parameter**. 8-bit optimisers (`adamw_8bit` / `paged_adamw_8bit`) cut this to about 2 bytes.

Mixed-precision training often also keeps an fp32 "master copy" of trainable weights (4 more bytes). Adding it all up gives the rule of thumb:

$$
\text{full fine-tuning} \approx 16 \text{ bytes per parameter} + \text{activations}
$$

## 4. Activations (and the KV cache)

Intermediate values stored for the backward pass. They scale with **batch size × sequence length × hidden size × number of layers**, and attention parts grow with the square of sequence length (less so with FlashAttention). This is the part that changes as you adjust batch size and sequence length, and it can easily dominate.

**Gradient checkpointing** stores only some activations and recomputes the rest during the backward pass, cutting this cost dramatically for about 20–30% more compute.

For inference there are no gradients or optimiser state, but there is the **KV cache**:

$$
\text{KV cache} = 2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{context length} \times \text{bytes}
$$

For Llama 3.1 8B (32 layers, 8 KV heads, head dim 128) in bf16, that's 2 × 32 × 8 × 128 × 2 bytes = 128 KB per token, so an 8,192-token context costs about 1 GB.

## On a Mac: unified memory

The same four costs apply on Apple Silicon. The difference is the budget: instead of a fixed amount of VRAM, the GPU shares the machine's RAM with macOS and your apps. By default, budget about two-thirds of RAM on Macs with 36 GB or less and three-quarters above that (for example about **36 GB** on a 48 GB Mac), or a little more after raising the limit (see [Apple Silicon](./apple-silicon.md)). Everything else you have open eats into the same pool.

## Worked examples on 8 GB

**Full fine-tuning a 1.5B model:** 1.5B × 16 bytes = 24 GB before activations. Doesn't fit.

**LoRA on a 1.5B model (bf16 base):**
- Frozen weights: 1.5B × 2 = 3 GB
- Adapters (say 20M params) × 16 bytes = 0.3 GB
- Activations with checkpointing, batch 2, length 1024: roughly 1–2 GB
- Total: about 5 GB. Fits.

**QLoRA on an 8B model:**
- Frozen weights in 4-bit: about 5 GB
- Adapters (about 40M params at r=16, all-linear) × ~10 bytes with 8-bit Adam: about 0.4 GB
- Activations with checkpointing, batch 1, length 512–1024: roughly 1–2 GB
- CUDA context and overhead: about 0.5 GB
- Total: about 7–8 GB. Right on the edge. This is why 8B QLoRA on an 8 GB card needs short sequences, and why tools like Unsloth that save memory matter here.

**Running an 8B model at Q4_K_M with 4k context:** about 4.9 GB weights + 0.5 GB KV cache + overhead. Fits comfortably.

## Worked examples on a 48 GB Mac (about 36 GB usable)

**LoRA on an 8B model (bf16 base):** 16 GB frozen weights + about 0.5 GB adapters and optimiser state + a few GB of activations. About 20 GB. Fits easily, without needing 4-bit at all.

**QLoRA on a 32B model:** about 18 GB of 4-bit weights + adapters + activations. Around 25–30 GB with short sequences. Fits, with care.

**Running a 70B model at 4-bit:** about 40 GB of weights alone, plus KV cache. Only fits after raising the GPU memory limit, with a short context.

## Measuring

```python
torch.cuda.reset_peak_memory_stats()
# ... run a few training steps ...
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
```

```bash
nvidia-smi -l 1          # live view, refresh every second
```

On a Mac, use `torch.mps.driver_allocated_memory()` from PyTorch, `mx.get_peak_memory()` from MLX (older versions: `mx.metal.get_peak_memory()`), and the Memory tab of Activity Monitor.

PyTorch reserves memory in chunks, so `nvidia-smi` shows more than `max_memory_allocated`. If you're right at the limit, setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can reduce fragmentation.

:::tip[Try it]
Before your first QLoRA run, fill in this estimate for your chosen model and settings in `notes/`, then compare it with the measured peak. Being able to predict memory use within a gigabyte or so is a genuinely useful skill.
:::

## Check your understanding

<MultipleChoice
  id="vram-1"
  question="Roughly how much memory does full fine-tuning a 3B model need, before activations?"
  options={["6 GB", "12 GB", "48 GB", "3 GB"]}
  answer={2}
  explanation="About 16 bytes per parameter (bf16 weights and gradients, fp32 master weights and two AdamW values): 3B × 16 = 48 GB."
/>

<MultipleChoice
  id="vram-2"
  question="Which part of training memory changes when you change the batch size or sequence length?"
  options={["Weights", "Optimiser state", "Activations", "None of them"]}
  answer={2}
  explanation="Activations scale with batch × sequence length × hidden size × layers. Weights, gradients and optimiser state depend only on the model."
/>

<MultipleChoice
  id="vram-3"
  question="What does gradient checkpointing trade?"
  options={["Accuracy for speed", "About 20–30% more compute for much less activation memory", "Disk space for RAM", "Nothing, it's free"]}
  answer={1}
  explanation="It stores only some activations and recomputes the rest during the backward pass."
/>

<OpenQuestion
  id="vram-4"
  question="Estimate the memory to run a 14B model at 4-bit with an 8,000-token context, assuming 48 layers, 8 KV heads and a head dim of 128 in bf16. Does it fit on a 12 GB card?">

**Weights:** 14B × about 0.55 bytes ≈ **7.7 GB**.

**KV cache per token:** 2 (K and V) × 48 layers × 8 heads × 128 head dim × 2 bytes = 196,608 bytes ≈ **192 KB**.

**KV cache for 8,000 tokens:** 8,000 × 192 KB ≈ **1.5 GB**.

**Overhead** (CUDA context, buffers): about **0.5–1 GB**.

**Total:** 7.7 + 1.5 + 0.75 ≈ **10 GB**. It fits on a 12 GB card with a little room to spare, but not on an 8 GB card (there, some layers would spill to the CPU, or you'd need a shorter context and a smaller quantisation).

</OpenQuestion>

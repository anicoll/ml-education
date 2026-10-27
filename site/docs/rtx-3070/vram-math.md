---
sidebar_position: 2
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
- Total: about 7–8 GB. Right on the edge. This is why 8B QLoRA on a 3070 needs short sequences, and why tools like Unsloth that save memory matter here.

**Running an 8B model at Q4_K_M with 4k context:** about 4.9 GB weights + 0.5 GB KV cache + overhead. Fits comfortably.

## Measuring

```python
torch.cuda.reset_peak_memory_stats()
# ... run a few training steps ...
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
```

```bash
nvidia-smi -l 1          # live view, refresh every second
```

PyTorch reserves memory in chunks, so `nvidia-smi` shows more than `max_memory_allocated`. If you're right at the limit, setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can reduce fragmentation.

:::tip Try it
Before your first QLoRA run, fill in this estimate for your chosen model and settings in `notes/`, then compare it with the measured peak. Being able to predict memory use within a gigabyte or so is a genuinely useful skill.
:::

---
sidebar_position: 2
title: "NVIDIA GPUs (8 GB and up)"
---

# What fits on an NVIDIA GPU

If you also have an Apple Silicon Mac, see [Choosing your machine](./overview.md) for which jobs suit which machine.

A consumer NVIDIA card with 8 GB of VRAM (for example an RTX 3060 Ti, 3070, 4060 or 4060 Ti 8 GB) is a genuinely useful ML card. Its limit is memory, not speed. This page uses 8 GB as the baseline because it's the most common size and the tightest; if your card has more, everything shifts up (see the table at the end). [VRAM maths](./vram-math.md) explains how to work out the numbers yourself.

## What to look for in a card

| Spec | Typical value | Why it matters |
| --- | --- | --- |
| VRAM | 8–24 GB | The hard limit on model and batch size. |
| Architecture | Ampere (RTX 30-series) or newer | Supports **bf16** and TF32, FlashAttention 2, and bitsandbytes 4-bit. Older cards (RTX 20-series) work but lack bf16. |
| Tensor cores | 3rd gen or newer | Fast mixed-precision matrix maths. |
| Memory bandwidth | ~250–1,000 GB/s | Determines inference speed for LLMs (generation is memory-bound). |

Run `nvidia-smi` to see your card's name and VRAM. Your desktop and browser also use some VRAM, so on an 8 GB card plan on about **7–7.5 GB usable**. Closing other GPU-heavy apps helps.

## Cheat sheet for 8 GB

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

## If your card has more memory

| VRAM | Comfortable QLoRA | Comfortable LoRA (bf16 base) | Inference at 4-bit |
| --- | --- | --- | --- |
| 8 GB | 3B, tight 8B | 1.5B | 7–8B |
| 12 GB | 8B | 3B | 14B |
| 16 GB | 8B with longer sequences, tight 14B | 3–4B | 14B, tight 24B |
| 24 GB | 14B, comfy 8B with long sequences | 8B | 32B |

## Habits for a memory-limited card

1. **Always use bf16** (or 4-bit for frozen weights).
2. **Small batch + gradient accumulation** instead of big batches.
3. **Gradient checkpointing** on whenever memory is tight.
4. **Shorter sequences** are the biggest single lever for LLM fine-tuning memory.
5. **Measure, don't guess:** `torch.cuda.max_memory_allocated()` after a few steps, and `nvidia-smi`.
6. **Know when to rent.** If a run won't fit, an hour on a 24–80 GB cloud GPU costs less than an evening fighting memory. See [When to rent a GPU](./cloud-gpus.md).

## Check your understanding

<MultipleChoice
  id="hw-nvidia-1"
  question="You want to fine-tune a 3B model on an 8 GB card. Which approach is the comfortable fit?"
  options={["Full fine-tuning in fp32", "Full fine-tuning in bf16", "QLoRA (4-bit frozen base plus small LoRA adapters)", "None: 3B models can't be fine-tuned on 8 GB"]}
  answer={2}
  explanation="Full fine-tuning costs about 16 bytes per parameter (48 GB for 3B). QLoRA keeps the base frozen at 4-bit (about 2 GB) and trains only small adapters, which is why 1.5–3B QLoRA is the sweet spot on 8 GB."
/>

<MultipleChoice
  id="hw-nvidia-2"
  question="Your training run crashes with CUDA out of memory. Which change usually saves the most memory for LLM fine-tuning?"
  options={["Lowering the learning rate", "Shortening the sequence length", "Training for fewer epochs", "Switching optimiser from AdamW to Adam"]}
  answer={1}
  explanation="Activation memory grows with batch size × sequence length (and attention with its square). Learning rate and epochs don't change memory at all."
/>

<OpenQuestion
  id="hw-nvidia-3"
  question="Explain gradient accumulation in your own words. Why does it help on a small GPU?">

Instead of processing one big batch of, say, 32 examples at once (which might not fit in memory), you process 8 small batches of 4, call `loss.backward()` after each one so the gradients **add up**, and only call `optimizer.step()` after the 8th. The update is (almost) the same as with a batch of 32, but you only ever hold 4 examples' activations in memory at a time.

```python
accum_steps = 8
for i, (x, y) in enumerate(loader):        # loader gives batches of 4
    loss = loss_fn(model(x), y) / accum_steps
    loss.backward()                         # gradients accumulate
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

The cost is time, not memory: it takes as long as running 32 examples one small batch at a time.

</OpenQuestion>

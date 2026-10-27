---
sidebar_position: 3
title: Apple Silicon (unified memory)
---

# Apple Silicon: M4 Pro, M4 Max and newer

Apple Silicon Macs are a very different machine from a gaming PC, and in one important way a better one for LLMs: **memory**. An M4 Max with 48 GB can hold models that would never fit on an 8 GB card.

## Unified memory

On a PC, the GPU has its own VRAM, separate from system RAM. On Apple Silicon, the CPU and GPU share one pool of **unified memory**. There's no copying between them, and the GPU can use most of the machine's RAM.

- By default macOS lets the GPU use about **75% of RAM** on a 48 GB machine, so roughly **36 GB**.
- You can raise this limit (until the next reboot) when you need it for a big model. Leave at least 8 GB for macOS:

```bash
sudo sysctl iogpu.wired_limit_mb=40960   # allow the GPU about 40 GB
```

- **Memory bandwidth** decides how fast LLMs generate text. The M4 Max has roughly 410–546 GB/s depending on the chip variant, and the M4 Pro about 273 GB/s. The RTX 3070 has about 448 GB/s. So token generation speed per GB of model is similar to the 3070, but you can load models four to five times bigger.
- **Compute** (for training) is where the 3070 tends to win. NVIDIA's tensor cores are very fast at the mixed-precision matrix maths training needs. Expect training steps for a model that fits on both to be somewhat slower on the Mac. Compare them yourself.

## Two ways to use the GPU

### PyTorch with MPS

PyTorch's **MPS** (Metal Performance Shaders) backend runs PyTorch code on the Mac GPU. Most code from Parts 3–5 runs unchanged once you set `device = "mps"`.

```python
import torch
print(torch.backends.mps.is_available())   # True
x = torch.randn(4096, 4096, device="mps")
y = x @ x
```

Things to know:

- A few operations aren't implemented on MPS yet. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so they quietly run on the CPU instead of erroring.
- bf16 works on recent macOS and PyTorch versions. If you hit dtype errors, fall back to float32 for small models (you have the memory).
- `torch.compile` support for MPS is limited. Skip it on the Mac.
- Memory tools differ: use `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` instead of the `torch.cuda` functions, and Activity Monitor's GPU and memory views instead of `nvidia-smi`.
- **bitsandbytes doesn't support MPS**, so the 4-bit `BitsAndBytesConfig` route from [Quantisation](../open-source-models/quantisation.md) and the [QLoRA walkthrough](../open-source-models/qlora-walkthrough.md) won't work here. Use MLX for that, or plain LoRA on a bf16 base (which fits in 48 GB for models up to about 8B).

### MLX

**MLX** is Apple's own array framework, designed for Apple Silicon. Its API looks like NumPy and PyTorch. For LLMs, the `mlx-lm` package gives you ready-made tools to run, quantise and fine-tune models, and it's usually the fastest option on a Mac.

```bash
pip install mlx-lm

# Run a 4-bit model from the mlx-community organisation on Hugging Face
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Explain LoRA in three sentences."

# Interactive chat
mlx_lm.chat --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

The `mlx-community` organisation on Hugging Face hosts pre-converted MLX versions of most popular open models at several quantisation levels.

## Fine-tuning with MLX (LoRA and QLoRA)

`mlx-lm` includes LoRA training. If the base model is quantised (for example a `-4bit` model), training on it is QLoRA.

1. Put your data in a folder as `train.jsonl` and `valid.jsonl`, using the same `{"messages": [...]}` chat format as the [QLoRA walkthrough](../open-source-models/qlora-walkthrough.md).
2. Train:

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --num-layers 16 \
  --iters 1000 \
  --learning-rate 1e-4 \
  --adapter-path ./adapters
```

3. Try it with the adapter:

```bash
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters --prompt "..."
```

4. Merge the adapter into the model with `mlx_lm.fuse` if you want a standalone model.

`--num-layers` controls how many transformer layers get LoRA adapters (fewer means less memory). Flag names change between versions, so check `mlx_lm.lora --help` and the [mlx-lm LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) if something errors.

The concepts are exactly the same as on the [LoRA and QLoRA](../open-source-models/lora-and-qlora.md) page. Only the tool changes.

## What fits in 48 GB

Assuming about 36 GB usable by the GPU (default limit), or about 40 GB after raising it. Use [VRAM maths](./vram-math.md) to check specific cases.

### Running LLMs

| Model size | Format | Fits? |
| --- | --- | --- |
| 7–8B | bf16 (about 16 GB) | Yes, comfortably, with long context |
| 14B | 8-bit (about 15 GB) | Yes |
| 27–32B | 4-bit (about 17–20 GB) | Yes, comfortably. The sweet spot for quality on this machine. |
| 30B-class mixture-of-experts | 4-bit | Yes, and fast, because only a few billion parameters are active per token |
| 70B | 4-bit (about 40 GB) | Only after raising the GPU limit, with short context and little else running. 3-bit fits more easily, with some quality loss. |
| 100B+ | Any | No |

### Fine-tuning

| Model size | Method | Fits? |
| --- | --- | --- |
| Up to about 1.5B | Full fine-tuning (about 16 bytes per parameter) | Yes |
| 7–8B | LoRA on a bf16 base (MLX or PyTorch MPS) | Yes |
| 14B | QLoRA (MLX, 4-bit base) | Yes, comfortably |
| 27–32B | QLoRA (MLX, 4-bit base) | Possible with short sequences, a small batch and fewer LoRA layers |
| 70B | Anything | No. Rent a GPU. |

### Training from scratch

Memory isn't the limit here, compute is. The tiny-GPT capstone works fine on the Mac. Bigger from-scratch runs (100M+ parameters) will be slow on either machine.

## Running models with Ollama and llama.cpp

Both use Metal on the Mac automatically and work well. Ollama is the easiest way to try many models. LM Studio also supports MLX models on the Mac and is a nice GUI for comparing them.

## Habits for a Mac

1. **Prefer MLX for LLM work** (inference and LoRA), and PyTorch MPS for the from-scratch lessons.
2. **Close memory-hungry apps** before loading a large model. Browsers and Electron apps share the same memory pool as the GPU.
3. **Watch memory pressure** in Activity Monitor. If it goes yellow or red, macOS is swapping and everything will crawl. Use a smaller model or a shorter context.
4. **Keep the laptop plugged in** for long runs, and expect the fans to spin up.
5. **When a tutorial says `cuda`**, change the device to `mps`. When it uses bitsandbytes, Unsloth or vLLM, look for the MLX equivalent or run it on the 3070.

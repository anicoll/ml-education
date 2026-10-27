---
sidebar_position: 3
title: Apple Silicon (unified memory)
---

# Apple Silicon Macs

Apple Silicon Macs are a very different machine from a gaming PC, and in one important way a better one for LLMs: **memory**. A Mac with 32 GB or more can hold models that would never fit on an 8 GB card. Any M-series chip works for this course; Pro, Max and Ultra chips have more GPU cores and more memory bandwidth, so they train and generate faster.

## Unified memory

On a PC, the GPU has its own VRAM, separate from system RAM. On Apple Silicon, the CPU and GPU share one pool of **unified memory**. There's no copying between them, and the GPU can use most of the machine's RAM.

- By default macOS lets the GPU use roughly **two-thirds of RAM** on Macs with 36 GB or less, and about **three-quarters** above that. So about 21 GB on a 32 GB Mac, 24 GB on 36 GB, 36 GB on 48 GB and 48 GB on 64 GB.
- You can raise this limit (until the next reboot) when you need it for a big model. Leave at least 8 GB for macOS. For example, on a 48 GB Mac:

```bash
sudo sysctl iogpu.wired_limit_mb=40960   # allow the GPU about 40 GB
```

- **Memory bandwidth** decides how fast LLMs generate text. It ranges from about 100–150 GB/s on base M-series chips, through roughly 200–300 GB/s on Pro chips, to 400–550 GB/s on Max chips. A mid-range NVIDIA card has roughly 250–500 GB/s. So token generation speed per GB of model is similar on a Pro or Max chip, but you can load much bigger models.
- **Compute** (for training) is where NVIDIA cards tend to win. NVIDIA's tensor cores are very fast at the mixed-precision matrix maths training needs. Expect training steps for a model that fits on both to be somewhat slower on the Mac. Compare them yourself.

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
- **bitsandbytes doesn't support MPS**, so the 4-bit `BitsAndBytesConfig` route from [Quantisation](../open-source-models/quantisation.md) and the [QLoRA walkthrough](../open-source-models/qlora-walkthrough.md) won't work here. Use MLX for that, or plain LoRA on a bf16 base (which fits on a 48 GB Mac for models up to about 8B).

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

## What fits at each memory size

These assume the default GPU memory limit. Use [VRAM maths](./vram-math.md) to check specific cases.

### Running LLMs

| Model size | Format | 16 GB Mac | 32–36 GB Mac | 48 GB Mac | 64 GB+ Mac |
| --- | --- | --- | --- | --- | --- |
| 7–8B | 4-bit (about 5 GB) | Yes | Yes | Yes | Yes |
| 7–8B | bf16 (about 16 GB) | No | Yes | Yes, with long context | Yes |
| 14B | 4-bit (about 9 GB) | Tight | Yes | Yes | Yes |
| 27–32B | 4-bit (about 17–20 GB) | No | Tight, short context | Yes, comfortably | Yes |
| 30B-class mixture-of-experts | 4-bit | No | Tight | Yes, and fast, because only a few billion parameters are active per token | Yes |
| 70B | 4-bit (about 40 GB) | No | No | Only after raising the GPU limit, with short context | Yes |

### Fine-tuning

| Model size | Method | Needs roughly |
| --- | --- | --- |
| Up to about 1.5B | Full fine-tuning (about 16 bytes per parameter) | 32 GB+ Mac |
| 1.5–3B | QLoRA (MLX, 4-bit base) | 16 GB Mac |
| 7–8B | QLoRA (MLX, 4-bit base) | 16 GB Mac, tightly; comfortable at 32 GB |
| 7–8B | LoRA on a bf16 base (MLX or PyTorch MPS) | 32–36 GB Mac, tightly; comfortable at 48 GB |
| 14B | QLoRA (MLX, 4-bit base) | 32 GB+ Mac |
| 27–32B | QLoRA (MLX, 4-bit base) | 48 GB+ Mac, with short sequences and fewer LoRA layers |
| 70B | Anything | Rent a GPU |

### Training from scratch

Memory isn't the limit here, compute is. The tiny-GPT capstone works fine on the Mac. Bigger from-scratch runs (100M+ parameters) will be slow on either machine.

## Running models with Ollama and llama.cpp

Both use Metal on the Mac automatically and work well. Ollama is the easiest way to try many models. LM Studio also supports MLX models on the Mac and is a nice GUI for comparing them.

## Habits for a Mac

1. **Prefer MLX for LLM work** (inference and LoRA), and PyTorch MPS for the from-scratch lessons.
2. **Close memory-hungry apps** before loading a large model. Browsers and Electron apps share the same memory pool as the GPU.
3. **Watch memory pressure** in Activity Monitor. If it goes yellow or red, macOS is swapping and everything will crawl. Use a smaller model or a shorter context.
4. **Keep the laptop plugged in** for long runs, and expect the fans to spin up.
5. **When a tutorial says `cuda`**, change the device to `mps`. When it uses bitsandbytes, Unsloth or vLLM, look for the MLX equivalent or run it on an NVIDIA machine.

---
sidebar_position: 3
title: Running models locally
---

# Running models locally

Before fine-tuning anything, get a feel for what open models can do by running them on your own machine. This is the quickest win in the course: you can be chatting with a capable model on your 3070 or Mac in about ten minutes.

## The options

| Tool | Good for | Notes |
| --- | --- | --- |
| **Ollama** | Easiest start | One-line install, one command to download and run a model. Runs a local API. |
| **LM Studio** | Point-and-click | Desktop app with a model browser and chat UI. |
| **llama.cpp** | Control and efficiency | The engine underneath Ollama and LM Studio. Runs GGUF quantised models on GPU, CPU or both. |
| **Hugging Face `transformers`** | Experimenting in Python | Most flexible, not the fastest. |
| **MLX (`mlx-lm`)** | Speed on a Mac | Apple's framework; usually the fastest way to run models on Apple Silicon. |
| **vLLM / SGLang** | Serving many requests fast | Best throughput; more setup; Linux or WSL2 with an NVIDIA GPU. |

## Quick start with Ollama

Install from [ollama.com](https://ollama.com), then:

```bash
ollama run llama3.1:8b        # downloads a 4-bit quantised 8B model (about 5 GB) and opens a chat
ollama run qwen2.5:7b
ollama list                   # what you've downloaded
ollama ps                     # what's loaded, and whether it's on the GPU
```

Ollama also exposes an OpenAI-compatible HTTP API at `http://localhost:11434`, so you can call your local model from code:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(model="llama3.1:8b", messages=[{"role": "user", "content": "Hi!"}])
print(resp.choices[0].message.content)
```

## GGUF and quantisation levels

llama.cpp and Ollama use **GGUF** files: model weights quantised to fewer bits. You'll see names like `Q4_K_M`, `Q5_K_M`, `Q8_0`:

- The number is roughly the bits per weight.
- `K_M` / `K_S` are variants of the "k-quant" scheme (medium and small).
- **Q4_K_M is the usual sweet spot**: about a quarter of the bf16 size, with small quality loss.

An 8B model at Q4_K_M is about 4.9 GB, which leaves room on your 8 GB card for the KV cache and a few thousand tokens of context.

## What happens when a model doesn't fit

llama.cpp can split layers between GPU and CPU (`--n-gpu-layers` / `-ngl`). Layers that don't fit on the GPU run on the CPU from system RAM. It works, but each layer on the CPU slows things down a lot. Rough expectations on a 3070:

- 7–8B at Q4, fully on GPU: fast, typically 40+ tokens per second.
- 13–14B at Q4: just over 8 GB, so a few layers spill to CPU. Usable but noticeably slower.
- 30B+: mostly on CPU. Slow, but it runs if you have 32 GB+ of system RAM.

On a 48 GB Mac, there's no split to worry about: the GPU uses unified memory, so 27–32B models at Q4 (about 17–20 GB) run entirely on the GPU at a usable speed. For the fastest generation on a Mac, try MLX models with `mlx_lm.chat` or in LM Studio. See [Apple Silicon](../hardware/apple-silicon.md).

## What to notice

As you play, pay attention to:

- **Speed** (tokens per second) and **VRAM** (`nvidia-smi`), and how both change with model size and context length.
- **Quality differences** between a 1.5B, a 3B and an 8B model on the same prompts.
- **Failure modes:** confident wrong facts (hallucination), losing track in long conversations, weak arithmetic.

These observations will guide what you choose to fine-tune and what you can realistically expect from it.

:::tip Try it
Install Ollama and run a 1–3B model and a 7–8B model. Write ten test prompts that matter to you (some factual, some reasoning, some writing, some code), run both models on them, and save the prompts and answers in `notes/model-comparison.md`. You'll reuse these prompts to evaluate your fine-tunes.
:::

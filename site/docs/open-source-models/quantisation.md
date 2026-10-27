---
sidebar_position: 4
title: Quantisation
---

# Quantisation

Quantisation stores model weights with fewer bits. It's the single biggest reason large models fit on consumer GPUs.

## The idea

Weights are normally 16-bit floats (2 bytes each). Quantisation maps them to 8-bit or 4-bit numbers, using a scale factor per small block of weights so the values can be recovered approximately:

$$
w \approx s \cdot q, \qquad q \in \{-8, \dots, 7\} \text{ for 4-bit}
$$

```text
bf16 weights:   0.031  -0.112   0.087   0.004  ...   (2 bytes each)
scale s = 0.016
4-bit codes:       2      -7       5       0   ...   (0.5 bytes each)
dequantised:   0.032  -0.112   0.080   0.000   ...
```

Every weight is now slightly off, but neural networks are surprisingly tolerant of small errors spread across billions of weights.

## Memory saved

| Precision | Bytes per param | 8B model weights |
| --- | --- | --- |
| fp32 | 4 | 32 GB |
| bf16 / fp16 | 2 | 16 GB |
| int8 | 1 | 8 GB |
| 4-bit | ~0.5–0.6 | ~4.5–5 GB |

That last row is why an 8B model fits on your card.

## Quality cost

- **8-bit:** almost no measurable loss.
- **4-bit (good methods):** small loss, usually acceptable. This is the standard for local use.
- **3-bit and below:** quality degrades noticeably, especially for smaller models.

A useful rule: **a bigger model at 4-bit usually beats a smaller model at 16-bit** for the same memory.

## Methods you'll meet

| Method | Used by | Notes |
| --- | --- | --- |
| **NF4** (bitsandbytes) | QLoRA, `transformers` | Quantises on the fly when loading. The standard for QLoRA training. |
| **GGUF k-quants** | llama.cpp, Ollama, LM Studio | The standard for local inference. |
| **GPTQ / AWQ** | vLLM, `transformers` | Pre-quantised models calibrated on sample data. Fast GPU inference. |

## Loading a 4-bit model in Python

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # "normal float 4", designed for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # also quantise the scale factors, saving a bit more
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             quantization_config=bnb, device_map="cuda")
print(model.get_memory_footprint() / 1e9, "GB")
```

bitsandbytes needs an NVIDIA GPU. On a Mac, use 4-bit models in MLX format (from the `mlx-community` organisation on Hugging Face) or GGUF models through llama.cpp and Ollama. See [Apple Silicon](../hardware/apple-silicon.md).

Weights are stored in 4-bit but **computed in bf16**: each layer's weights are dequantised on the fly as they're used. That saves memory, not compute.

## Quantisation and training

You can't train 4-bit weights directly: the steps would be too small to register in so few bits. This is where **QLoRA** comes in: keep the big base model frozen in 4-bit, and train small extra matrices in 16-bit alongside it. That's the next page.

:::tip Try it
Load the same 1.5B or 3B model in bf16, 8-bit and 4-bit. For each, record memory footprint and tokens per second, and run your ten test prompts from [Running models locally](./running-locally.md). Can you tell the difference in answer quality?
:::

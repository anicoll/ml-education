---
sidebar_position: 2
title: Hugging Face essentials
---

# Hugging Face essentials

Hugging Face is the hub of the open-model world: a site hosting models and datasets, plus the Python libraries everyone uses to work with them.

## The libraries

| Library | What it does |
| --- | --- |
| `transformers` | Load and run almost any model architecture with a common API. |
| `datasets` | Download, stream and process datasets efficiently. |
| `tokenizers` | Fast tokenisers (used under the hood by `transformers`). |
| `peft` | Parameter-efficient fine-tuning: LoRA, QLoRA and friends. |
| `trl` | Training recipes for LLMs: supervised fine-tuning (SFT), DPO, reward modelling. |
| `accelerate` | Handles devices, mixed precision and multi-GPU setup. |
| `bitsandbytes` | 8-bit and 4-bit quantisation (separate project, tightly integrated). |

```bash
pip install transformers datasets peft trl accelerate bitsandbytes
```

## Running a model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

messages = [{"role": "user", "content": "Explain gradient descent in two sentences."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
out = model.generate(inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

A 1.5B model in bf16 needs about 3 GB of VRAM, so this runs comfortably on your 3070.

## Chat templates

Instruct models were fine-tuned on conversations in a specific format, with special tokens marking who's speaking. For example (Qwen/ChatML style):

```text
<|im_start|>user
Explain gradient descent in two sentences.<|im_end|>
<|im_start|>assistant
```

`apply_chat_template` builds this from a list of messages. **Always use it** rather than formatting prompts by hand. A wrong template is one of the most common reasons a fine-tuned model behaves strangely.

```python
print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```

## Looking inside

The model is an ordinary PyTorch module, so everything from Parts 3–5 applies:

```python
print(model)                                     # the architecture
print(sum(p.numel() for p in model.parameters()) / 1e9, "B params")
print(model.config)                              # layers, heads, hidden size, vocab...
```

Compare what you see with the GPT you built in Part 5. The names differ, but you should recognise every component: embeddings, attention with q/k/v projections, MLPs, norms and the final projection to the vocabulary (`lm_head`).

## Where files go

Downloaded models are cached in `~/.cache/huggingface/hub` (on Windows, `C:\Users\<you>\.cache\huggingface\hub`). They're big: an 8B model is about 16 GB in bf16. Set the `HF_HOME` environment variable to put the cache on a drive with space.

Some models (like Llama) are "gated": accept the licence on the model's web page, then log in locally with `huggingface-cli login`.

:::tip Try it
Load `Qwen/Qwen2.5-0.5B-Instruct` and `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Print each architecture and identify the attention, MLP and normalisation layers. Then ask both the same five questions and compare the answers. Note VRAM use for each with `torch.cuda.max_memory_allocated()`.
:::

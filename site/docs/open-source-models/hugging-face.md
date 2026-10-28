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

A 1.5B model in bf16 needs about 3 GB of VRAM, so this runs comfortably on an 8 GB GPU or any recent Mac.

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

:::tip[Try it]
Load `Qwen/Qwen2.5-0.5B-Instruct` and `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Print each architecture and identify the attention, MLP and normalisation layers. Then ask both the same five questions and compare the answers. Note VRAM use for each with `torch.cuda.max_memory_allocated()`.
:::

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="tiZFewofSLM" title="The pipeline function" channel="Hugging Face">

A short video from the official Hugging Face course on the one-line `pipeline()` API.

</YouTube>

<YouTube id="QEaBAZQCtwE" title="Getting Started With Hugging Face in 15 Minutes | Transformers, Pipeline, Tokenizer, Models" channel="AssemblyAI">

A quick tour of pipelines, tokenizers and loading models, which is most of what this lesson covers.

</YouTube>

## Check your understanding

<MultipleChoice
  id="hf-1"
  question="Which Hugging Face library would you use to add LoRA adapters to a model?"
  options={["datasets", "tokenizers", "peft", "accelerate"]}
  answer={2}
  explanation="peft (parameter-efficient fine-tuning) provides LoRA, QLoRA and friends. trl provides training recipes like SFT on top."
/>

<MultipleChoice
  id="hf-2"
  question="Why should you use `tok.apply_chat_template(...)` instead of writing prompts like 'User: ... Assistant:' by hand?"
  options={[
    "It's faster",
    "Each instruct model was trained on a specific format with special tokens; the wrong format makes it behave strangely",
    "Hand-written prompts are not allowed",
    "It translates the prompt into English",
  ]}
  answer={1}
  explanation="A mismatched chat template is one of the most common reasons a model (especially a fine-tuned one) misbehaves."
/>

<MultipleChoice
  id="hf-3"
  question="About how much memory do the weights of a 1.5B-parameter model take in bf16?"
  options={["0.75 GB", "1.5 GB", "3 GB", "6 GB"]}
  answer={2}
  explanation="bf16 is 2 bytes per parameter: 1.5B × 2 = 3 GB."
/>

<OpenQuestion
  id="hf-4"
  question="After running `print(model)` on a Hugging Face LLM, how would you map what you see onto the GPT you built in Part 5?">

Look for the same parts under different names. For a Llama or Qwen-style model you'd see something like:

| In `print(model)` | In your GPT |
| --- | --- |
| `embed_tokens` | `self.tok` (token embeddings) |
| `layers` (a list of N decoder layers) | `self.blocks` |
| `self_attn` with `q_proj`, `k_proj`, `v_proj`, `o_proj` | the attention's query, key, value and output projections |
| `mlp` with `gate_proj`, `up_proj`, `down_proj` | the feed-forward MLP (SwiGLU has an extra "gate" matrix) |
| `input_layernorm`, `post_attention_layernorm` (RMSNorm) | `ln1`, `ln2` |
| `norm` | `ln_f` |
| `lm_head` | `self.head` |

There's no position embedding table because these models use RoPE inside attention instead. Otherwise, it's the same design at a larger scale.

</OpenQuestion>

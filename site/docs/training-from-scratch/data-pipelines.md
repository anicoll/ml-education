---
sidebar_position: 2
title: Data for language models
---

# Data for language models

For language models, data work means: get text, clean it, tokenise it, and feed it to the GPU fast enough that the GPU never waits.

## Where training text comes from

- **Small and classic:** Tiny Shakespeare (1 MB), perfect for first experiments.
- **Children's stories:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), synthetic simple stories designed so that very small models can learn coherent English. Excellent for a 3070.
- **Web scale:** FineWeb, The Pile, RedPajama. Billions to trillions of tokens. Far more than you need, but you can stream a slice.
- **Your own:** your notes, a book, a codebase. Small, but fun.

The [Hugging Face Hub](https://huggingface.co/datasets) hosts most of these:

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
print(ds[0]["text"])
```

## Cleaning

Real pre-training pipelines spend enormous effort here: removing duplicates, filtering low-quality or toxic text, stripping boilerplate, filtering by language. For your experiments, at minimum: deduplicate, and look at random samples of your data yourself. You will find surprises.

## Tokenise once, up front

Tokenising on the fly during training wastes time. Tokenise the whole dataset once, concatenate everything into one long array of token IDs, and save it to disk:

```python
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = []
for example in ds:
    ids.extend(tok.encode(example["text"]))
    ids.append(tok.eos_token_id)          # mark the boundary between documents

arr = np.array(ids, dtype=np.uint16)       # GPT-2 vocab (50257) fits in uint16
arr.tofile("train.bin")
```

## Sampling batches

Because everything is one long stream, a batch is just random windows of length $T+1$: the first $T$ tokens are inputs and the same window shifted by one is the target.

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")

def get_batch(batch_size=32, block_size=256):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
```

`np.memmap` reads from disk on demand, so the dataset doesn't need to fit in RAM.

## How much data?

A useful rule of thumb from the **Chinchilla** paper: for the best model for a fixed compute budget, train on about **20 tokens per parameter**.

| Model size | Compute-optimal tokens |
| --- | --- |
| 10M | 200M |
| 100M | 2B |
| 1B | 20B |

Modern small models are trained far beyond this (Llama 3 8B saw 15 trillion tokens) because a smaller model that's been over-trained is cheaper to *run*. For your experiments, 20 tokens per parameter is a good target.

## Throughput budget on a 3070

As a rough guide, a 3070 trains a 10M-parameter GPT at somewhere around 100k–200k tokens per second with bf16 and `torch.compile`. 200M tokens then takes about 20–30 minutes. A 100M-parameter model at 2B tokens would take a day or more. Measure your own numbers; they depend heavily on sequence length and batch size.

:::tip Try it
Download TinyStories, tokenise the first 100k stories with the GPT-2 tokeniser, and save `train.bin` and `val.bin` in `notebooks/05-training/data/`. How many tokens did you get? How many parameters would be compute-optimal for that many tokens?
:::

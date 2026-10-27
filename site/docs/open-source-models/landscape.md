---
sidebar_position: 1
title: The open-model landscape
---

# The open-model landscape

Training a capable LLM from scratch costs millions of dollars. But many organisations release their trained models' weights for anyone to download, run and modify. This is the world you'll work in for Part 6.

## "Open" means different things

- **Open weights:** you can download the trained parameters. Most "open" LLMs are this. The training data and code are usually *not* released.
- **Open source (fully):** weights, training code and data are all released (e.g. OLMo from AI2, Pythia from EleutherAI). Great for learning and research.
- **Licences vary.** Some are permissive (Apache 2.0, MIT). Some have custom terms with usage restrictions or limits for very large companies. Check before using a model for anything beyond personal learning.

## Base vs instruct models

Most model families are released in two forms:

- **Base** (sometimes "pretrained"): the raw output of pre-training. It continues text. Ask it a question and it might answer, or write five more similar questions.
- **Instruct** (or "chat", "-it"): base plus supervised fine-tuning and preference tuning. It follows instructions and holds a conversation.

For most fine-tuning projects, **start from an instruct model**. Start from a base model when you want to teach a completely different format or style and don't want the assistant behaviour.

## Model families worth knowing

New releases appear constantly, so check the [Hugging Face Hub](https://huggingface.co/models) and leaderboards for what's current. Families that have been consistently strong and come in sizes that suit an 8 GB GPU:

| Family | From | Notes |
| --- | --- | --- |
| Llama | Meta | The model family that kicked off the open-weights ecosystem. Huge tooling support. |
| Qwen | Alibaba | Wide range of sizes from under 1B up, strong at code and maths, permissive licences on most sizes. |
| Gemma | Google | Small, high-quality models. |
| Mistral / Ministral | Mistral AI | Efficient models, some Apache-licensed. |
| Phi | Microsoft | Small models trained heavily on high-quality and synthetic data. |
| SmolLM | Hugging Face | Tiny fully open models (135M–3B), excellent for learning on limited hardware. |
| DeepSeek | DeepSeek | Strong reasoning models; the big ones are far too large for a 3070, but distilled small versions exist. |

## Sizes and what they mean for you

The number in a model name (1.5B, 8B, 70B) is its parameter count. On an 8 GB RTX 3070, as a rough guide:

| Size | Run it (inference) | Fine-tune it |
| --- | --- | --- |
| Under 1B | Easily, even at full precision | Full fine-tuning possible |
| 1–3B | Easily | LoRA comfortably; QLoRA very comfortably |
| 7–8B | At 4-bit quantisation | QLoRA only, with short sequences and small batches |
| 13–14B | 4-bit, tight, with small context | Not really practical |
| 30B+ | Only with heavy CPU offload (slow) | Rent a GPU |

On a 48 GB Apple Silicon Mac everything shifts up a lot: 27–32B models run comfortably at 4-bit, and 7–8B models can be LoRA fine-tuned without quantisation. See [Choosing your machine](../hardware/overview.md), and [VRAM maths](../hardware/vram-math.md) for where these numbers come from.

## Model cards

Every model on the Hub has a **model card**. Read it. It tells you the licence, the intended use, the **chat template** (the exact format for conversations, which matters a lot for fine-tuning), the context length and known limitations.

:::tip Try it
Browse the Hub and pick three instruct models between 1B and 8B parameters. For each, write down in `notes/`: its licence, context length, chat template, and the size of its weight files. Which would you choose to fine-tune first, and why?

Here's what one row might look like. Check the model card yourself, as details change between releases:

| Model | Licence | Context | Chat template | Weights (bf16) |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-1.5B-Instruct` | Apache 2.0 | 32k tokens | ChatML (`im_start` / `im_end` tokens) | About 3 GB |

A good first choice is small (fast to iterate on), permissively licensed, and has a well-documented chat template.
:::

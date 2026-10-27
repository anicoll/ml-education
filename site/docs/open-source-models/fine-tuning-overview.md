---
sidebar_position: 5
title: "Fine-tuning: when and how"
---

# Fine-tuning: when and how

**Fine-tuning** means continuing to train a pre-trained model on your own, smaller dataset so it gets better at a specific task, style or format. It's the same training loop as always, just starting from good weights instead of random ones.

## First: do you need to fine-tune?

Fine-tuning is often not the first tool to reach for. Try these in order:

1. **Prompting.** Clear instructions, a few examples in the prompt ("few-shot"), a system prompt. Free and instant.
2. **Retrieval (RAG).** If the model needs *knowledge* it doesn't have (your documents, recent facts), fetch relevant text and put it in the prompt. Fine-tuning is a poor way to add facts; models learn them unreliably and hallucinate around them.
3. **Fine-tuning.** When you need to change *behaviour*: a consistent format or style, a narrow task done reliably, a smaller model matching a bigger one on your task, or shorter prompts.

| Goal | Best tool |
| --- | --- |
| "Answer questions about my documents" | RAG |
| "Always reply in this JSON schema" | Fine-tuning (or structured output features) |
| "Write in my personal style" | Fine-tuning |
| "Get a 3B model to do one task as well as a much bigger model" | Fine-tuning on outputs from the bigger model (distillation) |
| "Know about events after its training cutoff" | RAG |

For learning purposes, of course, you should fine-tune anyway. It's the best way to understand how these models work.

## Kinds of fine-tuning

- **Supervised fine-tuning (SFT):** train on example prompts and ideal responses. The loss is usually computed only on the response tokens. This is what you'll do first.
- **Preference tuning (DPO, ORPO, KTO):** train on pairs of a better and a worse response to the same prompt, nudging the model towards the preferred one. Usually done after SFT.
- **Continued pre-training:** keep training on raw text from a new domain (medical papers, a programming language) before SFT. Needs much more data.
- **RLHF / RL with verifiable rewards:** reinforcement learning against a reward model or a checker (e.g. "did the code pass the tests?"). How reasoning models are trained. More advanced.

## Full fine-tuning vs parameter-efficient fine-tuning

**Full fine-tuning** updates every weight. Best quality, but very memory-hungry: weights, gradients and AdamW's two extra values per parameter add up to roughly **16 bytes per parameter** in mixed precision, plus activations. A 1B model needs about 16 GB, already more than your 3070.

**Parameter-efficient fine-tuning (PEFT)** freezes the original weights and trains a small number of new ones. The dominant method is **LoRA**, and its memory-saving variant **QLoRA**. Quality is close to full fine-tuning for most tasks, at a fraction of the memory. On an 8 GB card, this is what you'll use. See [LoRA and QLoRA](./lora-and-qlora.md).

## Data: the part that matters most

The quality of your fine-tune depends far more on your data than on hyperparameters.

- **Format:** a list of conversations, usually as JSON Lines:

```json
{"messages": [{"role": "system", "content": "You are a concise tutor."}, {"role": "user", "content": "What is overfitting?"}, {"role": "assistant", "content": "When a model memorises its training data..."}]}
```

- **Quantity:** a few hundred to a few thousand high-quality examples is often enough for a style or format change. Narrow tasks can work with less.
- **Quality:** every example teaches. Inconsistent, sloppy or wrong examples teach inconsistency, sloppiness and errors. Read through your data.
- **Coverage:** include the range of inputs you expect, including edge cases and examples of when to refuse or say "I don't know".
- **Hold out a test set** of prompts the model never trains on.

## Catastrophic forgetting

Training hard on a narrow dataset can make the model worse at everything else. Mitigations: lower learning rates, fewer epochs (1–3 is typical), LoRA (which naturally limits how far the model can move), and mixing in some general-purpose data.

:::tip Try it
Pick a fine-tuning project for yourself and write a one-page plan in `notes/fine-tuning-plan.md`: the behaviour you want, why prompting alone isn't enough, where you'll get 300–1,000 examples, and how you'll judge success. Some ideas: a model that explains ML concepts in your preferred style, one that converts informal notes into structured JSON, or one that writes in the voice of your favourite author.
:::

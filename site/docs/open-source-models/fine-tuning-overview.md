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

## Worked example: a fine-tuning plan

Here's a complete plan for one project, to show the level of detail that's useful.

**Project: commit messages from diffs.**

- **Behaviour wanted:** given a `git diff`, write a commit message in a consistent house style: a short imperative summary line under 72 characters, a blank line, then a few lines explaining *why*.
- **Why prompting alone isn't enough:** a big model can do this with a good prompt, but a small local model (1.5–3B) drifts from the format, writes vague summaries ("Update files") and describes *what* changed instead of *why*. Fine-tuning should make the format reliable and let a small, fast model do the job.
- **Where the data comes from:** open-source repositories with good commit hygiene. Take (diff, message) pairs from their git history with a script, and filter out merge commits, huge diffs (over about 1,500 tokens), and messages under 20 characters. Aim for 1,000 training examples and 100 test examples from a *different* repository, to check the model generalises.
- **Model and method:** `Qwen2.5-1.5B-Instruct` with QLoRA on the 3070, or LoRA via MLX on the Mac. Two epochs, rank 16.
- **How success is judged:**
  1. Format: the percentage of outputs with a summary line under 72 characters, in the imperative, followed by a blank line. A script can check this.
  2. Quality: score 30 test diffs by hand on a 0–2 scale for "summary is accurate" and "explains why". Compare with the base model and with a big model given a careful prompt.
  3. Regression check: run the ten prompts from [Running models locally](./running-locally.md) to make sure general ability hasn't collapsed.

### More project ideas, each with a data source

| Project | Input → output | Where 300–1,000 examples come from |
| --- | --- | --- |
| ML tutor in your style | ML question → explanation pitched the way you like | Write 50 by hand, generate more with a big model, then edit them. |
| Notes to JSON | Messy meeting notes → `{"decisions": [...], "actions": [...], "owners": [...]}` | Your own notes, plus a big model drafting the JSON that you then correct. |
| Incident summariser | Alert and log excerpt → one-paragraph incident summary | Past incident tickets and postmortems (scrub anything sensitive first). |
| Bank transaction categoriser | Raw transaction line → category | Your categorised bank exports (see [What is machine learning?](../foundations/what-is-ml.md#4-expense-categorisation)). |
| Author's voice | Topic → paragraph in that author's style | Public-domain books from Project Gutenberg, split into passages, with a big model writing a "topic" for each one. |

:::tip Try it
Pick a fine-tuning project for yourself and write a one-page plan in `notes/fine-tuning-plan.md`, following the structure of the worked example: the behaviour you want, why prompting alone isn't enough, where you'll get 300–1,000 examples, which model and method, and how you'll judge success. Borrowing one of the ideas above is completely fine.
:::

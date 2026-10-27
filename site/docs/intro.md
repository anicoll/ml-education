---
sidebar_position: 0
title: How to use this site
slug: /intro
---

# How to use this site

This is a self-paced course on machine learning, written for someone who can program but has not studied ML before. It goes from "what is a model, really?" all the way to fine-tuning an open-source language model on your own RTX 3070.

## The big picture in one paragraph

Machine learning is **writing programs whose behaviour is learned from data instead of written by hand**. You choose a flexible function with lots of adjustable numbers (the *parameters*), you define a score for how wrong it is on your data (the *loss*), and you repeatedly nudge the parameters to make the loss smaller (the *optimiser*). Everything else, from linear regression to GPT, is a variation on that loop: different functions, different data, different scales.

## How the curriculum is organised

| Part | What you learn | Hands-on outcome |
| --- | --- | --- |
| 1. Foundations | What ML is, the core loop, the maths you need | Fit a line with gradient descent in plain Python |
| 2. Classical ML | Regression, classification, trees, evaluation | Build logistic regression from scratch; use scikit-learn |
| 3. Neural networks | Neurons, backprop, optimisers, PyTorch | Train an MLP on MNIST on your GPU |
| 4. Architectures | CNNs, embeddings, attention, transformers | Build a CNN and a self-attention layer |
| 5. Training from scratch | Training loops, data, debugging, efficiency | Train a tiny GPT on your own text |
| 6. Open-source models | Hugging Face, quantisation, LoRA, QLoRA | QLoRA fine-tune a small LLM on the 3070 |
| 7. Your RTX 3070 | VRAM budgeting, setup, when to rent | A working CUDA + PyTorch environment |

## How each lesson works

Each page follows the same pattern:

1. **Intuition first.** A plain-English explanation and, where it helps, a picture.
2. **Then the maths**, kept to what you need, with every symbol explained.
3. **Then code**, usually small enough to type out yourself.
4. **A "Try it" box** pointing at an exercise you should do in the repo's `notebooks/` folder.

:::tip Try it
Exercises look like this. They tell you what to build, not just what to read. Do them. Reading about ML without writing code is like reading about swimming.
:::

:::info Key idea
Boxes like this summarise the one thing to remember from a section.
:::

## Where things live in the repo

```text
ml-education/
├── site/        ← this website (Docusaurus)
├── notebooks/   ← exercises that go with each lesson
├── projects/    ← your own bigger experiments
└── notes/       ← your own notes, questions, things to revisit
```

To run the site locally:

```bash
cd site
npm install   # first time only
npm start     # opens http://localhost:3000
```

## Suggested pace

You don't need to finish all the theory before touching real models. A good rhythm is:

- **Weeks 1–2:** Parts 1 and 2. Get comfortable with the core loop.
- **Weeks 3–4:** Part 3, plus setting up your GPU (Part 7). Train something on the 3070.
- **Week 5:** Jump ahead and run a model locally (Part 6, "Running models locally"). It's motivating.
- **Weeks 6–8:** Part 4 and Part 5, ending with the tiny GPT capstone.
- **Weeks 9–10:** Part 6 properly: your first QLoRA fine-tune.

See the [learning roadmap](./roadmap.md) for a more detailed path and milestones.

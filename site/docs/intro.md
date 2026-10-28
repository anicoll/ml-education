---
sidebar_position: 0
title: How to use this site
slug: /intro
---

# How to use this site

This is a self-paced course on machine learning, written for someone who can program but has not studied ML before. It goes from "what is a model, really?" all the way to fine-tuning an open-source language model on your own hardware: a PC with an NVIDIA GPU (Windows with WSL2, or Ubuntu) or an Apple Silicon Mac.

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
| 6. Open-source models | Hugging Face, quantisation, LoRA, QLoRA | QLoRA fine-tune an LLM on an NVIDIA GPU or a Mac |
| 7. Reinforcement learning | Rewards, Q-learning, DQN, policy gradients, PPO | Train agents in Gymnasium, then an AI that drives in Trackmania |
| 8. Your hardware | NVIDIA vs Apple Silicon, memory budgeting, setup, when to rent | A working PyTorch environment on CUDA and/or MPS |

## How each lesson works

Each page follows the same pattern:

1. **Intuition first.** A plain-English explanation and, where it helps, a picture.
2. **Then the maths**, kept to what you need, with every symbol explained.
3. **Then code**, usually small enough to type out yourself.
4. **A "Try it" box** pointing at an exercise you should do in the repo's `notebooks/` folder.
5. **Check your understanding** questions at the end:
   - **Multiple choice** questions tell you straight away whether you were right, and why.
   - **Open questions** ask you to write an answer in your own words. Once you save it, a suggested answer (often a fully worked example) appears so you can compare.
6. A **Mark lesson complete** button. Completed lessons get a tick in the sidebar, and [My progress](/progress) shows everything you've done.

Your answers and progress are saved in this browser only (in its local storage), so nothing is sent anywhere. If you switch browsers or clear your browsing data you'll lose them, so use **Export progress** on the [My progress](/progress) page to keep a backup.

:::tip[Try it]
Exercises look like this. They tell you what to build, not just what to read. Do them. Reading about ML without writing code is like reading about swimming.
:::

:::info[Key idea]
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

The site is published to GitHub Pages automatically. To run it locally instead:

```bash
cd site
npm install   # first time only
npm start     # opens http://localhost:3000
```

## Suggested pace

You don't need to finish all the theory before touching real models. A good rhythm is:

- **Weeks 1–2:** Parts 1 and 2. Get comfortable with the core loop.
- **Weeks 3–4:** Part 3, plus setting up your machines (Part 8). Train something on the GPU.
- **Week 5:** Jump ahead and run a model locally (Part 6, "Running models locally"). It's motivating.
- **Weeks 6–8:** Part 4 and Part 5, ending with the tiny GPT capstone.
- **Weeks 9–10:** Part 6 properly: your first QLoRA fine-tune.
- **Weeks 11–14:** Part 7, reinforcement learning, ending with the Trackmania capstone.

See the [learning roadmap](./roadmap.md) for a more detailed path and milestones.

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="LPZh9BOjkQs" title="Large Language Models explained briefly" channel="3Blue1Brown">

An eight-minute taste of where this course ends up: what a language model is and roughly how it's trained. Don't worry if parts go over your head; every idea in it gets its own lesson here.

</YouTube>

## Check your understanding

Try these to see how the questions work.

<MultipleChoice
  id="intro-1"
  question="What are the three ingredients of the loop every ML model is trained with?"
  options={[
    "A database, a web server and a cache",
    "A function with adjustable parameters, a loss that scores how wrong it is, and an optimiser that nudges the parameters to reduce the loss",
    "A GPU, a dataset and a tokenizer",
    "Rules, if-statements and tests",
  ]}
  answer={1}
  explanation="Model, loss, optimiser. Everything from linear regression to GPT is a variation on that loop."
/>

<OpenQuestion
  id="intro-2"
  question="Why do you want to learn machine learning? Write a sentence or two; you can come back and compare later.">

There's no right answer here. This one is for you. Some common goals, to compare with yours:

- "I want to understand what's actually happening inside tools like ChatGPT, not just use them."
- "I want to fine-tune a small open model to do one task at work reliably and cheaply."
- "I want to build something fun, like an AI that drives in a racing game."

Revisit this when you finish the course and see how your answer has changed.

</OpenQuestion>

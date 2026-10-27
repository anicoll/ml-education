---
sidebar_position: 8
title: When to rent a GPU
---

# When to rent a GPU

A typical home machine (an 8 GB NVIDIA card or an Apple Silicon Mac) covers most of this course. For the rest, renting is cheap and often the smarter choice.

## When renting makes sense

- The run doesn't fit on your machine (for example anything 70B), or it needs CUDA-only tooling and doesn't fit in your card's VRAM.
- The run fits but would take days locally and hours on a bigger card.
- You want to try multi-GPU training to understand it.

A rough rule: if you've spent more than an hour fighting memory limits, an hour on a big GPU would have been cheaper.

## Options

| Type | Examples | Notes |
| --- | --- | --- |
| Free notebooks | Google Colab, Kaggle notebooks | Free T4 (16 GB) GPUs with time limits. Good for small experiments. |
| GPU marketplaces | RunPod, Vast.ai, Lambda | Rent by the hour. Consumer cards (RTX 4090/5090, 24–32 GB) are cheap; data-centre cards (A100, H100, 80 GB) cost more. |
| Big clouds | AWS, GCP, Azure | More setup and usually pricier for individuals. |

Prices change often, so check current rates. As a ballpark, 24 GB consumer cards rent for well under a dollar an hour, and an 80 GB data-centre GPU for a few dollars an hour.

## What more VRAM buys you

| VRAM | Comfortable QLoRA | Comfortable LoRA (bf16 base) |
| --- | --- | --- |
| 8 GB (common consumer card) | 3B, tight 8B | 1.5B |
| ~36–40 GB usable (a 48 GB Mac, via MLX) | 14B, careful 32B | 8B |
| 24 GB | 14B, comfy 8B with long sequences | 8B |
| 48 GB | 32B | 14B |
| 80 GB | 70B | 32B |

## A workflow that saves money

1. **Develop locally** with a small model and a slice of data until the script runs end to end without errors.
2. **Rent** a bigger GPU only for the full run.
3. **Push your code** to this repo, clone it on the rented machine, and run the same script with a bigger model in the config.
4. **Download the result** (a LoRA adapter is just tens of MB) and shut the machine down. Most providers charge while it's running, even if idle.

## Check your understanding

<MultipleChoice
  id="cloud-1"
  question="What's the money-saving workflow for a big fine-tuning run?"
  options={[
    "Rent the biggest GPU first and develop on it",
    "Develop locally on a small model and data slice until it runs end to end, then rent a big GPU only for the full run",
    "Never rent; just wait longer",
    "Use the free tier for the full run",
  ]}
  answer={1}
  explanation="Debugging on a rented GPU means paying by the hour to fix typos. Rent only for the run itself, then download the adapter and shut the machine down."
/>

<MultipleChoice
  id="cloud-2"
  question="You finished training on a rented GPU and downloaded the adapter. What should you do next?"
  options={["Leave the machine running in case you need it", "Shut it down: most providers charge while it's running, even if idle", "Delete the adapter", "Nothing"]}
  answer={1}
  explanation="An idle rented GPU costs the same as a busy one."
/>

<OpenQuestion
  id="cloud-3"
  question="When would you rent a GPU rather than use your own machine? Give two concrete situations.">

Any two of these, for example:

1. **It doesn't fit.** You want to QLoRA fine-tune a 14B model, but your card has 8 GB. A rented 24 GB GPU fits it comfortably, for well under a dollar an hour.
2. **It fits, but would take too long.** Training a 100M-parameter GPT on 2B tokens would take a day or more on a consumer GPU, and a few hours on a data-centre card.
3. **It needs CUDA and you only have a Mac.** For example, a tutorial built on bitsandbytes or vLLM.
4. **You want to learn multi-GPU training**, which you can't try with one GPU at home.

A quick test: if you've spent more than an hour fighting memory limits, an hour of rented GPU would have been cheaper.

</OpenQuestion>

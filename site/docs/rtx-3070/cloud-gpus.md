---
sidebar_position: 4
title: When to rent a GPU
---

# When to rent a GPU

Your 3070 covers most of this course. For the rest, renting is cheap and often the smarter choice.

## When renting makes sense

- The run doesn't fit in 8 GB, even with QLoRA, checkpointing and short sequences (for example QLoRA on 13B+, or 8B with long contexts).
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
| 8 GB (your 3070) | 3B, tight 8B | 1.5B |
| 24 GB | 14B, comfy 8B with long sequences | 8B |
| 48 GB | 32B | 14B |
| 80 GB | 70B | 32B |

## A workflow that saves money

1. **Develop locally** on the 3070 with a small model and a slice of data until the script runs end to end without errors.
2. **Rent** a bigger GPU only for the full run.
3. **Push your code** to this repo, clone it on the rented machine, and run the same script with a bigger model in the config.
4. **Download the result** (a LoRA adapter is just tens of MB) and shut the machine down. Most providers charge while it's running, even if idle.

---
sidebar_position: 1
title: Anatomy of a real training run
---

# Anatomy of a real training run

The [core loop](../foundations/the-core-loop.md) is five lines. A real training script wraps those lines in everything needed to run for hours, recover from crashes and tell you what's going on. This page is a checklist of those pieces.

## The skeleton

```python
import math, time, torch
import torch.nn.functional as F

def lr_at(step, max_lr, warmup, total):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))   # cosine decay

model = GPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

for step in range(total_steps):
    # 1. learning rate schedule
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step, 3e-4, warmup=200, total=total_steps)

    # 2. forward and backward, with mixed precision and gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accum_steps):
        x, y = get_batch("train")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):   # device is "cuda" or "mps"
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / accum_steps
        loss.backward()

    # 3. gradient clipping, then update
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # 4. logging
    if step % 50 == 0:
        print(f"step {step} loss {loss.item() * accum_steps:.3f} grad_norm {grad_norm:.2f}")

    # 5. periodic evaluation and checkpointing
    if step % 500 == 0:
        val_loss = estimate_loss("val")
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "step": step}, "ckpt.pt")
```

## The pieces, and why each exists

| Piece | Why |
| --- | --- |
| Learning rate warmup and decay | Stable start, better final loss. See [Optimisers](../neural-networks/optimisers.md). |
| Mixed precision (`autocast`) | Roughly halves memory for activations and runs much faster on tensor cores. |
| Gradient accumulation | Simulate a large batch on a small GPU. |
| Gradient clipping | Stops a single bad batch from blowing up training. |
| Validation loss | The only honest signal of whether the model is improving. |
| Checkpoints | Training crashes. Power cuts happen. Save regularly, including optimiser state, so you can resume. |
| Logging | You can't fix what you can't see. |

## Experiment tracking

Printing to the terminal works for a while. Once you're comparing runs, use a tracker:

- **TensorBoard:** local, simple, free. `pip install tensorboard`.
- **Weights & Biases (wandb):** hosted, very convenient, free for personal use.

Log at least: training loss, validation loss, learning rate, gradient norm, and throughput (tokens or examples per second).

## Reproducibility

Set seeds so runs can be repeated, and record the config (hyperparameters, dataset version, git commit) with each run:

```python
torch.manual_seed(1337)
```

GPU operations are not perfectly deterministic, so expect small differences between runs even with fixed seeds. Big differences mean something else changed.

## `torch.compile`

PyTorch 2 can compile your model into faster fused kernels. It often gives a 1.3–2× speed-up for free, at the cost of a slow first step:

```python
model = torch.compile(model)
```

On Windows this needs WSL2 or a recent PyTorch build with Triton support. On a Mac, MPS support is limited, so skip `torch.compile` there. See [Choosing your machine](../hardware/overview.md).

:::tip[Try it]
Take your MNIST script from Part 3 and add: a cosine learning rate schedule with warmup, bf16 autocast, gradient clipping, a validation loss check every epoch, checkpoint saving, and TensorBoard logging. Kill the script halfway, then make it resume from the checkpoint.
:::

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="l8pRSuU81PU" title="Let's reproduce GPT-2 (124M)" channel="Andrej Karpathy">

A real training run built up live. Most of this lesson's pieces appear: mixed precision, `torch.compile`, gradient accumulation, a warmup-plus-cosine learning-rate schedule and evaluation during training. Long, so it's fine to dip into the sections you need.

</YouTube>

## Check your understanding

<MultipleChoice
  id="loop-1"
  question="Why does the checkpoint save the optimiser's state as well as the model's weights?"
  options={[
    "It doesn't need to; that's just a habit",
    "AdamW keeps running averages per parameter; without them a resumed run restarts those statistics and training is disrupted",
    "The optimiser state contains the training data",
    "So the checkpoint file is bigger",
  ]}
  answer={1}
  explanation="To resume exactly where you left off you need the weights, the optimiser's running averages, and the step number (for the learning rate schedule)."
/>

<MultipleChoice
  id="loop-2"
  question="What does `clip_grad_norm_(model.parameters(), 1.0)` protect against?"
  options={["Overfitting", "A single bad batch producing a huge gradient that wrecks the weights", "Running out of disk space", "Slow data loading"]}
  answer={1}
  explanation="If the total gradient size is above 1.0, it's scaled down to 1.0 before the update. The returned grad_norm is also worth logging."
/>

<MultipleChoice
  id="loop-3"
  question="What does mixed precision (bf16 autocast) give you?"
  options={["Better accuracy than fp32", "Roughly half the activation memory and much faster maths on tensor cores", "Automatic checkpointing", "A learning rate schedule"]}
  answer={1}
  explanation="bf16 values take 2 bytes instead of 4, and modern GPUs have dedicated hardware for low-precision matrix maths."
/>

<OpenQuestion
  id="loop-4"
  question="List the five things you should log in every training run, and say what problem each one helps you spot.">

- **Training loss:** whether the model is learning at all. Flat means something's broken; NaN means instability.
- **Validation loss:** the honest signal. If it rises while training loss falls, you're overfitting.
- **Learning rate:** confirms the schedule (warmup and decay) is doing what you think. A bug here is easy to miss otherwise.
- **Gradient norm:** a sudden jump often comes just before a loss spike; near zero suggests gradients aren't flowing.
- **Throughput (tokens or examples per second):** spots a slow data pipeline or accidental CPU work, and tells you how long the run will take.

</OpenQuestion>

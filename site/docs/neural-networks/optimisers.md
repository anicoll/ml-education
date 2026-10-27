---
sidebar_position: 3
title: Optimisers and learning rates
---

# Optimisers and learning rates

Plain gradient descent works, but on large networks it's slow and fussy. Modern optimisers add a couple of tricks.

## SGD with momentum

Instead of stepping purely along the current gradient, keep a running average of recent gradients and step along that:

$$
\begin{aligned}
v &\leftarrow \beta v + \nabla_\theta L \\
\theta &\leftarrow \theta - \eta v
\end{aligned}
$$

Think of a ball rolling downhill. It builds up speed in directions where the gradient is consistent and smooths out zig-zagging in directions where it keeps flipping sign. $\beta$ is typically 0.9.

## Adam and AdamW

**Adam** keeps two running averages per parameter: the gradient (momentum) and the squared gradient (how large gradients typically are). It divides the step by the square root of the second one, so each parameter gets its own effective learning rate. Parameters with consistently tiny gradients get bigger steps, noisy ones get smaller steps.

**AdamW** is Adam with weight decay applied correctly. It's the default optimiser for training transformers, and the safe first choice for most deep learning.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

:::note[Memory cost]
Adam stores two extra numbers for every parameter. With 32-bit floats that's 8 extra bytes per parameter, which matters a lot when you're budgeting VRAM. More in [VRAM maths](../hardware/vram-math.md).
:::

## The learning rate is the most important hyperparameter

If you only tune one thing, tune this. Typical starting points:

| Setting | Learning rate |
| --- | --- |
| SGD on small networks | 0.01–0.1 |
| AdamW, training from scratch | 1e-4 to 1e-3 (3e-4 is a classic default) |
| AdamW, full fine-tuning a pre-trained model | 1e-5 to 5e-5 |
| LoRA fine-tuning | 1e-4 to 2e-4 |

Fine-tuning uses smaller rates because the model already knows a lot, and big steps would wreck that knowledge.

## Learning rate schedules

The best learning rate changes during training. A standard schedule for transformers:

1. **Warmup:** increase the learning rate linearly from near zero over the first few hundred steps. Early gradients are large and erratic, and Adam's statistics aren't settled yet.
2. **Decay:** decrease it gradually, commonly following a cosine curve, to near zero by the end. Smaller steps let the model settle into a good minimum.

```python
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=10_000)
# call scheduler.step() after each optimizer.step()
```

## Batch size

- Larger batches give smoother gradient estimates and use the GPU more efficiently, but need more memory.
- If the batch you want doesn't fit in VRAM, use **gradient accumulation**: run several small batches, add up their gradients, then take one optimiser step. Mathematically equivalent to one big batch. You'll use this constantly on an 8 GB card.

```python
accum_steps = 4
for i, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

:::tip[Try it]
Train the same small MLP on MNIST with SGD (lr 0.1), SGD with momentum, and AdamW (lr 1e-3). Plot all three loss curves on one chart. Then try AdamW at lr 1e-1 and 1e-5 to see "too high" and "too low" in practice.
:::

## Check your understanding

<MultipleChoice
  id="optim-1"
  question="What's a sensible starting learning rate for full fine-tuning a pre-trained model with AdamW?"
  options={["1.0", "0.1", "3e-4", "2e-5"]}
  answer={3}
  explanation="Full fine-tuning uses small rates (1e-5 to 5e-5) because big steps would wreck what the model already knows. 3e-4 is a from-scratch default."
/>

<MultipleChoice
  id="optim-2"
  question="Why is learning rate warmup used at the start of transformer training?"
  options={[
    "To save memory",
    "Early gradients are large and erratic and Adam's statistics aren't settled, so starting with tiny steps avoids destabilising the model",
    "To make training finish faster",
    "Warmup is only for inference",
  ]}
  answer={1}
  explanation="Ramping the learning rate up over the first few hundred steps prevents early chaos. A decay (often cosine) then shrinks it towards the end."
/>

<MultipleChoice
  id="optim-3"
  question="You want an effective batch size of 64 but only 16 examples fit in memory. What do you do?"
  options={[
    "Give up and use 16",
    "Gradient accumulation: run 4 batches of 16, adding up gradients, then take one optimiser step",
    "Raise the learning rate 4×",
    "Use 4 times more epochs",
  ]}
  answer={1}
  explanation="Accumulating gradients over 4 small batches is mathematically equivalent to one batch of 64, at the cost of time rather than memory."
/>

<OpenQuestion
  id="optim-4"
  question="Explain momentum using the ball-rolling-downhill picture. What problem does it solve?">

Plain gradient descent only looks at the slope right under its feet at each step. In a long narrow valley the slope mostly points across the valley, so it zig-zags from side to side and makes slow progress along the valley floor.

Momentum keeps a running average of recent gradients, like a ball that carries its speed. Directions where the gradient keeps pointing the same way (along the valley) build up speed. Directions where it keeps flipping (across the valley) cancel out. The result is faster, smoother progress towards the minimum. β = 0.9 means the ball remembers roughly the last 10 steps.

</OpenQuestion>

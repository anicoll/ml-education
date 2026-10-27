---
sidebar_position: 3
title: Debugging training
---

# Debugging training

Neural networks fail silently. A bug rarely crashes; instead, the model just trains a bit worse than it should, and you never find out. These habits catch most problems.

## Sanity checks before a long run

1. **Check the initial loss.** For a classifier over $K$ classes, a freshly initialised model should have loss close to $\ln K$. For a GPT-2 vocabulary that's $\ln 50257 \approx 10.8$. Much higher means something is wrong with initialisation or the loss.
2. **Overfit a single batch.** Train repeatedly on one small batch. The loss should go to almost zero. If it can't memorise 32 examples, there's a bug, full stop.
3. **Look at your data, as the model sees it.** Decode a batch of token IDs back to text. Display augmented images. Check that inputs and targets line up (are targets shifted by exactly one?).
4. **Start small.** Get a tiny model working end to end on a small dataset before scaling anything up.

## Reading loss curves

| What you see | Likely cause |
| --- | --- |
| Loss doesn't move at all | Learning rate far too low, gradients not flowing (forgot `optimizer.step()`, or detached tensors), or data/labels mismatched. |
| Loss goes to `NaN` or explodes | Learning rate too high, missing gradient clipping, fp16 overflow (use bf16), or a log of zero somewhere. |
| Loss drops then plateaus high | Model too small, learning rate too high to settle, or a bug that limits what it can learn. |
| Train loss falls, validation loss rises | Overfitting. More data, regularisation, or stop earlier. |
| Very noisy loss | Batch too small or learning rate too high. |
| Periodic spikes | Bad examples in the data, or shuffling issues. |

## Classic PyTorch bugs

- Forgetting `optimizer.zero_grad()`, so gradients pile up across steps.
- Forgetting `model.eval()` for evaluation, so dropout and batchnorm misbehave.
- Applying softmax before `F.cross_entropy`, which already applies it.
- Wrong dimension in `softmax(dim=...)` or a reshape that silently scrambles data. (`view` vs `transpose`: they are not the same.)
- Tensors accidentally on CPU, making training mysteriously slow.
- Calling `.item()` or printing GPU tensors every step, which forces the CPU to wait for the GPU.

## Monitoring the gradients

Log the overall gradient norm each step. A sudden jump often comes before a loss spike. Per-layer gradient norms can show if some layers aren't learning (near-zero) or are unstable (huge).

## Out-of-memory errors

`CUDA out of memory` is the error you'll hit most on 8 GB. In rough order of what to try:

1. Reduce batch size and use gradient accumulation to compensate.
2. Use bf16 mixed precision.
3. Reduce sequence length.
4. Enable **gradient checkpointing**: recompute activations during the backward pass instead of storing them. Costs about 30% more compute, saves a lot of memory.
5. Use a smaller model, or for fine-tuning, LoRA/QLoRA.

Watch memory use live with `nvidia-smi -l 1` in a terminal, or from Python with `torch.cuda.max_memory_allocated()`.

:::info Key idea
Always overfit one batch first. It takes two minutes and catches a huge fraction of bugs before you waste hours on a broken run.
:::

:::tip Try it
Deliberately break your MNIST script in each of the "classic bugs" ways above, one at a time, and note in `notes/` what each one does to the loss curve. Next time you see that shape, you'll recognise it.
:::

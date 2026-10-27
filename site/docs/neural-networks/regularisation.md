---
sidebar_position: 4
title: Regularisation, normalisation and initialisation
---

# Regularisation, normalisation and initialisation

These are the "tricks" that turn a network that *can* learn into one that learns reliably. You'll see all of them inside a transformer.

## Initialisation

If all weights start at zero, every neuron in a layer computes the same thing and gets the same gradient, so they never become different. Weights must start **random**.

The scale matters too. Too large and activations explode as they pass through layers; too small and they shrink to nothing. Schemes like **Xavier/Glorot** and **Kaiming/He** pick the scale based on the number of inputs to each layer, keeping activations roughly the same size throughout. PyTorch's layers use sensible defaults, so you rarely set this by hand.

## Normalisation layers

Normalisation layers rescale activations inside the network to have a stable mean and variance. This makes training faster and much less sensitive to the learning rate.

- **BatchNorm** normalises each feature across the batch. Standard in CNNs. Behaves differently in training and evaluation mode, which is a classic source of bugs (remember `model.eval()`).
- **LayerNorm** normalises across the features of each individual example. Standard in transformers.
- **RMSNorm** is a simpler LayerNorm (no mean subtraction), used by Llama and most recent LLMs.

## Residual connections

Instead of $h_{\text{out}} = F(h_{\text{in}})$, a residual block computes:

$$
h_{\text{out}} = h_{\text{in}} + F(h_{\text{in}})
$$

The layer only has to learn a *change* to its input. Gradients can flow straight back through the `+` without being shrunk, which makes networks with hundreds of layers trainable. ResNets introduced this in 2015, and every transformer uses it.

## Dropout

During training, randomly set a fraction of activations (say 10%) to zero on each step. The network can't rely on any single neuron, so it learns more robust, spread-out features. At evaluation time dropout is switched off.

```python
nn.Dropout(p=0.1)
```

Large LLMs trained on huge datasets often use little or no dropout, because they see each example only once and overfitting isn't the main problem. Small models on small data benefit a lot.

## Weight decay

Shrinks weights slightly towards zero each step (the L2 penalty from [Overfitting](../classical-ml/overfitting.md)). Built into AdamW via `weight_decay`.

## Putting it together: a modern block

Here's roughly what one block of a modern network looks like. You'll meet it again in the transformer:

```python
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))   # pre-norm residual block
```

:::info[Key idea]
Random init breaks symmetry. Normalisation keeps activations well-scaled. Residual connections give gradients a highway. Dropout and weight decay fight overfitting. Every one of these exists to make optimisation easier.
:::

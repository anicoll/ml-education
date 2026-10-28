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

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="P6sfmUTpUmc" title="Building makemore Part 3: Activations & Gradients, BatchNorm" channel="Andrej Karpathy">

Shows, with plots, why initialisation scale matters, what dead and saturated activations look like, and why batch normalisation was such a big deal. Pairs perfectly with this lesson.

</YouTube>

<YouTube id="Q81RR3yKn30" title="Regularization Part 1: Ridge (L2) Regression" channel="StatQuest with Josh Starmer">

The idea behind weight decay, shown on simple regression: a small penalty on large weights trades a little bias for much less variance.

</YouTube>

## Check your understanding

<MultipleChoice
  id="reg-1"
  question="Why can't you initialise all of a network's weights to zero?"
  options={[
    "Zero weights use too much memory",
    "Every neuron in a layer would compute the same thing and get the same gradient, so they'd never become different",
    "PyTorch doesn't allow it",
    "The loss would be negative",
  ]}
  answer={1}
  explanation="Random initialisation breaks the symmetry so different neurons can learn different features."
/>

<MultipleChoice
  id="reg-2"
  question="What does a residual connection compute?"
  options={["h_out = F(h_in)", "h_out = h_in + F(h_in)", "h_out = h_in × F(h_in)", "h_out = 0 during training"]}
  answer={1}
  explanation="The layer only learns a change to its input, and gradients flow straight back through the + without shrinking. That's what makes very deep networks trainable."
/>

<MultipleChoice
  id="reg-3"
  question="Your model with dropout scores worse at evaluation time than you expect, and results change every run on the same input. What did you probably forget?"
  options={["`model.eval()`", "`optimizer.step()`", "Weight decay", "A learning rate schedule"]}
  answer={0}
  explanation="Without model.eval(), dropout stays on (and BatchNorm keeps using batch statistics), so predictions are randomly degraded."
/>

<MultipleChoice
  id="reg-4"
  question="Which normalisation layer do Llama and most recent LLMs use?"
  options={["BatchNorm", "RMSNorm", "Dropout", "None"]}
  answer={1}
  explanation="RMSNorm is a simplified LayerNorm without mean subtraction. Transformers use LayerNorm-family norms; BatchNorm is standard in CNNs."
/>

<OpenQuestion
  id="reg-5"
  question="Read the `Block` class at the end of this page and describe, line by line, what happens to `x` in `forward`.">

`return x + self.ff(self.norm(x))` does four things:

1. `self.norm(x)`: LayerNorm rescales each example's features to a stable mean and spread, so the next layers see well-behaved numbers.
2. `self.ff(...)`: the feed-forward network. A `Linear` layer expands the size from `dim` to `4 * dim`, `GELU` adds the non-linearity, a second `Linear` shrinks it back to `dim`, and `Dropout` randomly zeroes 10% of values during training.
3. `x + ...`: the residual connection. The block's output is the original input plus the change the feed-forward network computed.
4. The result has the same shape as the input, so blocks like this can be stacked as deep as you like.

Because the norm is applied *before* the feed-forward layer (inside the residual branch), this is called a "pre-norm" block, the layout most modern transformers use.

</OpenQuestion>

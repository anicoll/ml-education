---
sidebar_position: 3
title: Attention
---

# Attention

Attention is the mechanism at the heart of transformers, and so at the heart of every modern LLM. It lets each token look at every other token and decide which ones matter for understanding it.

## The problem attention solves

Take: *"The animal didn't cross the street because **it** was too tired."*

To understand "it", the model needs to connect it to "animal", several words back. Older sequence models (**RNNs** and **LSTMs**) read text one token at a time, squeezing everything seen so far into a single hidden vector. Information from far back gets diluted, and processing is sequential, so it's slow to train on GPUs.

Attention lets every token directly pull information from every other token, in parallel.

## Queries, keys and values

Each token's vector $x_i$ is projected into three new vectors with learned weight matrices:

- **Query** $q_i = W_Q x_i$: "what am I looking for?"
- **Key** $k_i = W_K x_i$: "what do I contain?"
- **Value** $v_i = W_V x_i$: "what will I pass on if you attend to me?"

Think of a library search: your query is compared to every book's key (its catalogue entry), and you take a weighted mix of the matching books' values (their contents).

## The computation

1. **Score** how relevant each token $j$ is to token $i$ with a dot product: $q_i \cdot k_j$.
2. **Scale** by $\sqrt{d_k}$ (the key dimension), to stop scores growing with dimension and saturating softmax.
3. **Softmax** the scores so they become weights that sum to 1.
4. **Mix** the values using those weights.

For all tokens at once, in matrix form:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

$QK^\top$ is a $T \times T$ matrix ($T$ = sequence length) of every token's score against every other. That's why attention cost grows with the **square** of the context length, and why long contexts are expensive in memory.

## Causal masking

A language model predicts the next token, so token $i$ must not see tokens after it (that would be cheating). Before the softmax, set the scores for future positions to $-\infty$, so their weights become 0. This **causal mask** makes the attention matrix lower-triangular.

## Self-attention from scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, dim, head_dim, max_len=256):
        super().__init__()
        self.q = nn.Linear(dim, head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):                         # x: (B, T, dim)
        B, T, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x) # each (B, T, head_dim)
        scores = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5   # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)       # (B, T, T)
        return weights @ v                        # (B, T, head_dim)
```

In real code use the fused, memory-efficient version, which computes the same thing:

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## Multi-head attention

One attention head can only focus on one kind of relationship at a time. **Multi-head attention** runs several heads in parallel (say 12), each with its own $W_Q, W_K, W_V$ and a smaller head dimension, then concatenates their outputs and projects back. Different heads learn different jobs: one might track grammar, another which noun a pronoun refers to, another nearby tokens.

## The KV cache

When generating text one token at a time, the keys and values of earlier tokens don't change. Inference engines store them in a **KV cache** instead of recomputing them. This makes generation fast, but the cache grows with context length and uses a lot of VRAM, which is a key consideration when running models on your 3070.

:::info Key idea
Attention: each token asks a question (query), compares it with every token's label (key), and takes a weighted average of their content (value). Causal masking stops tokens looking at the future.
:::

:::tip Try it
Implement `SelfAttentionHead` in `notebooks/04-architectures/attention.ipynb`. Feed it a random tensor of shape `(2, 8, 32)`, check the output shape, and print the attention weights for one example to confirm the upper triangle is zero. Then verify your output matches `F.scaled_dot_product_attention` with `is_causal=True` using the same q, k and v.
:::

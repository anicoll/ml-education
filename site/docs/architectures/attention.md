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

When generating text one token at a time, the keys and values of earlier tokens don't change. Inference engines store them in a **KV cache** instead of recomputing them. This makes generation fast, but the cache grows with context length and uses a lot of VRAM, which is a key consideration when running models on a GPU with limited memory.

:::info[Key idea]
Attention: each token asks a question (query), compares it with every token's label (key), and takes a weighted average of their content (value). Causal masking stops tokens looking at the future.
:::

:::tip[Try it]
Implement `SelfAttentionHead` in `notebooks/04-architectures/attention.ipynb`. Feed it a random tensor of shape `(2, 8, 32)`, check the output shape, and print the attention weights for one example to confirm the upper triangle is zero. Then verify your output matches `F.scaled_dot_product_attention` with `is_causal=True` using the same q, k and v.
:::

## Check your understanding

<MultipleChoice
  id="attn-1"
  question="In attention, what is the role of the query vector?"
  options={[
    "It describes what a token contains",
    "It describes what a token is looking for; it's compared against every other token's key",
    "It's the information passed on when attended to",
    "It's the position of the token",
  ]}
  answer={1}
  explanation="Query: what am I looking for? Key: what do I contain? Value: what I pass on. Scores are query · key."
/>

<MultipleChoice
  id="attn-2"
  question="You double the context length from 4,000 to 8,000 tokens. Roughly how does the size of the attention score matrix change?"
  options={["It stays the same", "It doubles", "It quadruples", "It halves"]}
  answer={2}
  explanation="The score matrix is T × T, so doubling T multiplies it by four. That quadratic cost is why long contexts are expensive."
/>

<MultipleChoice
  id="attn-3"
  question="What does the causal mask do in a language model?"
  options={[
    "Hides padding tokens",
    "Stops each token attending to tokens that come after it, so the model can't peek at the answer",
    "Removes rare tokens",
    "Randomly drops attention heads",
  ]}
  answer={1}
  explanation="Future positions get a score of −∞ before softmax, so their weight becomes 0 and the attention matrix is lower-triangular."
/>

<MultipleChoice
  id="attn-4"
  question="Why does generating text use a KV cache?"
  options={[
    "To store the training data",
    "Earlier tokens' keys and values don't change, so storing them avoids recomputing them for every new token",
    "To make the model smaller",
    "To hold the optimiser state",
  ]}
  answer={1}
  explanation="It makes generation much faster, at the cost of memory that grows with context length."
/>

<OpenQuestion
  id="attn-5"
  question="Explain attention to a friend who programs but knows no ML, using the library analogy or one of your own.">

A worked explanation:

"Imagine every word in a sentence is a person at a meeting. Each person writes a sticky note saying what they're looking for (the **query**) and wears a badge saying what they know about (the **key**). They also have a notebook of what they'd share (the **value**).

To update their understanding, each person compares their sticky note with everyone's badge. The better the match, the more attention they pay. They then build their new understanding as a blend of everyone's notebooks, weighted by how much attention they paid each one.

For the word 'it' in 'the animal didn't cross the street because it was too tired', the query is roughly 'which noun am I?', and the badge of 'animal' matches strongly, so 'it' ends up carrying information about the animal.

In a language model there's one rule: you can only listen to people who spoke before you (the causal mask). And there are several meetings happening in parallel, each about a different kind of relationship (multi-head attention)."

</OpenQuestion>

---
sidebar_position: 4
title: The transformer
---

# The transformer

The transformer was introduced in the 2017 paper *"Attention Is All You Need"*. Nearly every frontier AI model since, for text, images, audio and code, is built from it. Now that you know about [embeddings](./embeddings.md), [attention](./attention.md) and [residual blocks](../neural-networks/regularisation.md#residual-connections), you have all the parts.

## A decoder-only transformer (GPT-style)

LLMs like GPT, Llama, Qwen and Mistral use the **decoder-only** variant:

```mermaid
flowchart TB
  A["Input token IDs"] --> B["Token embeddings (+ position info)"]
  B --> C
  subgraph C["Transformer block × N"]
    direction TB
    C1["Norm"] --> C2["Masked multi-head self-attention"]
    C2 --> C3(("+ residual"))
    C3 --> C4["Norm"]
    C4 --> C5["Feed-forward MLP"]
    C5 --> C6(("+ residual"))
  end
  C --> D["Final norm"]
  D --> E["Linear layer to vocabulary size"]
  E --> F["Softmax → next-token probabilities"]
```

Each block does two things:

1. **Attention** mixes information *between* tokens.
2. **Feed-forward** (an MLP applied to each token on its own) processes information *within* each token. Most of a model's parameters, and arguably much of its stored knowledge, live here.

Each has a residual connection around it, so the token vectors flow through the network as a "residual stream" that every layer reads from and adds to.

## In code

```python
class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, dim=384, n_heads=6, n_layers=6, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.Sequential(*[Block(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):                       # idx: (B, T) token IDs
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        return self.head(self.ln_f(x))            # (B, T, vocab_size) logits
```

That's a complete GPT. With these default sizes it has about 10M parameters and trains well on a single consumer GPU or an Apple Silicon Mac. You'll do exactly that in the [capstone](../training-from-scratch/capstone-tiny-gpt.md).

## Training objective

At every position, predict the next token. If the input is tokens `[t0, t1, t2, t3]`, the targets are `[t1, t2, t3, t4]`. One sequence of length $T$ gives $T$ training examples at once, thanks to the causal mask. The loss is the cross-entropy averaged over all positions.

## Generating text

To generate, repeatedly:

1. Run the model on the tokens so far.
2. Take the logits for the **last** position and turn them into probabilities.
3. **Sample** a token from those probabilities (or pick the most likely one).
4. Append it and repeat.

Sampling knobs you'll see everywhere:

- **Temperature:** divide logits by $T$ before softmax. Below 1 is more focused and repetitive; above 1 is more random.
- **Top-k:** only sample from the $k$ most likely tokens.
- **Top-p (nucleus):** only sample from the smallest set of tokens whose probabilities add up to $p$.

## Variants

| Variant | Examples | Used for |
| --- | --- | --- |
| Encoder-only | BERT, RoBERTa | Understanding: classification, embeddings, search. Sees the whole input (no causal mask). |
| Decoder-only | GPT, Llama, Qwen, Mistral, Gemma | Generating text. The dominant LLM design. |
| Encoder-decoder | T5, original transformer, Whisper | Input-to-output tasks like translation and speech-to-text. |

## What changed between GPT-2 and modern LLMs

The core is the same. The refinements:

- **RMSNorm** instead of LayerNorm, applied before each sub-layer (pre-norm).
- **RoPE** rotary position embeddings instead of learned positions, which handle longer contexts better.
- **SwiGLU** activations in the MLP.
- **Grouped-query attention (GQA):** several query heads share each key/value head, shrinking the KV cache.
- **Mixture of experts (MoE)** in some models: many MLPs per layer, and a router picks a few per token. More total parameters, same compute per token.
- And above all: **much more data and much more compute.**

:::tip[Try it]
Read [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar for a visual walk-through. Then copy the `GPT` class above into a notebook, instantiate it with a vocabulary of 65 (characters), and check that a random input of shape `(4, 32)` produces logits of shape `(4, 32, 65)`. Count its parameters.
:::

## Check your understanding

<MultipleChoice
  id="transformer-1"
  question="Inside a transformer block, what does attention do versus the feed-forward MLP?"
  options={[
    "Attention mixes information between tokens; the MLP processes each token on its own",
    "Attention processes each token on its own; the MLP mixes between tokens",
    "They do the same thing",
    "Attention is only used during training",
  ]}
  answer={0}
  explanation="Attention is the only place tokens talk to each other. The MLP holds most of the parameters and works on each token's vector separately."
/>

<MultipleChoice
  id="transformer-2"
  question="For input tokens [t0, t1, t2, t3], what are the training targets?"
  options={["[t0, t1, t2, t3]", "[t1, t2, t3, t4]", "Only t4", "[t3, t2, t1, t0]"]}
  answer={1}
  explanation="Every position predicts the next token, so one sequence gives T training examples at once."
/>

<MultipleChoice
  id="transformer-3"
  question="You want a chatbot's answers to be more predictable and less random. Which sampling change helps?"
  options={["Raise temperature above 1", "Lower temperature below 1", "Increase top-p to 1.0", "Remove the causal mask"]}
  answer={1}
  explanation="Dividing logits by a temperature below 1 sharpens the distribution towards the most likely tokens."
/>

<MultipleChoice
  id="transformer-4"
  question="Which variant would you use to produce sentence embeddings for search?"
  options={["Decoder-only (GPT)", "Encoder-only (BERT-style)", "A CNN", "None, transformers can't do that"]}
  answer={1}
  explanation="Encoder-only models see the whole input without a causal mask, which suits understanding tasks like classification and embeddings."
/>

<OpenQuestion
  id="transformer-5"
  question="Describe, step by step, how a trained GPT generates the sentence that follows a prompt.">

1. **Tokenise** the prompt into token IDs.
2. **Run the model** on all the tokens. It outputs logits at every position, but only the **last** position matters: it's the model's guess for the next token.
3. **Turn those logits into probabilities**, after dividing by the temperature (and optionally cutting to the top-k or top-p tokens).
4. **Sample** one token from that distribution.
5. **Append** it to the sequence.
6. **Repeat** from step 2 until the model produces an end-of-text token or you hit a length limit.
7. **Decode** the token IDs back into text.

In code it's a short loop:

```python
for _ in range(max_new_tokens):
    logits = model(idx)[:, -1, :] / temperature   # last position only
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)
```

Real inference engines also use a KV cache so step 2 doesn't recompute the earlier tokens every time.

</OpenQuestion>

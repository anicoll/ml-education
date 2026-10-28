---
sidebar_position: 4
title: "Capstone: train a tiny GPT"
---

# Capstone: train a tiny GPT

Time to put everything together and train a language model from nothing but random numbers and text. On a consumer GPU this takes between ten minutes and a couple of hours, depending on how big you go.

## The plan

1. Get a dataset and tokenise it.
2. Build the [GPT model](../architectures/transformer.md).
3. Train it with the [full training loop](./the-training-loop.md).
4. Sample text and watch it improve.

## Recommended path: follow Karpathy

Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) builds this exact model from an empty file in about two hours, on Tiny Shakespeare. Follow along and type every line yourself into `notebooks/05-training/tiny-gpt/`. Pause and predict what the next line will be before he writes it.

<YouTube id="kCc8FmEb1nY" title="Let's build GPT: from scratch, in code, spelled out." channel="Andrej Karpathy">

The main video for this capstone. If you haven't done the [makemore videos](https://karpathy.ai/zero-to-hero.html) yet, they build up to this one, but it stands on its own too.

</YouTube>

After that, his [nanoGPT](https://github.com/karpathy/nanoGPT) repo (and its successor [build-nanogpt](https://github.com/karpathy/build-nanogpt)) is a clean, readable training codebase for the same model at larger scale.

## Suggested configurations for an 8 GB GPU

| Config | Params | Data | Context | Approx. time | What to expect |
| --- | --- | --- | --- | --- | --- |
| Char-level Shakespeare | ~10M | 1 MB of text | 256 | 10–20 min | Shakespeare-ish nonsense with correct formatting and plausible words. |
| TinyStories, GPT-2 tokeniser | ~15–30M | 200–500M tokens | 512 | 1–3 hours | Short, mostly coherent children's stories. Genuinely impressive. |
| TinyStories, bigger | ~60–100M | 1–2B tokens | 512 | Overnight or longer | Better grammar and plot consistency. |

On the Mac (PyTorch with `device="mps"`), the same configurations work but expect them to take longer, since `torch.compile` isn't available and per-step compute is usually lower. Memory won't be a constraint at these sizes. Running the same config on both machines and comparing tokens per second is a worthwhile experiment in itself.

Times are rough; measure your own throughput. Use bf16 autocast, `torch.compile` if you can, and batch size plus gradient accumulation to fill memory without going out of it.

## Sampling

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]              # crop to context length
        logits = model(idx_cond)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

Sample every few hundred steps and save the outputs. Reading them in order is the most satisfying part of the whole course: you'll see the model go from random characters, to word-shaped blobs, to real words, to grammar, to something like style.

## Questions to investigate

Once it works, experiment. Change one thing at a time and record results in `notes/`:

1. What happens to validation loss if you double the number of layers? Double the width?
2. How does loss scale with training tokens? Plot validation loss against tokens seen on a log-log scale. Is it roughly a straight line? (That's a **scaling law**.)
3. Remove the residual connections. Remove the layer norms. What breaks, and how badly?
4. Train two models of different sizes for the same amount of *compute*. Which wins?
5. Compare a character-level tokeniser with GPT-2's BPE on the same text.

## Then: GPT-2 itself

Karpathy's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) video reproduces the real GPT-2 small. A full reproduction on 10B tokens is out of reach for a single consumer GPU in reasonable time (it's a few hours on 8 high-end data-centre GPUs), but you can follow the code, train on a slice of the data, and rent a cloud GPU for a few dollars if you want to see it through. See [When to rent a GPU](../hardware/cloud-gpus.md).

<YouTube id="l8pRSuU81PU" title="Let's reproduce GPT-2 (124M)" channel="Andrej Karpathy" />

:::info[Key idea]
Everything in the big LLMs is here in miniature: tokeniser, embeddings, attention, MLPs, residuals, next-token loss, AdamW, a cosine schedule. The difference between your model and a frontier one is mostly scale, data quality and the post-training steps covered in Part 6.
:::

## Check your understanding

<MultipleChoice
  id="tinygpt-1"
  question="Early in training, your tiny GPT's samples are random characters. Later they become word-shaped blobs, then real words. What does this progression show?"
  options={[
    "The model is overfitting",
    "The model is learning structure in stages: character statistics first, then words, then grammar",
    "The temperature is changing",
    "The tokeniser is being retrained",
  ]}
  answer={1}
  explanation="Easy, frequent patterns reduce the loss most, so they're learned first. Longer-range structure comes later."
/>

<MultipleChoice
  id="tinygpt-2"
  question="In the generate function, what does `idx[:, -block_size:]` do?"
  options={["Reverses the sequence", "Keeps only the most recent block_size tokens, because the model can't handle a longer context", "Removes the prompt", "Picks a random window"]}
  answer={1}
  explanation="The position embeddings only go up to block_size, so the input is cropped to the last block_size tokens."
/>

<OpenQuestion
  id="tinygpt-3"
  question="Pick one of the 'Questions to investigate' and write down your hypothesis before running it: what do you expect to happen and why?">

A worked example for question 3 (removing residual connections):

**Hypothesis:** with 6 layers, removing the residual connections will make training much slower and the final loss noticeably worse, and with 12+ layers it may barely train at all. The reason: without the `x + ...` shortcut, gradients have to pass through every layer's transformations on the way back, and they shrink (or grow) at each step. Residual connections give gradients a direct path to the early layers.

**How to test it:** change `x = x + self.attn(...)` to `x = self.attn(...)` (and the same for the MLP line), train both versions with identical settings and seeds, and plot validation loss against steps on one chart.

Writing the hypothesis first matters: it forces you to make a prediction, and being wrong is where most of the learning happens.

</OpenQuestion>

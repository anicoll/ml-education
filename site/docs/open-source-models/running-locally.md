---
sidebar_position: 3
title: Running models locally
---

# Running models locally

Before fine-tuning anything, get a feel for what open models can do by running them on your own machine. This is the quickest win in the course: you can be chatting with a capable model on your own GPU or Mac in about ten minutes.

## The options

| Tool | Good for | Notes |
| --- | --- | --- |
| **Ollama** | Easiest start | One-line install, one command to download and run a model. Runs a local API. |
| **LM Studio** | Point-and-click | Desktop app with a model browser and chat UI. |
| **llama.cpp** | Control and efficiency | The engine underneath Ollama and LM Studio. Runs GGUF quantised models on GPU, CPU or both. |
| **Hugging Face `transformers`** | Experimenting in Python | Most flexible, not the fastest. |
| **MLX (`mlx-lm`)** | Speed on a Mac | Apple's framework; usually the fastest way to run models on Apple Silicon. |
| **vLLM / SGLang** | Serving many requests fast | Best throughput; more setup; Linux or WSL2 with an NVIDIA GPU. |

## Quick start with Ollama

Install from [ollama.com](https://ollama.com), then:

```bash
ollama run llama3.1:8b        # downloads a 4-bit quantised 8B model (about 5 GB) and opens a chat
ollama run qwen2.5:7b
ollama list                   # what you've downloaded
ollama ps                     # what's loaded, and whether it's on the GPU
```

Ollama also exposes an OpenAI-compatible HTTP API at `http://localhost:11434`, so you can call your local model from code:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(model="llama3.1:8b", messages=[{"role": "user", "content": "Hi!"}])
print(resp.choices[0].message.content)
```

## GGUF and quantisation levels

llama.cpp and Ollama use **GGUF** files: model weights quantised to fewer bits. You'll see names like `Q4_K_M`, `Q5_K_M`, `Q8_0`:

- The number is roughly the bits per weight.
- `K_M` / `K_S` are variants of the "k-quant" scheme (medium and small).
- **Q4_K_M is the usual sweet spot**: about a quarter of the bf16 size, with small quality loss.

An 8B model at Q4_K_M is about 4.9 GB, which leaves room on an 8 GB card for the KV cache and a few thousand tokens of context.

## What happens when a model doesn't fit

llama.cpp can split layers between GPU and CPU (`--n-gpu-layers` / `-ngl`). Layers that don't fit on the GPU run on the CPU from system RAM. It works, but each layer on the CPU slows things down a lot. Rough expectations on an 8 GB card:

- 7–8B at Q4, fully on GPU: fast, typically 40+ tokens per second.
- 13–14B at Q4: just over 8 GB, so a few layers spill to CPU. Usable but noticeably slower.
- 30B+: mostly on CPU. Slow, but it runs if you have 32 GB+ of system RAM.

On a Mac there's no split to worry about: the GPU uses unified memory, so with 32 GB or more, 27–32B models at Q4 (about 17–20 GB) run entirely on the GPU at a usable speed. For the fastest generation on a Mac, try MLX models with `mlx_lm.chat` or in LM Studio. See [Apple Silicon](../hardware/apple-silicon.md).

## What to notice

As you play, pay attention to:

- **Speed** (tokens per second) and **VRAM** (`nvidia-smi`), and how both change with model size and context length.
- **Quality differences** between a 1.5B, a 3B and an 8B model on the same prompts.
- **Failure modes:** confident wrong facts (hallucination), losing track in long conversations, weak arithmetic.

These observations will guide what you choose to fine-tune and what you can realistically expect from it.

## Building a test prompt set

A good personal test set mixes different skills, includes some prompts you know the answer to, and includes some that are likely to trip a small model up. Here's an example set of ten you can use as-is or adapt:

| # | Skill | Prompt | What a good answer looks like |
| --- | --- | --- | --- |
| 1 | Factual | What's the difference between a process and a thread? | Correct, concise, mentions shared memory. |
| 2 | Factual (likely to hallucinate) | Who wrote the paper "Attention Is All You Need", and in what year? | Vaswani et al., 2017. Watch for made-up author names. |
| 3 | Reasoning | A bat and a ball cost \$1.10 in total. The bat costs \$1.00 more than the ball. How much is the ball? | 5 cents, not 10. |
| 4 | Multi-step maths | I run 5 km at 5:30 per km, then 3 km at 6:10 per km. What's my total time and average pace? | 46:00 total, 5:45 per km. |
| 5 | Code | Write a Python function that returns the n most common words in a text file, ignoring case and punctuation. | Runs, uses `collections.Counter`, handles punctuation. |
| 6 | Code review | Paste a 20-line function with a subtle bug and ask "What's wrong with this code?" | Finds the bug without inventing others. |
| 7 | Writing | Rewrite this paragraph to be half as long without losing information: (paste a paragraph of your own) | Shorter, nothing important lost. |
| 8 | Structured output | Extract the name, date and amount from this text as JSON: "Paid Sarah 45 dollars on 3 March for concert tickets." | Valid JSON with the right three fields. |
| 9 | Explaining | Explain gradient descent to someone who knows programming but no calculus. | Clear analogy, no errors. |
| 10 | Knowing its limits | What was the closing price of Apple stock yesterday? | Says it can't know this, rather than inventing a number. |

Score each answer quickly (for example 0 = wrong, 1 = partly right, 2 = good) so you can compare models and, later, your fine-tunes with a number as well as a gut feeling.

:::tip[Try it]
Install Ollama and run a 1–3B model and a 7–8B model (on the Mac, add a 27–32B model too). Run the ten prompts above, or your own version of them, on each model, and save the prompts, answers and scores in `notes/model-comparison.md`. You'll reuse these prompts to evaluate your fine-tunes.
:::

## Check your understanding

<MultipleChoice
  id="local-1"
  question="What does `Q4_K_M` in a GGUF file name tell you?"
  options={[
    "The model has 4 billion parameters",
    "It's quantised to roughly 4 bits per weight using the medium k-quant variant",
    "It needs 4 GPUs",
    "It's the fourth version of the model",
  ]}
  answer={1}
  explanation="Q4_K_M is the usual sweet spot: about a quarter of the bf16 size with small quality loss."
/>

<MultipleChoice
  id="local-2"
  question="A 14B model at Q4 is slightly too big for an 8 GB card, so llama.cpp puts a few layers on the CPU. What happens?"
  options={["It refuses to run", "It runs, but noticeably slower, because layers on the CPU are much slower", "It runs at the same speed", "The answers become wrong"]}
  answer={1}
  explanation="Splitting works, but every layer that runs from system RAM on the CPU slows generation down."
/>

<MultipleChoice
  id="local-3"
  question="Ollama exposes an OpenAI-compatible API on localhost. What does that let you do?"
  options={[
    "Use OpenAI's models for free",
    "Call your local model from code using the same client library you'd use for a hosted API",
    "Fine-tune models",
    "Share your model on the internet",
  ]}
  answer={1}
  explanation="Point the OpenAI client at http://localhost:11434/v1 and your code talks to the local model instead."
/>

<OpenQuestion
  id="local-4"
  question="Why does it matter to build and save a fixed test prompt set before you start fine-tuning?">

Because without it you can't tell whether a fine-tune actually helped. If you test with whatever prompts come to mind each time, you'll remember the impressive answers and forget the bad ones, and you'll unknowingly pick prompts the new model is good at.

A fixed set, written before training, gives you:

- **A baseline:** the original model's scores on exactly these prompts.
- **A fair comparison:** every model and every fine-tune answers the same questions under the same settings.
- **Coverage:** prompts across different skills, including ones that test what you *don't* want to break (general knowledge, saying "I don't know").
- **A number**, as well as a gut feeling, to compare runs.

</OpenQuestion>

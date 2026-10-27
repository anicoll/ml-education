---
sidebar_position: 7
title: "Walkthrough: your first QLoRA fine-tune"
---

# Walkthrough: your first QLoRA fine-tune

This page takes you through a complete QLoRA fine-tune of a small instruct model on an 8 GB NVIDIA GPU. Budget an evening for it, most of it spent on data.

:::info[On a Mac?]
This walkthrough uses bitsandbytes for 4-bit quantisation, which needs an NVIDIA GPU. On Apple Silicon, follow the same steps for data (2), baseline (3) and evaluation (5), but do the training with MLX instead: see [Fine-tuning with MLX](../hardware/apple-silicon.md#fine-tuning-with-mlx-lora-and-qlora). With 32 GB or more of unified memory you can also skip quantisation entirely and run plain LoRA on a bf16 base up to about 8B, by deleting `quantization_config` below and using `device_map="mps"`.
:::

:::note[Library versions]
The Hugging Face libraries change quickly, and argument names occasionally move between versions. If something here errors, check the current [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer) and [PEFT docs](https://huggingface.co/docs/peft). The concepts don't change.
:::

## 0. Environment

Follow [Setting up your machine](../hardware/overview.md) first. Then:

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes
```

## 1. Choose a base model

Start with something small so iterations are fast: `Qwen/Qwen2.5-1.5B-Instruct` or `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Once the whole pipeline works, move up to a 3B model, and later try 7–8B.

## 2. Prepare data

Create `data/train.jsonl` and `data/test.jsonl`, one conversation per line, in the `messages` format from [Fine-tuning: when and how](./fine-tuning-overview.md#data-the-part-that-matters-most). Aim for 300–1,000 training examples and 30–100 test prompts.

If you don't have data yet, a public dataset is fine for a first run, for example a slice of `HuggingFaceH4/no_robots` or `HuggingFaceTB/smoltalk`.

Check your data with the model's chat template before training:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
ds = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/test.jsonl"})

print(tok.apply_chat_template(ds["train"][0]["messages"], tokenize=False))
lengths = [len(tok.apply_chat_template(ex["messages"])) for ex in ds["train"]]
print("max tokens:", max(lengths), "mean:", sum(lengths) / len(lengths))
```

Knowing your token lengths tells you what `max_length` to set; anything longer is truncated.

## 3. Baseline

Run the **untuned** model on your test prompts and save the outputs. Without a baseline you can't tell whether fine-tuning helped.

## 4. Train

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="cuda")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="runs/qwen1.5b-qlora",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch size 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    optim="paged_adamw_8bit",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    peft_config=peft_config,
    processing_class=tok,
)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model("runs/qwen1.5b-qlora/final")
```

While it runs:

- Watch `nvidia-smi`. If you run out of memory, lower `per_device_train_batch_size` to 1 (and double `gradient_accumulation_steps`), then lower `max_length`.
- Watch training and eval loss in TensorBoard (`tensorboard --logdir runs`). Eval loss rising while training loss falls means overfitting: use fewer epochs.
- Training loss for SFT on good data typically ends somewhere around 0.5–1.5. Near zero means the model is memorising.

## 5. Evaluate

Load the base model plus your adapter and run the same test prompts as the baseline:

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, "runs/qwen1.5b-qlora/final")
```

Compare side by side with the baseline. Ask yourself:

- Did it learn the behaviour you wanted?
- Did it get worse at anything general? Try some of your ten prompts from [Running models locally](./running-locally.md).
- Where does it still fail? Usually the answer is more or better data covering those cases.

## 6. Merge and run in Ollama (optional)

```python
merged = model.merge_and_unload()
merged.save_pretrained("runs/qwen1.5b-merged")
tok.save_pretrained("runs/qwen1.5b-merged")
```

Then convert to GGUF with llama.cpp's `convert_hf_to_gguf.py`, quantise to Q4_K_M, and create an Ollama model from it with a `Modelfile`. Now your fine-tuned model runs alongside the others in Ollama.

## 7. Iterate

The loop that matters is: **look at failures → improve data → retrain → compare**. Changing hyperparameters helps a little; changing data helps a lot. Keep a table in `notes/` of each run: data version, settings, and what got better or worse.

## Going further

- Move up to a 3B model, then 7–8B with QLoRA (and probably Unsloth).
- Try preference tuning with `trl`'s `DPOTrainer` on pairs of good and bad answers.
- Compare LoRA ranks 8, 16 and 64 on the same data.
- Distil: generate high-quality answers for your prompts with a large model, and fine-tune your small model on them.

## Check your understanding

<MultipleChoice
  id="qlora-walk-1"
  question="Why run the untuned model on your test prompts before training?"
  options={["To warm up the GPU", "To get a baseline, so you can tell whether fine-tuning actually helped", "Because SFTTrainer requires it", "To download the model weights"]}
  answer={1}
  explanation="Without a baseline you can only guess whether the fine-tuned answers are better."
/>

<MultipleChoice
  id="qlora-walk-2"
  question="With per_device_train_batch_size=2 and gradient_accumulation_steps=8, what's the effective batch size?"
  options={["2", "8", "10", "16"]}
  answer={3}
  explanation="2 examples per step × 8 accumulated steps = 16 examples per optimiser update."
/>

<MultipleChoice
  id="qlora-walk-3"
  question="Training loss drops close to zero and eval loss starts rising. What should you change first?"
  options={["Train for more epochs", "Train for fewer epochs (or add more data): the model is memorising", "Increase the learning rate", "Increase max_length"]}
  answer={1}
  explanation="SFT loss on good data typically ends around 0.5–1.5. Near zero with rising eval loss is overfitting."
/>

<MultipleChoice
  id="qlora-walk-4"
  question="Your fine-tuned model still fails on a type of input. What usually helps most?"
  options={["Changing the LoRA rank", "Adding better training examples that cover those cases", "Switching optimiser", "Training on the test set"]}
  answer={1}
  explanation="Changing hyperparameters helps a little; changing data helps a lot. Look at failures, improve the data, retrain, compare."
/>

<OpenQuestion
  id="qlora-walk-5"
  question="You run the training script and get 'CUDA out of memory' on the first step. Which settings in SFTConfig would you change, in what order, and what does each trade away?">

1. **`per_device_train_batch_size` from 2 to 1**, and `gradient_accumulation_steps` from 8 to 16. The effective batch stays 16; you trade some speed.
2. **`max_length` from 1024 to 512** (or just above your longest example, from the token-length check in step 2). Biggest single saving; you lose anything past the cut-off, so check how many examples get truncated.
3. **Make sure `gradient_checkpointing=True` and `bf16=True`** are on (they are in the example). Checkpointing trades about 30% speed for much less activation memory.
4. **Lower the LoRA rank** (`r=8`) or target fewer modules. Small memory saving; slightly less adapter capacity.
5. **Use a smaller base model**, or switch to Unsloth, if it still doesn't fit.

After each change, watch `nvidia-smi` for a few steps so you know how much headroom you have.

</OpenQuestion>

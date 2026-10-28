---
sidebar_position: 5
title: PyTorch essentials
---

# PyTorch essentials

PyTorch is the dominant framework for deep learning research and for the open-source LLM ecosystem. This page covers the parts you'll use every day.

## Tensors

A tensor is an n-dimensional array, like a NumPy array, that can live on a GPU and track gradients.

```python
import torch

x = torch.randn(32, 784)          # batch of 32 flattened images
print(x.shape, x.dtype, x.device) # torch.Size([32, 784]) torch.float32 cpu

if torch.cuda.is_available():
    device = "cuda"               # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"                # Apple Silicon GPU
else:
    device = "cpu"
x = x.to(device)                  # move to the GPU
```

The rest of this course often writes `"cuda"` directly. On the Mac, use `device` (or `"mps"`) instead. See [Choosing your machine](../hardware/overview.md).

Operations between tensors must be on the same device. "Expected all tensors to be on the same device" is an error you'll see often.

### Shapes and broadcasting

```python
a = torch.randn(32, 10)
b = torch.randn(10)
c = a + b            # b is broadcast across the 32 rows -> (32, 10)

x.view(32, 28, 28)   # reshape (must keep the same number of elements)
x.unsqueeze(0)       # add a dimension: (1, 32, 784)
x.transpose(0, 1)    # swap dimensions: (784, 32)
```

### Data types

| dtype | Bytes | Use |
| --- | --- | --- |
| `float32` | 4 | Default. Safe, but uses the most memory. |
| `float16` | 2 | Mixed-precision training. Can overflow. |
| `bfloat16` | 2 | Same range as float32, less precision. Preferred for LLMs. Supported on NVIDIA RTX 30-series (Ampere) and newer, and on recent macOS/PyTorch with MPS. |
| `int8` / 4-bit | 1 / 0.5 | Quantised weights for inference and QLoRA. |

## Modules

Models are built from `nn.Module` subclasses. Define layers in `__init__` and the computation in `forward`.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)   # raw logits; the loss applies softmax

model = MLP().to(device)
print(sum(p.numel() for p in model.parameters()))  # parameter count
```

## Datasets and DataLoaders

A `Dataset` returns one example by index; a `DataLoader` batches and shuffles them.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=512)
```

## A complete training script

This is the [core loop](../foundations/the-core-loop.md) with real components. It should reach about 98% test accuracy in a few epochs on a GPU (and not much longer on a CPU).

```python
import torch.nn.functional as F

model = MLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device).flatten(1), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device).flatten(1), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {correct / len(test_ds):.4f}")
```

Things to notice:

- `model.train()` and `model.eval()` switch dropout and batchnorm behaviour.
- `torch.no_grad()` skips gradient tracking during evaluation, saving memory and time.
- `F.cross_entropy` takes raw logits and applies softmax internally (more numerically stable).
- `optimizer.zero_grad()` is needed because PyTorch *adds* new gradients to old ones by default.

## Saving and loading

```python
torch.save(model.state_dict(), "mlp.pt")
model.load_state_dict(torch.load("mlp.pt"))
```

:::tip[Try it]
Put this script in `notebooks/03-neural-networks/mnist_mlp.py` and run it on your GPU. Then:

1. Time an epoch on CPU vs GPU.
2. Go back to your `make_moons` notebook from Part 2 and solve it with a small MLP. Plot the curved decision boundary.
3. Replace the MLP with a single `nn.Linear(784, 10)` (that's softmax regression). How much accuracy do the hidden layers buy you?
:::

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="PaCmpygFfXo" title="The spelled-out intro to language modeling: building makemore" channel="Andrej Karpathy">

Part 2 of Zero to Hero, and a great hands-on PyTorch tutorial in disguise: tensors, indexing, broadcasting and a first `loss.backward()` training loop, all on a tiny name-generating model.

</YouTube>

## Check your understanding

<MultipleChoice
  id="pytorch-1"
  question="You get the error 'Expected all tensors to be on the same device'. What's the usual cause?"
  options={[
    "The model is too big",
    "The model is on the GPU but the input batch is still on the CPU (or the other way round)",
    "The batch size is odd",
    "You forgot to call model.eval()",
  ]}
  answer={1}
  explanation="Move both the model and every batch to the same device: model.to(device) once, and x.to(device), y.to(device) inside the loop."
/>

<MultipleChoice
  id="pytorch-2"
  question="Why is `optimizer.zero_grad()` needed in every step?"
  options={[
    "It resets the weights",
    "PyTorch adds new gradients to the old ones by default, so they must be cleared before each backward pass",
    "It moves the model to the GPU",
    "It saves a checkpoint",
  ]}
  answer={1}
  explanation="Without it, gradients from every previous step pile up. (Gradient accumulation uses this behaviour on purpose.)"
/>

<MultipleChoice
  id="pytorch-3"
  question="What does wrapping evaluation in `torch.no_grad()` do?"
  options={[
    "Turns off dropout",
    "Skips gradient tracking, saving memory and time, since you won't call backward during evaluation",
    "Freezes the weights permanently",
    "Makes the model more accurate",
  ]}
  answer={1}
  explanation="Dropout is controlled by model.eval(); no_grad() just stops PyTorch recording operations for backprop."
/>

<MultipleChoice
  id="pytorch-4"
  question="`F.cross_entropy(logits, y)` expects which input?"
  options={["Probabilities that already went through softmax", "Raw logits; it applies softmax internally", "One-hot labels only", "Log probabilities from sigmoid"]}
  answer={1}
  explanation="Pass raw logits. Applying softmax yourself first is a common bug that quietly hurts training."
/>

<OpenQuestion
  id="pytorch-5"
  question="If you don't know Python yet, read this line slowly: `correct += (model(x).argmax(1) == y).sum().item()`. Explain what each part does.">

Read it from the inside out:

- `model(x)`: run the batch through the model. The result has shape `(batch_size, 10)`: ten scores (logits) per image, one per digit.
- `.argmax(1)`: for each image, take the index of the biggest score along dimension 1 (the 10 classes). That's the predicted digit. Shape `(batch_size,)`.
- `== y`: compare each prediction with the true label, giving a tensor of `True`/`False`.
- `.sum()`: count the `True` values (True counts as 1). That's how many predictions in this batch were right.
- `.item()`: turn the one-number tensor into a plain Python number.
- `correct += ...`: add it to the running total. After the loop, `correct / len(test_ds)` is the accuracy.

</OpenQuestion>

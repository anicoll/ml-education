---
sidebar_position: 4
title: The maths you actually need
---

# The maths you actually need

You do not need a maths degree to do ML. You need a working intuition for four things. This page gives you that intuition; dig deeper later when a specific topic demands it.

## 1. Vectors and matrices (linear algebra)

A **vector** is a list of numbers. A data point with 3 features is a vector:

$$
x = [\text{size}, \text{rooms}, \text{age}] = [120, 3, 15]
$$

A **matrix** is a grid of numbers. A dataset of $N$ examples with $d$ features is an $N \times d$ matrix, one row per example.

The single most important operation is the **dot product**: multiply matching elements and sum.

$$
w \cdot x = w_1 x_1 + w_2 x_2 + \dots + w_d x_d
$$

A linear model is just a dot product plus a bias: $\hat{y} = w \cdot x + b$. A neural network layer does many dot products at once, which is a **matrix multiplication**:

$$
h = W x + b
$$

If $W$ has shape $(\text{out}, \text{in})$ and $x$ has shape $(\text{in})$, then $h$ has shape $(\text{out})$. Most bugs in deep learning code are shape mismatches, so get into the habit of writing shapes next to every line.

```python
import torch
W = torch.randn(4, 3)   # 3 inputs -> 4 outputs
x = torch.randn(3)
h = W @ x               # shape: (4,)
```

**Why GPUs?** A matrix multiply is thousands of independent multiply-adds. GPUs are built to do thousands of those in parallel. That's the whole reason your 3070 is useful for ML.

## 2. Derivatives and gradients (calculus)

A **derivative** tells you how fast a function's output changes as you nudge its input.

$$
f(x) = x^2 \quad\Rightarrow\quad f'(x) = 2x
$$

At $x = 3$, the slope is 6: nudge $x$ up by 0.01 and $f$ goes up by about 0.06.

With many inputs, you take a **partial derivative** for each one, holding the others fixed. Collect them into a vector and you have the **gradient** $\nabla f$, which points in the direction of steepest increase.

The **chain rule** is the one rule you really need. If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg}\cdot\frac{dg}{dx}
$$

A neural network is a long chain of functions, so the chain rule lets you compute the gradient of the loss with respect to every parameter by working backwards through the chain. That algorithm is called **backpropagation**, covered in [Part 3](../neural-networks/backpropagation.md).

## 3. Probability

Many models output **probabilities**, not hard answers. A classifier says "92% cat, 8% dog". A language model outputs a probability for every possible next word.

Things to be comfortable with:

- A probability distribution assigns a number between 0 and 1 to each outcome, and they sum to 1.
- **Softmax** turns any list of numbers (called *logits*) into a probability distribution:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

- **Log probabilities**: multiplying many small probabilities underflows, so we add their logs instead. That's why you'll see "log" everywhere in loss functions.

## 4. Cross-entropy: the loss for classification

If the model assigns probability $p$ to the correct class, the loss is:

$$
L = -\log p
$$

- Model is confident and right ($p = 0.99$): loss is about 0.01.
- Model is unsure ($p = 0.5$): loss is about 0.69.
- Model is confident and wrong ($p = 0.01$): loss is about 4.6.

This is **cross-entropy loss**, and it's the loss used to train every LLM: at each position, $-\log$ of the probability the model gave to the actual next token.

A handy fact: a model guessing uniformly over $K$ classes has loss $\log K$. For a vocabulary of 50,000 tokens that's about 10.8. If your language model's loss starts near there and goes down, training is working.

## Do I need more?

For this course, no. If you want to build stronger foundations later:

- **3Blue1Brown's "Essence of Linear Algebra" and "Essence of Calculus"** video series are the best visual introductions available.
- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) is free online and covers everything in depth.

:::tip Try it
In `notebooks/01-foundations/maths.ipynb`:

1. Implement `softmax` in NumPy. Check that the outputs sum to 1.
2. What happens to softmax if you multiply every input by 10? By 0.1? (This is "temperature", and it's how you control randomness when sampling from an LLM.)
3. Compute the derivative of $f(x) = x^3$ at $x = 2$ numerically using $(f(x+h) - f(x))/h$ with $h = 0.0001$, and compare to the exact answer.
:::

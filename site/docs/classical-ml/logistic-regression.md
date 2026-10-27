---
sidebar_position: 2
title: Logistic regression and classification
---

# Logistic regression and classification

Linear regression predicts a number. For classification we want a **probability**: "how likely is this email to be spam?"

## Squashing a line into a probability

Take the same linear score $z = w \cdot x + b$ (which can be any number, called a **logit**), and squash it into the range 0 to 1 with the **sigmoid** function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- Large positive $z$ gives a value near 1.
- Large negative $z$ gives a value near 0.
- $z = 0$ gives exactly 0.5.

So $\hat{p} = \sigma(w \cdot x + b)$ is the predicted probability of the positive class. Despite its name, logistic regression is a **classification** model.

## The loss: binary cross-entropy

MSE works badly for probabilities. Instead we use **cross-entropy**, which heavily punishes confident wrong answers:

$$
L = -\frac{1}{N}\sum_i \Big[ y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i) \Big]
$$

When the label $y_i = 1$ only the first term matters ($-\log \hat{p}$); when $y_i = 0$ only the second ($-\log(1-\hat{p})$). This is the same idea as in [the maths page](../foundations/maths-you-need.md#4-cross-entropy-the-loss-for-classification).

A lovely fact: the gradient comes out in exactly the same form as linear regression.

$$
\nabla_w L = \frac{1}{N} X^\top (\hat{p} - y)
$$

## From scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logreg(X, y, lr=0.1, steps=1000):
    N, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        w -= lr * X.T @ (p - y) / N
        b -= lr * (p - y).mean()
    return w, b

def predict(X, w, b, threshold=0.5):
    return (sigmoid(X @ w + b) >= threshold).astype(int)
```

## Decision boundaries

The model predicts class 1 when $w \cdot x + b > 0$. That inequality defines a straight line (or plane) splitting the input space: the **decision boundary**. Logistic regression can only separate classes with a straight boundary. Data shaped like two interlocking moons can't be separated that way, which is another motivation for neural networks.

## More than two classes: softmax regression

For $K$ classes, compute $K$ scores (one weight vector per class), then turn them into a probability distribution with **softmax**:

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

The loss is $-\log \hat{p}_{\text{correct class}}$. This exact setup (a linear layer, softmax, cross-entropy) sits at the end of almost every classification network, and at the end of every LLM, where the "classes" are the tokens in its vocabulary.

:::info[Key idea]
Linear score → sigmoid or softmax → probability → cross-entropy loss. Memorise this pipeline; it's the output end of nearly every model you'll train.
:::

:::tip[Try it]
Use `sklearn.datasets.make_moons` to generate two-class data. Train your from-scratch logistic regression, plot the decision boundary, and see where it fails. Keep this notebook: you'll come back and solve the same data with a neural network in Part 3.
:::

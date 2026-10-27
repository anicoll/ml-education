---
sidebar_position: 1
title: Linear regression
---

# Linear regression

Linear regression is the "hello world" of machine learning. It's simple, but it contains every idea you'll need for deep learning: parameters, a loss, gradients, and vectorised maths.

## The model

With $d$ input features, the model is a weighted sum plus a bias:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b = w \cdot x + b
$$

Each weight $w_j$ says how much feature $j$ pushes the prediction up or down. That makes linear models **interpretable**: you can read the weights and understand what the model learned.

## Vectorising

Writing loops over examples is slow. Stack all $N$ examples into a matrix $X$ of shape $(N, d)$ and compute every prediction at once:

$$
\hat{y} = X w + b
$$

This one line is the same operation a neural network layer does. Getting comfortable thinking in whole matrices, not individual numbers, is the main skill to build here.

## Training it

The loss is mean squared error, and the gradients in vector form are:

$$
\nabla_w L = \frac{2}{N} X^\top (\hat{y} - y) \qquad \frac{\partial L}{\partial b} = \frac{2}{N}\sum_i (\hat{y}_i - y_i)
$$

```python
import numpy as np

rng = np.random.default_rng(0)
N, d = 500, 3
X = rng.normal(size=(N, d))
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + 4.0 + rng.normal(scale=0.1, size=N)

w = np.zeros(d)
b = 0.0
lr = 0.1

for step in range(300):
    pred = X @ w + b                  # (N,)
    err = pred - y                    # (N,)
    loss = (err ** 2).mean()
    grad_w = 2 / N * X.T @ err        # (d,)
    grad_b = 2 * err.mean()
    w -= lr * grad_w
    b -= lr * grad_b

print(w, b)   # close to [2, -1, 0.5] and 4
```

:::note
Linear regression also has a closed-form solution (the "normal equation", $w = (X^\top X)^{-1} X^\top y$) that skips gradient descent entirely. We use gradient descent anyway because it's the method that scales to neural networks, where no closed form exists.
:::

## With scikit-learn

In practice you'd use a library:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
```

## Limits of linear models

A linear model can only draw straight lines (or flat planes in higher dimensions). If the real relationship is curved, it can't capture it. You have two options:

1. **Engineer features**: add $x^2$, $x_1 x_2$, $\log x$ as new columns. The model is still linear in its parameters, but can now fit curves.
2. **Learn the features**: let a neural network discover useful transformations of the input itself. This is the leap that deep learning makes.

:::tip Try it
In `notebooks/02-classical-ml/linear-regression.ipynb`, run the NumPy version above on the California housing dataset (standardise first). Compare your weights to scikit-learn's. Then add squared features and see whether validation error improves.
:::

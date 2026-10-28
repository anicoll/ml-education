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

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="yIYKR4sgzI8" title="StatQuest: Logistic Regression" channel="StatQuest with Josh Starmer">

The S-shaped curve, how it turns a score into a probability, and how it's used for classification.

</YouTube>

## Check your understanding

<MultipleChoice
  id="logreg-1"
  question="What is sigmoid(0)?"
  options={["0", "0.5", "1", "Undefined"]}
  answer={1}
  explanation="σ(0) = 1 / (1 + e⁰) = 1/2. A logit of 0 means the model is exactly undecided."
/>

<MultipleChoice
  id="logreg-2"
  question="Despite the name, what kind of problem does logistic regression solve?"
  options={["Regression: predicting a number", "Classification: predicting the probability of a class", "Clustering", "Reinforcement learning"]}
  answer={1}
  explanation="It outputs a probability between 0 and 1 for the positive class, and you threshold it (usually at 0.5) to pick a class."
/>

<MultipleChoice
  id="logreg-3"
  question="Why can't plain logistic regression separate the two interlocking 'moons' in make_moons?"
  options={[
    "The dataset is too small",
    "Its decision boundary is always a straight line (or flat plane), and the moons need a curved boundary",
    "Sigmoid can't output values close to 1",
    "Cross-entropy doesn't work with two classes",
  ]}
  answer={1}
  explanation="The model predicts class 1 when w · x + b > 0, which is a straight boundary. Curved boundaries need engineered features or a neural network."
/>

<MultipleChoice
  id="logreg-4"
  question="What sits at the output end of nearly every classifier, including every LLM?"
  options={[
    "A decision tree",
    "A linear layer producing logits, then softmax, trained with cross-entropy",
    "Mean squared error",
    "k-fold cross-validation",
  ]}
  answer={1}
  explanation="An LLM is a classifier over its vocabulary: one logit per token, softmax into probabilities, and −log p of the real next token as the loss."
/>

<OpenQuestion
  id="logreg-5"
  question="The from-scratch `predict` function uses `threshold=0.5`. Describe a situation where you'd want a lower threshold, and what you give up by lowering it.">

Lower the threshold when **missing a positive is expensive**. For example, a model that flags possibly fraudulent transactions for a human to review: with a threshold of 0.2, anything with at least a 20% predicted chance of fraud gets flagged. You catch more of the real fraud (higher **recall**).

The price is more false alarms: more legitimate transactions get flagged (lower **precision**), which costs reviewer time or annoys customers. Choosing the threshold is choosing a point on that trade-off, based on the real-world cost of each kind of mistake. The [Evaluating models](./evaluation.md) page covers this in detail.

</OpenQuestion>

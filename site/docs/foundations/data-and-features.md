---
sidebar_position: 5
title: Data and features
---

# Data and features

Models are only as good as their data. Practitioners joke that ML is 80% data work and 20% modelling, and it's only partly a joke.

## Everything becomes numbers

Models operate on numbers, so every input has to be converted into a vector of numbers.

| Data type | How it becomes numbers |
| --- | --- |
| Tabular (spreadsheets) | Each column is a feature. Categories become one-hot vectors. |
| Images | A grid of pixel intensities: height × width × 3 colour channels. |
| Audio | A waveform (amplitude over time) or a spectrogram. |
| Text | Split into **tokens**, each token mapped to an ID, each ID mapped to a learned vector (an **embedding**). |

**One-hot encoding** turns a category into a vector with a single 1:

```text
colour = "green"  with categories [red, green, blue]  →  [0, 1, 0]
```

## Train, validation and test splits

This is the most important habit in ML. Split your data **before** you do anything else:

- **Training set** (e.g. 80%): the model learns from this.
- **Validation set** (e.g. 10%): you use this to make choices, like learning rate or model size, and to decide when to stop training.
- **Test set** (e.g. 10%): touched **once**, at the very end, to report how good the model really is.

Why three? Every time you look at a score and change something, you leak a little information about that data into your decisions. If you tune on the test set, your test score becomes optimistic, and you'll be disappointed in the real world.

:::warning
**Data leakage** is when information from the validation or test set sneaks into training. Classic examples: normalising using statistics computed on the whole dataset, or the same customer appearing in both train and test. Leakage makes models look great on paper and fail in reality.
:::

## Normalisation

Features on wildly different scales (house size in square metres vs number of rooms) make gradient descent struggle, because one parameter needs huge steps and another needs tiny ones. The standard fix is to **standardise** each feature to mean 0 and standard deviation 1:

$$
x' = \frac{x - \mu}{\sigma}
$$

Compute $\mu$ and $\sigma$ on the **training set only**, then apply the same values to validation and test.

## Data quality beats data quantity (up to a point)

- **Label noise:** wrong labels teach the model wrong things. A few percent is tolerable; a lot is not.
- **Distribution shift:** if your training data doesn't look like the real data the model will see, it will do worse than your test score suggests.
- **Class imbalance:** if 99% of emails are not spam, a model that always says "not spam" gets 99% accuracy and is useless. See [Evaluation](../classical-ml/evaluation.md).

For fine-tuning LLMs this matters even more: a thousand carefully written examples usually beat a hundred thousand sloppy ones.

:::tip Try it
Download the classic [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) or [California housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) dataset via scikit-learn. Split it 80/10/10, standardise the features using training statistics only, and plot a histogram of each feature before and after.
:::

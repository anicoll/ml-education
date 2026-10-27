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

:::tip[Try it]
Download the classic [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) or [California housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) dataset via scikit-learn. Split it 80/10/10, standardise the features using training statistics only, and plot a histogram of each feature before and after.
:::

## Check your understanding

<MultipleChoice
  id="data-features-1"
  question="What is the test set for?"
  options={[
    "Choosing the learning rate",
    "Deciding when to stop training",
    "Reporting how good the final model really is, used once at the very end",
    "Training the model when the training set is too small",
  ]}
  answer={2}
  explanation="Choices like learning rate and when to stop are made on the validation set. The test set stays untouched until the end so its score is honest."
/>

<MultipleChoice
  id="data-features-2"
  question="You standardise every feature using the mean and standard deviation of the whole dataset, then split into train and test. What's the problem?"
  options={[
    "None, that's the standard approach",
    "Data leakage: statistics from the test set influenced how the training data was processed",
    "Standardisation only works on images",
    "The features will no longer be numbers",
  ]}
  answer={1}
  explanation="Compute μ and σ on the training set only, then apply those same values to validation and test. Otherwise information from the test set leaks into training."
/>

<MultipleChoice
  id="data-features-3"
  question="How would one-hot encoding represent the category 'blue' with categories [red, green, blue]?"
  options={["[3]", "[0, 0, 1]", "[1, 1, 0]", "[0.33, 0.33, 0.33]"]}
  answer={1}
  explanation="One-hot puts a single 1 in the position of the category and 0 everywhere else."
/>

<OpenQuestion
  id="data-features-4"
  question="You're building a model to predict whether a customer will cancel their subscription. Each customer has many rows (one per month). Describe how you'd split the data and one way leakage could sneak in.">

Split **by customer**, not by row. Put each customer's rows entirely into train, validation or test (for example 80/10/10 of customers). If you split rows randomly, the same customer appears in both train and test, and the model can score well just by recognising them, which won't happen with real new customers.

Other leakage traps to mention:

- **Using the future.** A feature like "number of support tickets in the month after" includes information that wouldn't exist at prediction time. Only use data from before the moment you'd make the prediction. Often the cleanest split is by time: train on older months, test on the most recent ones.
- **Normalising on everything.** Compute averages and standard deviations on the training customers only.

</OpenQuestion>

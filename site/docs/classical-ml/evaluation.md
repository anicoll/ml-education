---
sidebar_position: 5
title: Evaluating models
---

# Evaluating models

A model is only as trustworthy as the way you measured it. Picking the wrong metric is one of the easiest ways to fool yourself.

## Regression metrics

| Metric | Formula | Notes |
| --- | --- | --- |
| MSE | $\frac{1}{N}\sum (\hat{y}-y)^2$ | Punishes big errors heavily. Units are squared. |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as the target, easier to interpret. |
| MAE | $\frac{1}{N}\sum \lvert\hat{y}-y\rvert$ | Less sensitive to outliers. |
| $R^2$ | $1 - \frac{\text{MSE}}{\text{Var}(y)}$ | 1 is perfect, 0 means "no better than predicting the mean". |

## Classification metrics

Start with the **confusion matrix**. For a spam filter:

| | Predicted spam | Predicted not spam |
| --- | --- | --- |
| **Actually spam** | True positive (TP) | False negative (FN) |
| **Actually not spam** | False positive (FP) | True negative (TN) |

From it:

- **Accuracy** $= \frac{TP + TN}{\text{all}}$. Misleading when classes are imbalanced.
- **Precision** $= \frac{TP}{TP + FP}$. Of the emails flagged as spam, how many really were?
- **Recall** $= \frac{TP}{TP + FN}$. Of the real spam, how much did we catch?
- **F1** is the harmonic mean of precision and recall, a single number balancing both.

There's always a trade-off: lowering the decision threshold catches more spam (higher recall) but flags more real email (lower precision). Which matters more depends on the cost of each mistake. Missing a cancer diagnosis is far worse than a false alarm; deleting a real email is worse than letting some spam through.

**ROC-AUC** summarises performance across all thresholds: 0.5 is random guessing and 1.0 is perfect.

## Evaluating language models

LLMs need different tools:

- **Perplexity** $= e^{\text{loss}}$: roughly, "how many tokens is the model choosing between at each step?" Lower is better. Useful for comparing models with the same tokeniser on the same data.
- **Benchmarks** like MMLU (knowledge), GSM8K (maths word problems) and HumanEval (code). Useful, but models are increasingly trained on data that overlaps with benchmarks, so treat them with suspicion.
- **Your own eval set.** For fine-tuning, the most useful evaluation is a set of 50–200 prompts representative of *your* use case, with answers you've checked. Compare models side by side on it.
- **LLM-as-judge:** use a strong model to grade outputs. Cheap and scalable, but it has biases (it tends to prefer longer answers, for example).

:::info Key idea
Choose the metric that matches the real-world cost of mistakes, and always compare against a dumb baseline (predict the mean, predict the most common class). If you can't beat the baseline, nothing else matters.
:::

:::tip Try it
Create an imbalanced dataset with `make_classification(weights=[0.95, 0.05])`. Train logistic regression and report accuracy, precision, recall and F1. Then compare against a model that always predicts the majority class. Plot precision and recall as you vary the threshold from 0.1 to 0.9.
:::

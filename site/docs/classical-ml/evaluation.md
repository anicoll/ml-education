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

:::info[Key idea]
Choose the metric that matches the real-world cost of mistakes, and always compare against a dumb baseline (predict the mean, predict the most common class). If you can't beat the baseline, nothing else matters.
:::

:::tip[Try it]
Create an imbalanced dataset with `make_classification(weights=[0.95, 0.05])`. Train logistic regression and report accuracy, precision, recall and F1. Then compare against a model that always predicts the majority class. Plot precision and recall as you vary the threshold from 0.1 to 0.9.
:::

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="Kdsp6soqA7o" title="Machine Learning Fundamentals: The Confusion Matrix" channel="StatQuest with Josh Starmer">

How to read a confusion matrix, the table every classification metric is built from.

</YouTube>

<YouTube id="4jRBRDbJemM" title="ROC and AUC, Clearly Explained!" channel="StatQuest with Josh Starmer">

What happens as you slide the decision threshold, and how the ROC curve and AUC summarise it.

</YouTube>

## Check your understanding

<MultipleChoice
  id="eval-1"
  question="Only 1 in 1,000 transactions is fraud. A model that always predicts 'not fraud' has what accuracy?"
  options={["0.1%", "50%", "99.9%", "It can't be calculated"]}
  answer={2}
  explanation="It's right 999 times out of 1,000, and completely useless. That's why accuracy misleads on imbalanced data."
/>

<MultipleChoice
  id="eval-2"
  question="A spam filter flagged 100 emails. 90 really were spam. There were 300 spam emails in total. What are its precision and recall?"
  options={["Precision 90%, recall 30%", "Precision 30%, recall 90%", "Precision 90%, recall 90%", "Precision 100%, recall 30%"]}
  answer={0}
  explanation="Precision = TP / flagged = 90/100. Recall = TP / all real spam = 90/300 = 30%. It's accurate when it flags, but misses most spam."
/>

<MultipleChoice
  id="eval-3"
  question="For a medical screening test where missing a disease is much worse than a false alarm, which metric matters most?"
  options={["Precision", "Recall", "Accuracy", "Perplexity"]}
  answer={1}
  explanation="Recall measures how many real cases you catch. You accept more false alarms (lower precision) to miss fewer real cases."
/>

<MultipleChoice
  id="eval-4"
  question="What does a regression model with R² = 0 mean?"
  options={["Perfect predictions", "No better than always predicting the mean", "Worse than random", "The model has no parameters"]}
  answer={1}
  explanation="R² compares your error with the error of predicting the average. 0 means no improvement over that baseline; 1 is perfect."
/>

<OpenQuestion
  id="eval-5"
  question="You've fine-tuned a model to answer customer support questions. Describe how you would evaluate whether it's actually better than the original model.">

A worked plan:

1. **Build an eval set before training.** Collect 50–200 real support questions that were *not* in the training data, covering the common cases and a few tricky ones. For each, write down (or take from past tickets) what a good answer contains.
2. **Run both models** (original and fine-tuned) on every question with the same settings.
3. **Compare side by side.** Score each answer, for example 0 (wrong or harmful), 1 (partly right), 2 (correct and well formatted). Hide which model wrote which answer while scoring, so you don't favour your own.
4. **Optionally use an LLM as a judge** to score faster, but spot-check its grades, since judges tend to prefer longer answers.
5. **Report the numbers and look at the failures.** "Fine-tuned wins 70% of head-to-heads" is useful; reading the 30% it lost tells you what to fix next.

The key ideas: a held-out set that represents your real use case, a baseline to compare against, and a consistent scoring method.

</OpenQuestion>

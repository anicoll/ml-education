---
sidebar_position: 4
title: Overfitting and generalisation
---

# Overfitting and generalisation

The goal of ML is not to do well on the training data. It's to do well on data the model has never seen. This page is about the gap between those two.

## Underfitting and overfitting

- **Underfitting:** the model is too simple to capture the pattern. Training error and validation error are both high. Example: a straight line fit to a curve.
- **Overfitting:** the model is so flexible it memorises the training data, including its noise. Training error is low but validation error is high. Example: a degree-20 polynomial threading through every point.
- **Good fit:** both errors are low and close together.

```mermaid
xychart-beta
  title "Typical training curves when overfitting"
  x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  y-axis "Loss" 0 --> 1
  line [0.9, 0.6, 0.45, 0.35, 0.28, 0.22, 0.17, 0.13, 0.1, 0.08]
  line [0.92, 0.65, 0.5, 0.43, 0.4, 0.4, 0.42, 0.46, 0.5, 0.55]
```

The lower line is training loss, which keeps falling. The upper line is validation loss, which falls, bottoms out, then rises. The bottom of the validation curve is where you'd want to stop.

## The bias-variance trade-off

The classical way to think about this:

- **Bias** is error from wrong assumptions (a line can't fit a curve). Simple models have high bias.
- **Variance** is error from being too sensitive to the specific training sample. Flexible models have high variance.

Making a model more flexible reduces bias but increases variance. The sweet spot is in the middle.

:::note
Modern deep learning complicates this picture. Very large networks trained on lots of data often generalise *better* as they get bigger, a phenomenon studied under the name "double descent". The trade-off is still a useful mental model for small models and small data.
:::

## How to fight overfitting

1. **More data.** The single most reliable fix.
2. **Simpler model.** Fewer parameters, shallower trees.
3. **Regularisation.** Add a penalty for large weights to the loss. **L2 regularisation** (also called *weight decay*) adds $\lambda \sum_j w_j^2$, nudging weights towards zero.
4. **Early stopping.** Track validation loss and stop when it stops improving.
5. **Data augmentation.** Create variations of training examples (flip, crop or rotate images; paraphrase text).
6. **Dropout** (for neural networks): randomly switch off neurons during training. See [Regularisation](../neural-networks/regularisation.md).

## Cross-validation

With small datasets, a single validation split is noisy. **k-fold cross-validation** splits the training data into $k$ parts, trains $k$ times using each part once as validation, and averages the scores. It's standard for classical ML; it's usually too expensive for deep learning.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores.mean(), scores.std())
```

:::tip[Try it]
Generate 30 noisy points from $y = \sin(x)$. Fit polynomials of degree 1, 3, 9 and 20 with `numpy.polyfit`, and plot each against the data. Then generate 1,000 fresh points and compute the error of each fit on them. Which degree wins? Now add L2 regularisation (`sklearn.linear_model.Ridge` with polynomial features) to the degree-20 model and see what happens.
:::

## Check your understanding

<MultipleChoice
  id="overfit-1"
  question="Training loss keeps falling but validation loss has started rising. What should you do?"
  options={[
    "Keep training: training loss is what matters",
    "Stop (or roll back to the checkpoint with the lowest validation loss): the model has started overfitting",
    "Increase the model size",
    "Evaluate on the test set to decide",
  ]}
  answer={1}
  explanation="This is the classic overfitting curve. Early stopping keeps the model from the bottom of the validation curve."
/>

<MultipleChoice
  id="overfit-2"
  question="Training error and validation error are both high and close together. What's the likely problem?"
  options={["Overfitting", "Underfitting: the model is too simple", "Data leakage", "Too much regularisation is impossible, so it must be the data"]}
  answer={1}
  explanation="If the model can't even fit the training data, it's too simple (or too heavily regularised). Try a more flexible model or better features."
/>

<MultipleChoice
  id="overfit-3"
  question="What does L2 regularisation (weight decay) add to the loss?"
  options={["A penalty proportional to the sum of squared weights", "Random noise on the inputs", "Extra training data", "A penalty for training too long"]}
  answer={0}
  explanation="λ Σ w² nudges weights towards zero, which makes the model smoother and less able to memorise noise."
/>

<OpenQuestion
  id="overfit-4"
  question="Name three ways to fight overfitting and explain in a sentence why each one works.">

Any three of these, for example:

- **More data:** with more examples, the noise in any one example matters less, and memorising them all gets harder than learning the real pattern.
- **A simpler model:** fewer parameters means less capacity to memorise.
- **Regularisation (weight decay):** penalising large weights pushes the model towards smoother, simpler functions.
- **Early stopping:** stop at the point where validation loss is lowest, before the model starts memorising.
- **Data augmentation:** flipped, cropped or paraphrased copies teach the model which changes don't matter, which is like getting more data for free.
- **Dropout:** randomly switching off neurons stops the network relying on any single one, so it learns more robust features.

</OpenQuestion>

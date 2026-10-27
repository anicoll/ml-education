---
sidebar_position: 1
title: Glossary
---

# Glossary

**Activation.** The output of a layer (or neuron) for a given input. Also short for *activation function*.

**Activation function.** The non-linear function applied after a linear layer, such as ReLU, GELU or sigmoid.

**Adam / AdamW.** Optimisers that adapt the step size per parameter using running averages of gradients and squared gradients. AdamW is the default for transformers.

**Attention.** A mechanism letting each token build its representation from a weighted mix of other tokens, with weights based on query–key similarity.

**Autograd.** Automatic differentiation: the framework records operations and computes gradients for you via backpropagation.

**Backpropagation.** Computing gradients of the loss with respect to all parameters by applying the chain rule backwards through the network.

**Base model.** A model after pre-training only, before instruction tuning. It continues text rather than following instructions.

**Batch.** A group of examples processed together in one step.

**bf16 (bfloat16).** A 16-bit float with the same range as fp32 but less precision. Preferred for training LLMs on Ampere and newer GPUs.

**Chat template.** The exact text format, with special tokens, an instruct model expects for conversations.

**Checkpoint.** A saved copy of model (and often optimiser) state during training.

**Context length (context window).** The maximum number of tokens a model can process at once.

**Cross-entropy.** The standard classification loss: the negative log probability assigned to the correct class or token.

**DPO (Direct Preference Optimisation).** A preference-tuning method that trains on pairs of preferred and rejected responses, without a separate reward model.

**Dropout.** Regularisation that randomly zeros activations during training.

**Embedding.** A learned vector representing a discrete item, such as a token.

**Epoch.** One full pass through the training dataset.

**Fine-tuning.** Continuing to train a pre-trained model on a smaller, task-specific dataset.

**GGUF.** The file format used by llama.cpp and Ollama for (usually quantised) models.

**Gradient.** The vector of partial derivatives of the loss with respect to each parameter; points in the direction of steepest increase.

**Gradient accumulation.** Summing gradients over several small batches before one optimiser step, to simulate a larger batch.

**Gradient checkpointing.** Saving memory by recomputing some activations during the backward pass instead of storing them.

**Hallucination.** A model confidently producing false or made-up information.

**Hyperparameter.** A setting chosen by you rather than learned, such as learning rate, batch size or number of layers.

**Inference.** Using a trained model to make predictions or generate output.

**Instruct model.** A model fine-tuned to follow instructions and hold conversations.

**KV cache.** Stored keys and values of previous tokens, reused during generation to avoid recomputation.

**Learning rate.** How large a step the optimiser takes each update. The most important hyperparameter.

**Logits.** Raw, unnormalised scores output by a model before softmax.

**LoRA.** Low-Rank Adaptation: fine-tuning by training small low-rank matrices added to frozen weights.

**Loss.** A number measuring how wrong the model is; training minimises it.

**MoE (Mixture of Experts).** An architecture where each token is routed to a few of many expert MLPs per layer.

**Overfitting.** When a model fits its training data well but generalises poorly to new data.

**Parameter.** A learned number in the model (a weight or bias).

**Perplexity.** $e^{\text{loss}}$ for a language model; lower is better.

**Pre-training.** The first, large-scale training phase on huge amounts of unlabelled data, usually by next-token prediction.

**QLoRA.** LoRA on top of a base model stored in 4-bit, to save memory.

**Quantisation.** Storing weights (and sometimes activations) with fewer bits.

**RAG (Retrieval-Augmented Generation).** Fetching relevant documents and adding them to the prompt so the model can use them.

**Regularisation.** Any technique that reduces overfitting, such as weight decay, dropout or data augmentation.

**Residual connection.** Adding a layer's input to its output, so the layer learns a change. Lets gradients flow through deep networks.

**RLHF.** Reinforcement Learning from Human Feedback: tuning a model with RL against a reward model trained on human preferences.

**SFT (Supervised Fine-Tuning).** Fine-tuning on example prompts and ideal responses.

**Softmax.** Converts a vector of logits into a probability distribution.

**Temperature.** Divides logits before softmax when sampling; lower is more deterministic, higher more random.

**Tensor.** An n-dimensional array; the basic data structure in PyTorch.

**Token.** A unit of text (often a word piece) that a language model reads and writes.

**Tokeniser.** Converts text to token IDs and back.

**Transformer.** The neural network architecture built from attention and MLP blocks with residual connections, used by modern LLMs.

**Validation set.** Data held out from training, used to tune choices and detect overfitting.

**VRAM.** The GPU's memory. Your RTX 3070 has 8 GB.

**Weight decay.** Shrinking weights slightly each step to regularise; equivalent to an L2 penalty for plain SGD.

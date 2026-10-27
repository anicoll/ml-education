---
sidebar_position: 2
title: Further resources
---

# Further resources

A short, opinionated list. Better to finish two of these than start ten.

## Video courses

- **Andrej Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html).** Builds everything from backprop to GPT in code. The single best companion to Parts 3–5 of this site.
- **3Blue1Brown, [Neural networks series](https://www.3blue1brown.com/topics/neural-networks).** Beautiful visual explanations of backprop, transformers and attention.
- **fast.ai, [Practical Deep Learning for Coders](https://course.fast.ai/).** Top-down and practical: train real models first, understand them later.
- **Stanford CS231n** (vision) and **CS224n** (NLP). University-level depth; lectures are online.

## Books

- **Dive into Deep Learning** ([d2l.ai](https://d2l.ai/)). Free, interactive, with runnable code for every concept.
- **Understanding Deep Learning** by Simon Prince ([udlbook](https://udlbook.github.io/udlbook/)). Free, modern, excellent diagrams.
- **Build a Large Language Model (From Scratch)** by Sebastian Raschka. A whole book doing what Part 5 does, in more depth.
- **Hands-On Machine Learning** by Aurélien Géron. Strong on classical ML and scikit-learn.

## Reinforcement learning

- **Reinforcement Learning: An Introduction** by Sutton and Barto. The classic textbook, free online. Read chapters 1–6 alongside Part 7.
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course). Free, hands-on, from Q-learning to PPO.
- [OpenAI Spinning Up](https://spinningup.openai.com/). Clear explanations of policy gradients, PPO and SAC.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and [CleanRL](https://docs.cleanrl.dev/). Reliable implementations; CleanRL puts each algorithm in a single readable file.
- [Gymnasium](https://gymnasium.farama.org/). The standard environment interface.
- [TMRL](https://github.com/trackmania-rl/tmrl) and [Linesight](https://github.com/Linesight-RL/linesight). Trackmania RL frameworks for the capstone.

## Hugging Face

- [Hugging Face LLM Course](https://huggingface.co/learn). Transformers, datasets, fine-tuning, all with their libraries.
- [PEFT docs](https://huggingface.co/docs/peft) and [TRL docs](https://huggingface.co/docs/trl). Reference for LoRA, QLoRA, SFT and DPO.

## Papers worth reading

You can read these now. Skim for the main idea and the diagrams; don't worry about every equation.

- *Attention Is All You Need* (Vaswani et al., 2017). The transformer.
- *Language Models are Few-Shot Learners* (Brown et al., 2020). GPT-3 and in-context learning.
- *Training Compute-Optimal Large Language Models* (Hoffmann et al., 2022). The "Chinchilla" scaling laws.
- *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021).
- *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023).
- *Deep Residual Learning for Image Recognition* (He et al., 2015). ResNets.
- *Human-level control through deep reinforcement learning* (Mnih et al., 2015). DQN on Atari.
- *Proximal Policy Optimization Algorithms* (Schulman et al., 2017). PPO.
- *Soft Actor-Critic* (Haarnoja et al., 2018). SAC.

## Tools and code

- [micrograd](https://github.com/karpathy/micrograd), [nanoGPT](https://github.com/karpathy/nanoGPT), [build-nanogpt](https://github.com/karpathy/build-nanogpt). Small, readable reference implementations.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Ollama](https://ollama.com). Local inference.
- [Unsloth](https://github.com/unslothai/unsloth). Fast, memory-efficient LoRA and QLoRA, great for an 8 GB GPU.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm). Running and fine-tuning models on Apple Silicon.
- [TensorFlow Playground](https://playground.tensorflow.org/). Train tiny networks in the browser.

## Keeping up

The field moves fast. A light touch is enough: follow the Hugging Face trending models page, and read a paper when a new idea keeps coming up. Depth on the fundamentals in this course will outlast any particular model release.

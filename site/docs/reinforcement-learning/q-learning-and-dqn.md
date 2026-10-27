---
sidebar_position: 2
title: Q-learning and DQN
---

# Q-learning and DQN

Value-based methods learn how good each action is, then pick the best one. They're the most intuitive place to start, and the DQN family is behind some of the best Trackmania AIs.

## The Bellman equation

The value of taking action $a$ in state $s$ is the immediate reward plus the discounted value of the best thing you can do next:

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

This recursive relationship is the **Bellman equation**. It lets you learn Q-values from single steps of experience, without waiting until the end of an episode.

## Tabular Q-learning

When there are only a few states and actions, store $Q$ in a table. After each step $(s, a, r, s')$, nudge the table entry towards the Bellman target:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{target}} - Q(s, a)\Big]
$$

$\alpha$ is a learning rate. The bracketed term is the **temporal-difference (TD) error**: how surprised you were.

To explore, use **epsilon-greedy**: with probability $\varepsilon$ take a random action, otherwise take the best known one. Start with $\varepsilon = 1$ (fully random) and decay it over time.

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, eps = 0.1, 0.99, 1.0

for episode in range(2000):
    s, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < eps:
            a = env.action_space.sample()          # explore
        else:
            a = int(np.argmax(Q[s]))               # exploit
        s2, r, terminated, truncated, _ = env.step(a)
        target = r + gamma * np.max(Q[s2]) * (not terminated)
        Q[s, a] += alpha * (target - Q[s, a])
        s, done = s2, terminated or truncated
    eps = max(0.05, eps * 0.995)

print(Q.argmax(axis=1).reshape(4, 4))   # the learned policy: best action per square
```

FrozenLake has 16 states and 4 actions, so the table has 64 entries. A racing game has effectively infinite states (every possible screen), so a table is impossible.

## Deep Q-Networks (DQN)

Replace the table with a neural network $Q_\theta(s, a)$ that takes the state as input and outputs one Q-value per action. Train it to reduce the TD error, exactly like a regression problem where the targets come from the Bellman equation:

$$
L(\theta) = \Big(r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a)\Big)^2
$$

In 2013–2015, DeepMind's DQN learned to play dozens of Atari games from pixels with this approach. Two tricks made it stable:

1. **Experience replay.** Store every transition $(s, a, r, s')$ in a big buffer and train on random mini-batches from it. This breaks the correlation between consecutive frames and reuses each experience many times. That matters a lot when experience is expensive to collect, as it is in a real-time game.
2. **Target network.** Compute targets with a separate, slowly updated copy of the network ($\bar\theta$). Otherwise the target moves every time you update, and training chases its own tail.

## Improvements you'll hear about

| Variant | Fix |
| --- | --- |
| Double DQN | Reduces Q-value overestimation by choosing and evaluating the next action with different networks. |
| Dueling DQN | Splits the network into "how good is this state" and "how much better is each action". |
| Prioritised replay | Replays surprising transitions (big TD error) more often. |
| Distributional (C51, QR-DQN, **IQN**) | Learns the whole distribution of returns, not just the average. Linesight uses IQN. |
| Rainbow | Combines most of the above. |

## The limitation: discrete actions

DQN needs to compute $\max_{a'} Q(s', a')$, which means trying every action. That's fine for "left / right / accelerate / brake" combinations, but not for continuous analogue steering. For that you need policy-based or actor-critic methods (next page). Linesight gets around this by giving the car a small set of discrete action combinations, like a keyboard player.

## Using a library

Writing DQN from scratch once is a great exercise. For real experiments, use [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3), which has well-tested implementations:

```python
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50_000,
            exploration_fraction=0.1, target_update_interval=500, verbose=1,
            tensorboard_log="runs/dqn-cartpole")
model.learn(total_timesteps=100_000)
model.save("dqn_cartpole")
```

CartPole is "solved" at an average return of 500 (the episode limit). DQN on CartPole is notoriously sensitive to hyperparameters; the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) has tuned settings for many environments.

:::tip Try it
1. Run the tabular Q-learning code on FrozenLake. Print the learned policy grid and check it reaches the goal. Then set `is_slippery=True` and see how much harder it gets.
2. Train SB3's DQN on CartPole and watch `rollout/ep_rew_mean` in TensorBoard.
3. Stretch goal: write DQN yourself in PyTorch (a replay buffer, a target network and an MLP) and get CartPole above an average return of 400. Following the [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/), a single readable file, is a good guide.
:::

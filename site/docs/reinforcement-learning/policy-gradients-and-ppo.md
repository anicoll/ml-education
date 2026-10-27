---
sidebar_position: 3
title: Policy gradients, PPO and SAC
---

# Policy gradients, PPO and SAC

Instead of learning values and deriving a policy from them, policy gradient methods train the policy network directly. This handles continuous actions naturally (steering at 0.37 rather than "left"), which is what a racing game with an analogue controller needs.

## The policy as a network

For discrete actions, the policy network outputs a probability for each action (softmax, as in [classification](../classical-ml/logistic-regression.md)). For continuous actions, it outputs the **mean and spread of a Gaussian** for each action dimension (steering, throttle), and the action is sampled from it. The spread shrinks as the agent becomes confident, which is how exploration fades over time.

## REINFORCE: the simplest policy gradient

Play an episode. For every action taken, compute the return $G_t$ that followed. Then increase the log-probability of actions that led to high returns, and decrease it for low ones:

$$
\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t
$$

That's it: "do more of what worked". In PyTorch, the loss is just `-(log_probs * returns).mean()`.

The problem is **variance**. Returns vary hugely from episode to episode due to luck, so the gradient is very noisy and learning is slow.

## Actor-critic: adding a baseline

Rather than scaling by the raw return, scale by how much *better than expected* it was. This is the **advantage**:

$$
A(s_t, a_t) = G_t - V(s_t)
$$

A second network, the **critic**, learns $V(s)$. The policy network is the **actor**. If an action led to a return better than the critic expected, the advantage is positive and the action becomes more likely. This dramatically reduces variance.

## PPO: the reliable default

**Proximal Policy Optimisation (PPO)** is an actor-critic method that adds one key safeguard: don't let any single update change the policy too much. It clips the update so the new policy's probability for an action stays within about ±20% of the old one:

$$
L^{\text{CLIP}} = \mathbb{E}\Big[\min\big(\rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big],
\qquad \rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

The clip keeps training stable, and PPO works reasonably well across a huge range of problems with little tuning. It's the usual first choice, and it's also the algorithm behind the original RLHF for ChatGPT.

PPO is **on-policy**: it learns only from data collected by the current policy, then throws that data away. That's simple and stable, but wasteful when collecting data is slow. To compensate, people usually run many copies of the environment in parallel.

## SAC: sample-efficient continuous control

**Soft Actor-Critic (SAC)** is **off-policy**: like DQN, it stores experience in a replay buffer and reuses it many times. It also rewards the policy for staying random where it can (an "entropy bonus"), which keeps it exploring. SAC is one of the strongest algorithms for continuous control, and it's very **sample-efficient**.

That matters for Trackmania. TrackMania 2020 runs in real time, so you can't collect experience faster than the game plays. Reusing every frame many times from a replay buffer is the right trade-off, which is why **TMRL uses SAC** by default.

| | PPO | SAC |
| --- | --- | --- |
| On/off-policy | On-policy | Off-policy (replay buffer) |
| Actions | Discrete or continuous | Continuous |
| Sample efficiency | Lower | Higher |
| Stability and ease | Very robust | Robust, more hyperparameters |
| Best when | The simulator is fast or can run in parallel | Each sample is expensive (real time, real robots) |

## With Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO, SAC

# PPO on a discrete task
model = PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=1)
model.learn(total_timesteps=100_000)

# SAC on a continuous task
model = SAC("MlpPolicy", gym.make("Pendulum-v1"), verbose=1)
model.learn(total_timesteps=20_000)
```

:::info Key idea
Policy gradients push up the probability of actions that did better than expected. PPO adds a clip so updates stay small and stable. SAC reuses past experience from a replay buffer, which makes it the better choice when experience is slow to collect.
:::

:::tip Try it
1. Train PPO and DQN on CartPole for the same number of steps and plot both learning curves. Which learns faster, and which is more stable?
2. Train SAC on `Pendulum-v1` (continuous torque). Then try PPO on the same task and compare how many steps each needs.
3. Stretch goal: implement REINFORCE yourself on CartPole in about 50 lines of PyTorch. Then subtract a simple baseline (the average return) and see how much faster it learns.
:::

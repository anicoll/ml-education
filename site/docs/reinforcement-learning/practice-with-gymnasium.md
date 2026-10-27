---
sidebar_position: 4
title: Practice with Gymnasium
---

# Practice with Gymnasium

[Gymnasium](https://gymnasium.farama.org/) (the maintained successor to OpenAI Gym) provides a standard interface to hundreds of RL environments. Every environment, from CartPole to a Trackmania wrapper, looks the same to your code, so the skills transfer directly.

## The interface

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()                  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:                         # crashed/finished, or time limit
        obs, info = env.reset()
env.close()
```

- `observation_space` and `action_space` describe the shapes: `Discrete(2)` for left/right, `Box(...)` for continuous vectors or images.
- `terminated` means the episode ended for a real reason (the pole fell, the car finished). `truncated` means it hit a time limit. The difference matters when computing value targets.

TMRL exposes Trackmania through this same `reset` / `step` interface, so everything here carries straight over to the capstone.

## A progression of environments

Work through these in order. Each adds one new difficulty.

| Environment | Observation | Actions | New challenge | Rough training (SB3) |
| --- | --- | --- | --- | --- |
| `CartPole-v1` | 4 numbers | 2 discrete | The basics | PPO: about 100k steps, a few minutes on CPU |
| `LunarLander-v3` | 8 numbers | 4 discrete (or continuous variant) | Shaped rewards, crashing vs landing | PPO: about 1M steps, under an hour |
| `Pendulum-v1` | 3 numbers | 1 continuous | Continuous control | SAC: about 20k steps, minutes |
| `CarRacing-v3` | 96×96 colour image | 3 continuous (steer, gas, brake) or 5 discrete | Learning from pixels, driving | PPO or SAC with a CNN: about 1–2M steps, several hours |

Version suffixes change between Gymnasium releases (for example `CarRacing-v2` in older versions). If `gym.make` complains, check the current name in the docs. LunarLander and CarRacing need the Box2D extra:

```bash
pip install "gymnasium[box2d]" stable-baselines3 tensorboard
```

On Linux or WSL you may need `sudo apt install swig` first; on macOS, `brew install swig`.

## CarRacing: your Trackmania warm-up

CarRacing is a top-down racing game where the agent sees the screen and controls steering, gas and brake. The reward is +1000/N for every track tile visited (N tiles in total) minus 0.1 per frame. In other words, **progress along the track, with a time penalty**: the same idea as TMRL's reward. If you can get an agent driving here, you understand most of what the Trackmania capstone needs.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_vec_env("CarRacing-v3", n_envs=8)
env = VecFrameStack(env, n_stack=4)          # stack frames so the agent can see motion

model = PPO("CnnPolicy", env, n_steps=512, batch_size=128, learning_rate=3e-4,
            verbose=1, tensorboard_log="runs/carracing")
model.learn(total_timesteps=2_000_000)
model.save("ppo_carracing")
```

Things to try and notice:

- **Frame stacking:** remove it and the agent can't tell its speed or direction.
- **Grayscale and cropping:** shrink the input (SB3 has wrappers for this) and training gets faster.
- **Discrete vs continuous:** `gym.make("CarRacing-v3", continuous=False)` gives 5 actions, so you can compare DQN with PPO and SAC.
- **Watch it drive** every few hundred thousand steps. The learning curve tells you the score; watching tells you *how* it's getting it (and whether it's exploiting something).

## Practical RL habits

1. **Always compare against random.** You did this on the first RL page.
2. **Run several seeds.** RL results vary a lot between runs. One good run proves little; three runs that all learn prove something.
3. **Log everything to TensorBoard:** episode return, episode length, losses, entropy (how random the policy still is).
4. **Record videos** of the agent with Gymnasium's `RecordVideo` wrapper. Watching is the fastest way to spot reward hacking.
5. **Start from tuned hyperparameters.** The [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) has settings that are known to work.
6. **Expect slow starts.** RL curves are often flat for a long time before taking off. Don't kill a run too early, but set a budget.

## Which machine?

- Small MLP policies (CartPole, LunarLander, Pendulum) are often as fast on the **CPU** as on a GPU, because the environment simulation is the bottleneck, not the network. Either your PC or your Mac is fine.
- Image-based environments (CarRacing) benefit from a GPU for the CNN. The 3070 with CUDA is the most straightforward. Gymnasium and SB3 also run on the Mac.
- Trackmania needs Windows. See the [capstone](./capstone-trackmania.md).

:::tip Try it
Work through the table above in `notebooks/07-rl/`, one environment at a time:

1. PPO on CartPole until it reliably reaches 500.
2. PPO on LunarLander until the average return passes 200 (counted as solved).
3. SAC on Pendulum.
4. PPO with a CNN on CarRacing. Aim for an average return above 700, and record a video of the best run. Then try SAC on the same environment and compare.

Keep a table in `notes/rl-log.md` of each run: algorithm, key hyperparameters, steps, wall-clock time and final return.
:::

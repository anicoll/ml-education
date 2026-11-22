---
sidebar_position: 1
title: Overview
---

# Working with AI agents

The rest of this course is about how models are built. This part is about **using** them well as a software engineer: handing real coding work to AI agents, and getting code back that you can trust.

It leans heavily on the recent talks and writing of **Lauren Tan** ([@poteto on X](https://x.com/poteto)), an engineer on the Grok Bot team and the author of [pstack](https://github.com/cursor/plugins/tree/main/pstack), a public collection of agent skills. Her team reports shipping thousands of pull requests a month, mostly written and merged by agents, with a very low revert rate. The numbers are less interesting than *how*: almost none of it comes from clever prompting, and almost all of it comes from the environment the agents work in.

:::note[A different kind of "agent"]
In [Part 7](../reinforcement-learning/rl-basics.md), an *agent* is a policy that picks actions to maximise reward. In this part, an *agent* is a language model that calls tools in a loop to finish a task. The two share a shape (observe, act, get feedback, repeat), which is why the word got reused, but they are trained and used very differently.
:::

## What you'll learn

| Lesson | The big idea |
| --- | --- |
| [Agents vs bots](./agents-vs-bots.md) | A chat, an agent and a bot are three different things, and they suit different jobs |
| [Loops](./loops.md) | An agent is a loop. Verification is what lets the loop run without you |
| [Trust and the environment](./trust-and-environment.md) | Make the right thing the easy thing: code and checks beat longer instructions |
| [Verification](./verification.md) | Review the evidence, not every line of the diff |
| [Improving your AI usage](./improving-your-ai-usage.md) | A step-by-step playbook to apply all of this to your own work |

## Primary sources

Everything in this part is a summary. Go to the source when you can:

- **Talk:** *All Code Is Slop (Except Mine)*, AI Engineer CODE, November 2026 ([video](https://www.youtube.com/watch?v=O_zcc6Xl12g), [talk page](https://ai.engineer/code/2026/talks/lauren-tan)).
- **Livestream:** with Matt Pocock on shipping thousands of PRs a month, October 2026 ([video](https://www.youtube.com/watch?v=MN9dGgmLyso)).
- **Conversation:** *Are Agents About to Replace Software Engineering?* with Roshan Sadanani ([video](https://www.youtube.com/watch?v=A63sedG-p5Q)).
- **X posts:**
  - [The Grok Bot launch](https://x.com/poteto/status/2087227636590465030): "Bots do work on their own computers while you're at meetings, while you're having lunch, and while you sleep."
  - [Agents merging their own PRs](https://x.com/poteto/status/2087244771849089270), with code review moving to "looking at what they landed on main".
  - [The Complete Guide to pstack, Part 1](https://x.com/poteto/status/2094457600259842065).
  - [On trust before parallelism](https://x.com/poteto/status/2102050467505430555).
  - [When she uses a bot vs a big multi-agent project](https://x.com/poteto/status/2092852684751781905).
  - [Her AI Engineer talk announcement](https://x.com/poteto/status/2121673563489140863).
- **Code:** the [pstack README](https://github.com/cursor/plugins/blob/main/pstack/README.md) and its [guide](https://github.com/cursor/plugins/blob/main/pstack/docs/guide/README.md).

:::tip[Read critically]
These ideas come from one team with a very well-tooled codebase. The principles transfer; the exact numbers and tools may not. Several write-ups of her talks quote different PR counts because they measure different things (one person vs the whole team, features vs all changes).
:::

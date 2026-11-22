---
sidebar_position: 6
title: Improving your AI usage
---

# Improving your AI usage

The earlier lessons were the ideas. This one is a practical playbook, roughly in the order you'd adopt it. Each step comes from Lauren Tan's talks and posts, or from guides built on them.

## 1. Start with intent, not code

When execution is cheap, the bottleneck moves to **saying what you want**. She has said "domain expertise is more important than ever", and her team turns down many finished PRs because the feature wasn't agreed on, to avoid "a bit of a Frankensteiny product."

Practical habits:

- **Ask the agent to restate the goal first:** "restate in your own words what you think my goals are." pstack's `/in-your-own-words` does exactly this. Misunderstandings show up before any code is written.
- **Write the finish condition down** ([Loops](./loops.md)).
- **Ask how something works before changing it.** pstack has `/how` and `/why` for this, and they're great for learning a new codebase.

## 2. Make one agent reliable before running many

Her repeated message is **trust first, then parallel**. Early on she could only supervise one to five agent conversations at once. Opening many cloud agents before trust exists, as one write-up put it, produces "a pile of garbage pull requests."

So: run one agent, watch where you step in, and fix the reasons you had to.

## 3. Turn every correction into something permanent

Each time you correct an agent, ask which [pyramid layer](./trust-and-environment.md#the-inverted-pyramid) could make the correction unnecessary next time, starting from the strongest:

1. Can the **architecture or API** make the mistake impossible?
2. Can a **type, lint rule or test** catch it?
3. Can **verification** catch it?
4. Only then: a line in a **skill or rules file**.

## 4. Mine your own history for skills

A **skill** is a reusable, written-down procedure the agent loads when it's relevant (Claude Code and Cursor both support them). She describes a skill as "really just process" written in language.

Her approach: go back through old agent conversations, find the corrections and manual steps you keep repeating, and turn those into skills. Her `/recall` skill came from this. pstack's `/automate-me` drafts a personal "mode" skill from how you work.

Some guidance on skills from the same sources:

- **Start with zero skills** and add them only when a real repeated need shows up.
- **Prefer small, focused skills** to one giant instructions file.
- **Put deterministic steps in code** (a script or CLI) and keep only judgment in the skill text.
- **Expect skills to shrink** as models improve. Delete what's no longer needed.

## 5. Buffer feedback before fixing

Rather than fixing each problem the moment it appears, she has an agent log issues to a document and reviews it every few days. Often several reports turn out to be the same root cause, and one structural fix beats five patches. For bots, group related bug reports into one project rather than one agent per bug.

## 6. Review differently

- Ask for **small, ordered PRs** rather than one huge one.
- Review **verification artifacts** (screenshots, videos, benchmarks) first, code second.
- Be **skeptical in triage**: have a separate agent try to break the change (pstack's `/interrogate`).
- As trust grows, move from reviewing before merge to **sampling what landed** on `main`.

## 7. Keep your own skills sharp

Using agents well doesn't mean understanding less. pstack even has a `/teach` skill for having the agent explain something until you understand it. For a learner, that's the best use of all: get the agent to explain the training loop or a confusing error, then write the code yourself.

## Anti-patterns to avoid

From a guide built on her posts:

- Routines that run too often.
- Vague finish conditions ("make it better").
- Several parallel agents editing the **same** working copy.
- Reporting success because the build is green.
- Relying only on soft rules in prose.

:::tip[Try it]
Over your next five agent sessions, keep a tally in `notes/agent-corrections.md`: every time you step in, write one line saying what you corrected. At the end, group the lines and, for the most common one, decide which pyramid layer could remove it for good.
:::

## Check your understanding

<MultipleChoice
  id="agent-usage-1"
  question="What's the first thing to do when handing a non-trivial task to an agent?"
  options={[
    "Start ten agents on it at once",
    "Have the agent restate your goal and agree a finish condition",
    "Ask for the code immediately",
    "Write a very long rules file",
  ]}
  answer={1}
  explanation="With cheap execution, the bottleneck is intent. Restating the goal surfaces misunderstandings before any code exists."
/>

<MultipleChoice
  id="agent-usage-2"
  question="You notice you correct your agent the same way in most sessions. What does this playbook suggest?"
  options={[
    "Keep correcting it by hand",
    "Switch to a different model",
    "Make it permanent: a check if possible, otherwise a small skill",
    "Stop using agents",
  ]}
  answer={2}
  explanation="Repeated corrections are the raw material for checks and skills. Mine your history for them."
/>

<MultipleChoice
  id="agent-usage-3"
  question="Which is an anti-pattern?"
  options={[
    "Small, ordered PRs",
    "Several agents editing the same working copy at the same time",
    "A separate verifier agent",
    "A clear finish condition",
  ]}
  answer={1}
  explanation="Parallel agents in one working copy trip over each other's edits. Give each its own branch, worktree or cloud machine."
/>

<OpenQuestion
  id="agent-usage-4"
  question="Here is a correction log from five sessions. Group it, pick the most common problem, and say how you'd remove it for good. Log: (1) agent used print instead of logging, (2) agent forgot to activate the .venv and installed packages globally, (3) agent used print again, (4) agent wrote a 300-line notebook cell, (5) agent installed with pip instead of uv.">

**Worked example:**

**Group the corrections:**

- **Environment and installs:** (2) global install, (5) `pip` instead of `uv`. Two entries.
- **Logging:** (1) and (3) used `print`. Two entries.
- **Notebook style:** (4) one giant cell. One entry.

**Pick one.** Environment and logging tie. Environment wins, because a global install can break other projects, while a stray `print` is harmless.

**Remove it with the strongest layer that works:**

1. **Architecture:** make the right way the only way. Add a `Makefile` or script, for example `make install PKG=numpy`, that runs `uv add` inside the project. The agent no longer needs to remember anything.
2. **Static check:** add a check that fails if a `requirements.txt` file appears, or if `pyproject.toml` changes without `uv.lock` changing too. Its error message tells the agent to use `uv add`.
3. **Skill (last resort):** a short note in the project's agent instructions: "Install packages with `uv add`, never `pip install`."

**Then logging:** a lint rule (ruff has a rule, `T201`, that flags `print`) catches it every time.

Notice that each fix moves the knowledge out of your head and into the project, where every future agent, and future you, gets it for free.

</OpenQuestion>

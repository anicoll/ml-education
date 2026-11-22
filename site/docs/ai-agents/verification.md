---
sidebar_position: 5
title: Verification
---

# Verification

When agents write most of the code, reading every line stops scaling. Lauren Tan says her own review "has now really shrunk into looking at verification artifacts", and that "most of my token usage goes to verification." This lesson is about what that means and how to start.

## Tests are evidence, not proof

A passing test suite tells you the code does what the tests check. It doesn't tell you the button is visible, the page loads quickly, or the feature does what the user asked for. Write-ups of her livestream frame it this way: tests are supporting evidence, but **done means you've seen the intended behaviour happen in the running software**.

Before agents could check the running app, she described herself as a "human relay": copying what she saw in the browser's developer tools back to the agent. The fix was to give the agent the same eyes.

## Kinds of verification

| Kind | What the agent does | Good for |
| --- | --- | --- |
| **Static** | Type-checks, lints, builds | Catching whole classes of mistakes cheaply |
| **Tests** | Runs unit and integration tests | Logic and regressions |
| **Runtime** | Launches the app, clicks through it (for example with Playwright or Chrome DevTools), captures screenshots, traces or a video | "Does it actually work?" |
| **Property-based** | Generates many random inputs or UI actions and checks rules always hold | Edge cases nobody thought of |
| **Formal** | Proves properties with tools like TLA+ or Lean | Critical flows and tricky algorithms |

Match the check to the change. A copy edit needs a build; a new checkout flow needs runtime verification; a concurrency change might deserve a model in TLA+.

## Verification artifacts

Instead of a diff, ask the agent for **evidence**: a screenshot of the new page, a short video of the flow working, a before/after benchmark, a diagram of what changed. You can judge a ten-second video far faster than a 400-line diff, and it checks the thing you actually care about.

## Tools that make verification cheap

Her team built two things worth copying:

- **A small CLI the agent can use** to launch the app, drive it and capture traces. Build it once and reuse it every task, instead of the agent writing a fresh throwaway script each time. A guide based on her posts recommends agent-friendly options like `--dry-run` and `--json` output.
- **A feature map:** a "user-facing sitemap" listing the app's features and how to reach each one, so the agent knows what to check and where. It's kept up to date automatically.

pstack has skills for this: `/create-verification-skill` sets one up for your project and `/maintain-verification-skill` keeps it current.

The general rule from the livestream write-ups: **"a solved recurring operation should become a tool the next task can use."** Put deterministic steps in code; leave only the judgment to the agent.

## The judge should not be the author

An agent grading its own work tends to be generous. Use a **separate** verifier: a fresh agent, a different prompt, or a script. pstack's `/swarm` runs several workers and gathers one report, and `/interrogate` tries to break a diff. She has also said multi-agent verification is token-hungry, and that she cut it from ten verifiers to one when that was enough.

## Review after the fact

With strong checks in place, her team lets agents merge most of their own PRs. She then **samples** what landed on `main`, reverting or fixing anything bad, and turns each problem into a new check. She has also noted that PR count is "a poor basis for granting authority": what matters is the evidence for a given task, what the checks don't cover, and how easy it is to undo a mistake. Sensitive areas, such as infrastructure, still get human review first.

:::warning[Don't copy the autonomy before the checks]
Auto-merging works for her team because the environment and verification were built first. Without them, it just ships unreviewed bugs faster.
:::

:::tip[Try it]
Next time you ask an agent to change a lesson on this site, add to the prompt: "When you're done, run the build and show me a screenshot of the changed page in the browser." Compare how long it takes you to check that, versus reading the diff.
:::

## Check your understanding

<MultipleChoice
  id="agent-verify-1"
  question="An agent says the feature is done because all tests pass. What's missing?"
  options={[
    "Nothing, passing tests prove it works",
    "Evidence that the intended behaviour actually happens in the running app",
    "A longer commit message",
    "More comments in the code",
  ]}
  answer={1}
  explanation="Tests are supporting evidence. Runtime verification (screenshots, a video, driving the real app) shows the feature works."
/>

<MultipleChoice
  id="agent-verify-2"
  question="Why use a separate agent or script to verify, instead of asking the authoring agent if it's done?"
  options={[
    "It's cheaper",
    "The author tends to grade its own work generously; an independent judge is more likely to catch problems",
    "The authoring agent can't run commands",
    "It's required by GitHub",
  ]}
  answer={1}
  explanation="“The judge is never the author.” Fresh eyes, even artificial ones, are more critical."
/>

<MultipleChoice
  id="agent-verify-3"
  question="Your agent writes a new 40-line script to launch the app and take a screenshot on every task. What should you do?"
  options={[
    "Nothing, that's how agents work",
    "Turn it into one reusable, tested CLI the agent calls every time",
    "Ban screenshots",
    "Ask it to write the script faster",
  ]}
  answer={1}
  explanation="A recurring operation that's been solved should become a tool. It's faster, cheaper and more reliable than regenerating it."
/>

<OpenQuestion
  id="agent-verify-4"
  question="For this course site, sketch a verification plan an agent could follow after editing a lesson. Which checks, in what order, and what artifacts should it hand back?">

**Worked example:**

1. **Static:** run `npm run build` in `site/`. This catches broken MDX, duplicate quiz ids and broken internal links. Stop and fix if it fails.
2. **Runtime:** start the dev server (`npm start`), open the edited lesson with Playwright, and check that:
   - the page loads with no console errors,
   - every `<MultipleChoice>` shows its options, and picking the right answer shows "Correct",
   - every `<OpenQuestion>` reveals its suggested answer after saving,
   - every `<YouTube>` thumbnail loads.
3. **Content:** for each video, confirm the id exists and its title and channel match what the page says.
4. **Artifacts to hand back:** a full-page screenshot of the lesson in light and dark mode, the build output summary, and a short list of the videos checked.

You then review two screenshots and a list instead of the whole diff. If you later find a problem the plan missed (say, a broken image), add a check for it so the next run catches it automatically.

</OpenQuestion>

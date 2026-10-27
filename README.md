# ml-education

A personal workspace for learning machine learning, from first principles through to fine-tuning open-source LLMs on an RTX 3070 PC (WSL2 or Ubuntu) and an Apple Silicon Mac.

| Folder | What's in it |
| --- | --- |
| [`site/`](site/) | The learning website (Docusaurus). The curriculum lives here. |
| [`notebooks/`](notebooks/) | Exercises that go with each lesson's "Try it" boxes. |
| [`projects/`](projects/) | Bigger experiments and builds. |
| [`notes/`](notes/) | Personal notes, questions, run logs. |

## Running the learning site

Requires Node.js 20 or newer.

```bash
cd site
npm install   # first time only
npm start     # http://localhost:3000
```

The site is local only and isn't deployed anywhere.

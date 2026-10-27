# Learning site

The ML curriculum, built with [Docusaurus](https://docusaurus.io/). Local only.

```bash
npm install   # first time only
npm start     # dev server with live reload at http://localhost:3000
npm run build # full build; fails on broken links
```

- Lessons are Markdown files in `docs/`, one folder per part. Order comes from `sidebar_position` front matter and each folder's `_category_.json`.
- Maths uses KaTeX: `$inline$` and `$$block$$`.
- Diagrams use Mermaid fenced code blocks.
- Callouts use admonitions such as `:::tip Try it` and `:::info Key idea`.

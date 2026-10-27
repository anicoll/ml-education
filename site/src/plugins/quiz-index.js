// Scans the docs at build time for <MultipleChoice> and <OpenQuestion> tags so
// the progress page and lesson footers know how many questions each lesson has.
const fs = require('fs');
const path = require('path');

function walk(dir) {
  return fs.readdirSync(dir, {withFileTypes: true}).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(full);
    return /\.mdx?$/.test(entry.name) ? [full] : [];
  });
}

function categoryOf(dir) {
  const file = path.join(dir, '_category_.json');
  if (!fs.existsSync(file)) return {category: 'Start here', categoryPosition: 0};
  const {label, position} = JSON.parse(fs.readFileSync(file, 'utf8'));
  return {category: label, categoryPosition: position ?? 99};
}

module.exports = function quizIndexPlugin(context) {
  const docsDir = path.join(context.siteDir, 'docs');
  return {
    name: 'quiz-index',
    getPathsToWatch() {
      return [path.join(docsDir, '**/*.{md,mdx}')];
    },
    async loadContent() {
      const lessons = {};
      for (const file of walk(docsDir)) {
        const source = fs.readFileSync(file, 'utf8');
        const docId = path.relative(docsDir, file).replace(/\\/g, '/').replace(/\.mdx?$/, '');
        const title = (source.match(/^title:\s*(.+)$/m) || [])[1]?.replace(/^['"]|['"]$/g, '') ?? docId;
        const questions = [...source.matchAll(/<(MultipleChoice|OpenQuestion)\b[^>]*?\bid="([^"]+)"/gs)].map(
          ([, tag, id]) => ({id, type: tag === 'MultipleChoice' ? 'mc' : 'open'}),
        );
        const position = Number((source.match(/^sidebar_position:\s*(\d+)/m) || [])[1] ?? 99);
        const dir = path.dirname(file);
        const category = categoryOf(dir);
        lessons[docId] = {title, questions, position, path: `/docs/${docId}`, ...category};
      }
      const seen = new Set();
      for (const {questions} of Object.values(lessons)) {
        for (const {id} of questions) {
          if (seen.has(id)) throw new Error(`Duplicate question id "${id}" in docs/`);
          seen.add(id);
        }
      }
      return lessons;
    },
    async contentLoaded({content, actions}) {
      actions.setGlobalData({lessons: content});
    },
  };
};

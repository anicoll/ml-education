import {usePluginData} from '@docusaurus/useGlobalData';

export default function useLessons() {
  return usePluginData('quiz-index').lessons;
}

// Summarises saved answers for one lesson's questions.
export function lessonStats(questions, answers) {
  let answered = 0;
  let mc = 0;
  let correct = 0;
  for (const q of questions) {
    const a = answers[q.id];
    if (q.type === 'mc') mc += 1;
    if (!a) continue;
    answered += 1;
    if (a.type === 'mc' && a.correct) correct += 1;
  }
  return {total: questions.length, answered, mc, correct};
}

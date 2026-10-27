import React, {useRef, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {useProgress, exportProgress, importProgress, resetProgress} from '@site/src/lib/progress';
import useLessons, {lessonStats} from '@site/src/lib/useLessons';
import styles from './progress.module.css';

function groupLessons(lessons) {
  const groups = new Map();
  Object.entries(lessons)
    .map(([id, l]) => ({id, ...l}))
    .sort((a, b) => a.categoryPosition - b.categoryPosition || a.position - b.position)
    .forEach((l) => {
      if (!groups.has(l.category)) groups.set(l.category, []);
      groups.get(l.category).push(l);
    });
  return [...groups.entries()];
}

function download(text) {
  const url = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = `ml-course-progress-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Progress() {
  const {answers, completed} = useProgress();
  const lessons = useLessons();
  const fileInput = useRef(null);
  const [message, setMessage] = useState(null);

  const all = Object.entries(lessons);
  const doneCount = all.filter(([id]) => completed[id]).length;
  const totals = lessonStats(all.flatMap(([, l]) => l.questions), answers);
  const pct = all.length ? Math.round((doneCount / all.length) * 100) : 0;

  const onImport = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      importProgress(await file.text());
      setMessage('Progress imported.');
    } catch (err) {
      setMessage(err.message);
    }
    e.target.value = '';
  };

  const onReset = () => {
    if (window.confirm('Delete all your saved answers and completed lessons from this browser?')) {
      resetProgress();
      setMessage('Progress reset.');
    }
  };

  return (
    <Layout title="My progress" description="Your lessons completed and questions answered">
      <main className={clsx('container', styles.page)}>
        <Heading as="h1">My progress</Heading>
        <p>
          Everything here is saved in this browser only (localStorage). Nothing is sent anywhere. If you switch
          browsers or clear site data, use <strong>Export</strong> first and <strong>Import</strong> afterwards.
        </p>

        <div className={styles.summary}>
          <div className={styles.stat}>
            <div className={styles.big}>
              {doneCount}/{all.length}
            </div>
            lessons complete
          </div>
          <div className={styles.stat}>
            <div className={styles.big}>
              {totals.answered}/{totals.total}
            </div>
            questions answered
          </div>
          <div className={styles.stat}>
            <div className={styles.big}>
              {totals.correct}/{totals.mc}
            </div>
            multiple choice correct
          </div>
        </div>
        <div className={styles.bar} aria-label={`${pct}% of lessons complete`}>
          <div className={styles.fill} style={{width: `${pct}%`}} />
        </div>

        {groupLessons(lessons).map(([category, items]) => (
          <section key={category} className={styles.group}>
            <Heading as="h2">{category}</Heading>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Lesson</th>
                  <th>Questions answered</th>
                  <th>Multiple choice correct</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {items.map((l) => {
                  const s = lessonStats(l.questions, answers);
                  return (
                    <tr key={l.id}>
                      <td>
                        <Link to={l.path}>{l.title}</Link>
                      </td>
                      <td>{s.total ? `${s.answered}/${s.total}` : '–'}</td>
                      <td>{s.mc ? `${s.correct}/${s.mc}` : '–'}</td>
                      <td>{completed[l.id] ? <span className={styles.done}>✓ Complete</span> : 'Not yet'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>
        ))}

        <section className={styles.group}>
          <Heading as="h2">Back up or move your progress</Heading>
          <div className={styles.buttons}>
            <button className="button button--primary" onClick={() => download(exportProgress())}>
              Export progress
            </button>
            <button className="button button--secondary" onClick={() => fileInput.current?.click()}>
              Import progress
            </button>
            <button className="button button--outline button--danger" onClick={onReset}>
              Reset everything
            </button>
            <input ref={fileInput} type="file" accept="application/json" hidden onChange={onImport} />
          </div>
          {message && <p className={styles.message}>{message}</p>}
        </section>
      </main>
    </Layout>
  );
}

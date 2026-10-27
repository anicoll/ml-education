import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import {useProgress, setCompleted} from '@site/src/lib/progress';
import useLessons, {lessonStats} from '@site/src/lib/useLessons';
import styles from './styles.module.css';

export default function LessonProgress() {
  const {metadata} = useDoc();
  const {answers, completed} = useProgress();
  const lesson = useLessons()[metadata.id];
  const stats = lessonStats(lesson?.questions ?? [], answers);
  const done = !!completed[metadata.id];

  return (
    <div className={clsx(styles.box, done && styles.done)}>
      <div>
        <strong>{done ? '✓ Lesson complete' : 'Finished this lesson?'}</strong>
        {stats.total > 0 && (
          <div className={styles.stats}>
            {stats.answered}/{stats.total} questions answered
            {stats.mc > 0 && <>, {stats.correct}/{stats.mc} multiple choice correct</>}
          </div>
        )}
        <div className={styles.stats}>
          <Link to="/progress">See all your progress</Link>
        </div>
      </div>
      <button
        className={clsx('button', done ? 'button--secondary' : 'button--success')}
        onClick={() => setCompleted(metadata.id, !done)}>
        {done ? 'Mark as not complete' : 'Mark lesson complete'}
      </button>
    </div>
  );
}

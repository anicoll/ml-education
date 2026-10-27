import React, {useEffect, useState} from 'react';
import clsx from 'clsx';
import {useProgress, saveAnswer} from '@site/src/lib/progress';
import Inline from './inline';
import styles from './styles.module.css';

/**
 * <OpenQuestion id="unique-id" question="...">
 *   Suggested answer in markdown, revealed once the learner saves an answer.
 * </OpenQuestion>
 */
export default function OpenQuestion({id, question, children}) {
  const {answers} = useProgress();
  const saved = answers[id]?.type === 'open' ? answers[id] : null;
  const [text, setText] = useState('');
  const [editing, setEditing] = useState(false);

  // Fill the box with the saved answer once storage has loaded on the client.
  useEffect(() => {
    if (saved && !editing) setText(saved.text);
  }, [saved?.text]); // eslint-disable-line react-hooks/exhaustive-deps

  const revealed = !!saved && !editing;
  const save = () => {
    saveAnswer(id, {type: 'open', text: text.trim()});
    setEditing(false);
  };

  return (
    <div className={clsx(styles.card, revealed && styles.cardOpenDone)}>
      <div className={styles.kind}>Open question</div>
      <div className={styles.question}>
        <Inline>{question}</Inline>
      </div>
      {revealed ? (
        <>
          <div className={styles.yourAnswer}>
            <div className={styles.label}>Your answer</div>
            {saved.text ? <p>{saved.text}</p> : <p className={styles.muted}>(You revealed the answer without writing one.)</p>}
          </div>
          <div className={styles.suggested}>
            <div className={styles.label}>Suggested answer</div>
            {children}
          </div>
          <p className={styles.muted}>
            Compare the ideas, not the wording. If you missed something, edit your answer so your notes are right.
          </p>
          <button className="button button--secondary button--sm" onClick={() => setEditing(true)}>
            Edit my answer
          </button>
        </>
      ) : (
        <>
          <textarea
            className={styles.textarea}
            rows={4}
            value={text}
            placeholder="Write your answer in your own words. It's saved in this browser."
            onChange={(e) => setText(e.target.value)}
          />
          <div className={styles.actions}>
            <button className="button button--primary" disabled={!text.trim()} onClick={save}>
              Save and see suggested answer
            </button>
            {!saved && (
              <button className={clsx('button button--link button--sm', styles.skip)} onClick={save}>
                I'm stuck, just show me
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

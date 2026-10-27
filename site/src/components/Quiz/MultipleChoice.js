import React, {useState} from 'react';
import clsx from 'clsx';
import {useProgress, saveAnswer, clearAnswer} from '@site/src/lib/progress';
import Inline from './inline';
import styles from './styles.module.css';

/**
 * <MultipleChoice id="unique-id" question="..." options={['a', 'b']} answer={1}>
 *   Optional markdown explanation, shown after checking.
 * </MultipleChoice>
 */
export default function MultipleChoice({id, question, options, answer, explanation, children}) {
  const {answers} = useProgress();
  const saved = answers[id];
  const [picked, setPicked] = useState(null);

  const checked = saved?.type === 'mc';
  const choice = checked ? saved.choice : picked;
  const correct = checked && saved.choice === answer;

  const check = () => {
    if (picked === null) return;
    saveAnswer(id, {type: 'mc', choice: picked, correct: picked === answer});
  };
  const retry = () => {
    setPicked(null);
    clearAnswer(id);
  };

  return (
    <div className={clsx(styles.card, checked && (correct ? styles.cardRight : styles.cardWrong))}>
      <div className={styles.kind}>Multiple choice</div>
      <div className={styles.question}>
        <Inline>{question}</Inline>
      </div>
      <ul className={styles.options} role="radiogroup">
        {options.map((opt, i) => {
          const isChoice = choice === i;
          const isAnswer = i === answer;
          return (
            <li key={i}>
              <label
                className={clsx(
                  styles.option,
                  isChoice && styles.optionPicked,
                  checked && isAnswer && styles.optionRight,
                  checked && isChoice && !isAnswer && styles.optionWrong,
                )}>
                <input
                  type="radio"
                  name={id}
                  checked={isChoice}
                  disabled={checked}
                  onChange={() => setPicked(i)}
                />
                <span className={styles.optionText}>
                  <Inline>{opt}</Inline>
                </span>
                {checked && isAnswer && <span className={styles.mark}>✓ Correct answer</span>}
                {checked && isChoice && !isAnswer && <span className={styles.mark}>✗ Your answer</span>}
              </label>
            </li>
          );
        })}
      </ul>
      {!checked ? (
        <button className="button button--primary" disabled={picked === null} onClick={check}>
          Check answer
        </button>
      ) : (
        <>
          <div className={clsx(styles.verdict, correct ? styles.verdictRight : styles.verdictWrong)}>
            {correct ? '✓ Correct!' : '✗ Not quite.'}
          </div>
          {(explanation || children) && (
            <div className={styles.explanation}>
              {explanation && (
                <p>
                  <Inline>{explanation}</Inline>
                </p>
              )}
              {children}
            </div>
          )}
          <button className="button button--secondary button--sm" onClick={retry}>
            Try again
          </button>
        </>
      )}
    </div>
  );
}

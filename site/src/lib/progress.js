// All learner state lives in one localStorage entry so it is easy to export,
// import and reset. Nothing is ever sent to a server.
import {useSyncExternalStore} from 'react';

const KEY = 'ml-course-progress:v1';
const EMPTY = {answers: {}, completed: {}};

let cache = null;
const listeners = new Set();

function read() {
  if (cache) return cache;
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    cache = {...EMPTY, ...parsed};
  } catch {
    cache = EMPTY;
  }
  return cache;
}

function write(next) {
  cache = next;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Storage blocked (private mode etc.): keep working in memory.
  }
  listeners.forEach((l) => l());
}

function subscribe(listener) {
  listeners.add(listener);
  const onStorage = (e) => {
    if (e.key === KEY) {
      cache = null;
      listener();
    }
  };
  window.addEventListener('storage', onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener('storage', onStorage);
  };
}

export function useProgress() {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

export function saveAnswer(id, answer) {
  const state = read();
  write({
    ...state,
    answers: {...state.answers, [id]: {...answer, at: new Date().toISOString()}},
  });
}

export function clearAnswer(id) {
  const state = read();
  const answers = {...state.answers};
  delete answers[id];
  write({...state, answers});
}

export function setCompleted(docId, done) {
  const state = read();
  const completed = {...state.completed};
  if (done) completed[docId] = new Date().toISOString();
  else delete completed[docId];
  write({...state, completed});
}

export function exportProgress() {
  return JSON.stringify(read(), null, 2);
}

export function importProgress(json) {
  const parsed = JSON.parse(json);
  if (typeof parsed !== 'object' || !parsed.answers || !parsed.completed) {
    throw new Error('That file does not look like a progress export.');
  }
  write({...EMPTY, ...parsed});
}

export function resetProgress() {
  write(EMPTY);
}

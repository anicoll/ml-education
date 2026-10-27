import React from 'react';
import clsx from 'clsx';
import Link from '@theme-original/DocSidebarItem/Link';
import {useProgress} from '@site/src/lib/progress';

// Adds a tick next to lessons the learner has marked complete.
export default function LinkWrapper(props) {
  const {completed} = useProgress();
  const done = props.item.docId && completed[props.item.docId];
  return <Link {...props} item={{...props.item, className: clsx(props.item.className, done && 'lesson-done')}} />;
}

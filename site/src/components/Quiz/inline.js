import React from 'react';

// Renders `backticks` as inline code so options and questions passed as plain
// strings can still show code. Non-strings (JSX) pass straight through.
export default function Inline({children}) {
  if (typeof children !== 'string') return children;
  return children.split(/(`[^`]+`)/g).map((part, i) =>
    part.startsWith('`') && part.endsWith('`') && part.length > 1 ? (
      <code key={i}>{part.slice(1, -1)}</code>
    ) : (
      <React.Fragment key={i}>{part}</React.Fragment>
    ),
  );
}

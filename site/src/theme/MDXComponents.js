import MDXComponents from '@theme-original/MDXComponents';
import {MultipleChoice, OpenQuestion} from '@site/src/components/Quiz';
import YouTube from '@site/src/components/Video';

// Makes the quiz and video components available in every doc without an import.
export default {
  ...MDXComponents,
  MultipleChoice,
  OpenQuestion,
  YouTube,
};

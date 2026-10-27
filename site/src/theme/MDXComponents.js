import MDXComponents from '@theme-original/MDXComponents';
import {MultipleChoice, OpenQuestion} from '@site/src/components/Quiz';

// Makes the quiz components available in every doc without an import.
export default {
  ...MDXComponents,
  MultipleChoice,
  OpenQuestion,
};

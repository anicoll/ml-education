import React from 'react';
import Footer from '@theme-original/DocItem/Footer';
import LessonProgress from '@site/src/components/LessonProgress';

export default function FooterWrapper(props) {
  return (
    <>
      <LessonProgress />
      <Footer {...props} />
    </>
  );
}

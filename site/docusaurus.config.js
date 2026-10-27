// @ts-check
// Local-only learning site. Run with `npm start` from this folder.

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ML From First Principles',
  tagline: 'What machine learning is, how models are trained from scratch, and how to fine-tune open-source models on your own GPU.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // Not hosted anywhere; these just need to be valid.
  url: 'http://localhost',
  baseUrl: '/',

  onBrokenLinks: 'throw',

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        },
      },
      navbar: {
        title: 'ML From First Principles',
        logo: {
          alt: 'Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'curriculum',
            position: 'left',
            label: 'Curriculum',
          },
          {to: '/docs/hardware/overview', label: 'Hardware', position: 'left'},
          {to: '/docs/reference/glossary', label: 'Glossary', position: 'left'},
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Start here',
            items: [
              {label: 'How to use this site', to: '/docs/intro'},
              {label: 'Learning roadmap', to: '/docs/roadmap'},
            ],
          },
          {
            title: 'Reference',
            items: [
              {label: 'Glossary', to: '/docs/reference/glossary'},
              {label: 'Further resources', to: '/docs/reference/resources'},
            ],
          },
        ],
        copyright: `Personal learning notes. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'powershell', 'json', 'yaml'],
      },
    }),
};

export default config;

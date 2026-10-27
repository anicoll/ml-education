import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const parts = [
  {
    title: '0. Start here',
    to: '/docs/intro',
    text: 'How the site is organised, how to study it, and the recommended path.',
  },
  {
    title: '1. Foundations',
    to: '/docs/foundations/what-is-ml',
    text: 'What ML really is: learning a function from data. The model, loss and optimiser loop, and the maths you actually need.',
  },
  {
    title: '2. Classical ML',
    to: '/docs/classical-ml/linear-regression',
    text: 'Linear and logistic regression by hand, decision trees, overfitting and honest evaluation.',
  },
  {
    title: '3. Neural networks',
    to: '/docs/neural-networks/the-neuron',
    text: 'Neurons, activations, backpropagation and gradient descent, then the same thing in PyTorch.',
  },
  {
    title: '4. Architectures',
    to: '/docs/architectures/cnns',
    text: 'CNNs, embeddings, attention and the transformer, explained piece by piece.',
  },
  {
    title: '5. Training from scratch',
    to: '/docs/training-from-scratch/the-training-loop',
    text: 'Data pipelines, training loops, debugging, and a capstone: train your own tiny GPT.',
  },
  {
    title: '6. Open-source models',
    to: '/docs/open-source-models/landscape',
    text: 'Hugging Face, running models locally, quantisation, and fine-tuning with LoRA and QLoRA.',
  },
  {
    title: '7. Your hardware',
    to: '/docs/hardware/overview',
    text: 'RTX 3070 vs Apple Silicon, what fits in 8 GB vs 48 GB, setup for Windows/WSL2, Ubuntu and macOS, and when to rent.',
  },
];

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Home" description={siteConfig.tagline}>
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <Heading as="h1" className="hero__title">
            {siteConfig.title}
          </Heading>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link className="button button--secondary button--lg" to="/docs/intro">
              Start learning
            </Link>
            <Link className="button button--outline button--secondary button--lg" to="/docs/roadmap">
              See the roadmap
            </Link>
          </div>
        </div>
      </header>
      <main className={clsx('container', styles.parts)}>
        <div className="row">
          {parts.map((p) => (
            <div key={p.title} className={clsx('col col--6', styles.col)}>
              <Link to={p.to} className={styles.card}>
                <Heading as="h3">{p.title}</Heading>
                <p>{p.text}</p>
              </Link>
            </div>
          ))}
        </div>
      </main>
    </Layout>
  );
}

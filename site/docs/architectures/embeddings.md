---
sidebar_position: 2
title: Tokens and embeddings
---

# Tokens and embeddings

Before a language model can do anything with text, the text has to become numbers. This happens in two steps: **tokenisation** and **embedding**.

## Tokenisation

A **tokeniser** splits text into pieces called tokens and maps each to an integer ID.

- **Character-level:** each character is a token. Tiny vocabulary, but sequences get very long.
- **Word-level:** each word is a token. Short sequences, but huge vocabularies and no way to handle new words.
- **Subword (BPE, WordPiece, SentencePiece):** the standard compromise. Common words are single tokens; rare words are split into pieces.

```text
"Tokenisation is unbelievably useful"
→ ["Token", "isation", " is", " un", "believ", "ably", " useful"]
→ [ 9032,   7247,      374,   653,   41837,   2915,    5505 ]
```

**Byte-pair encoding (BPE)** builds its vocabulary by starting from individual bytes and repeatedly merging the most frequent adjacent pair into a new token, until the vocabulary reaches a target size (often 32k to 200k tokens).

Tokenisation explains a lot of odd LLM behaviour: why models struggle to count letters in a word (they never see the letters), why some languages cost more tokens than others, and why arithmetic on long numbers is hard.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello, world!")
print(ids, [tok.decode([i]) for i in ids])
```

## Embeddings

A token ID is just an index; the number 9032 means nothing on its own. An **embedding layer** is a lookup table: a matrix with one row per token, each row a learned vector of, say, 768 numbers.

```python
emb = nn.Embedding(num_embeddings=50_000, embedding_dim=768)
vectors = emb(torch.tensor([9032, 7247]))   # shape (2, 768)
```

These vectors start random and are learned by backpropagation like any other weights. After training, tokens used in similar contexts end up with similar vectors. Directions in this space can carry meaning: the classic example from word2vec is

$$
\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})
$$

## Embeddings beyond tokens

The idea generalises: an embedding is a learned vector representation of anything. Sentence embeddings, image embeddings, user embeddings for recommendations. **Similarity search** over embeddings (using cosine similarity) powers semantic search and RAG (retrieval-augmented generation), where an LLM looks up relevant documents before answering.

$$
\text{cosine}(a, b) = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert}
$$

## Position

An embedding table gives the same vector for "dog" wherever it appears. But order matters: "dog bites man" vs "man bites dog". Transformers add **positional information**, either as learned position embeddings added to each token vector (GPT-2), or by rotating vectors according to their position (**RoPE**, used by Llama and most modern models).

:::tip Try it
1. Use the GPT-2 tokeniser to tokenise a paragraph of English, some code, and a sentence in another language. Compare token counts per character.
2. Load a sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`), embed 20 sentences on mixed topics, and find the most similar pair for each sentence. This model is tiny and runs fine on your GPU or even CPU.
:::

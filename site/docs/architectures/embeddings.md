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

:::tip[Try it]
1. Use the GPT-2 tokeniser to tokenise a paragraph of English, some code, and a sentence in another language. Compare token counts per character.
2. Load a sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`), embed 20 sentences on mixed topics, and find the most similar pair for each sentence. This model is tiny and runs fine on your GPU or even CPU.
:::

## Watch

Optional videos that explain this lesson's ideas another way. Click a thumbnail to play it here, or the title to open it on YouTube.

<YouTube id="viZrOnJclY0" title="Word Embedding and Word2Vec, Clearly Explained!!!" channel="StatQuest with Josh Starmer">

How a network learns a vector for each word, and why similar words end up close together.

</YouTube>

<YouTube id="TCH_1BHY58I" title="Building makemore Part 2: MLP" channel="Andrej Karpathy">

Builds a small model with a learned embedding table from scratch and visualises the embeddings it learns.

</YouTube>

## Check your understanding

<MultipleChoice
  id="emb-1"
  question="Why do LLMs often struggle to count the letters in a word like 'strawberry'?"
  options={[
    "They have no memory",
    "They see tokens (chunks of characters), not individual letters",
    "The word is not in their training data",
    "Counting needs a GPU",
  ]}
  answer={1}
  explanation="With subword tokenisation, 'strawberry' might be two or three tokens. The model never sees the individual letters directly."
/>

<MultipleChoice
  id="emb-2"
  question="What is an embedding layer, mechanically?"
  options={[
    "A lookup table with one learned vector per token ID",
    "A tokeniser",
    "A softmax over the vocabulary",
    "A fixed, hand-designed encoding of each word",
  ]}
  answer={0}
  explanation="It's a matrix with one row per token. The rows start random and are learned by backpropagation like any other weights."
/>

<MultipleChoice
  id="emb-3"
  question="What does cosine similarity between two embedding vectors measure?"
  options={["Their length", "How closely they point in the same direction", "How many tokens they contain", "Their training loss"]}
  answer={1}
  explanation="1 means the same direction (very similar), 0 means unrelated, −1 means opposite. It's the basis of semantic search and RAG."
/>

<MultipleChoice
  id="emb-4"
  question="Without any position information, how would a transformer see 'dog bites man' and 'man bites dog'?"
  options={["As completely different", "As the same bag of tokens, since each token's embedding ignores where it is", "As an error", "It would reverse them automatically"]}
  answer={1}
  explanation="That's why transformers add position information, with learned position embeddings (GPT-2) or RoPE (Llama and most modern models)."
/>

<OpenQuestion
  id="emb-5"
  question="Explain how you could use sentence embeddings to build a 'search my notes by meaning' tool. What are the steps?">

A worked design:

1. **Split** your notes into chunks (for example, a paragraph each).
2. **Embed** every chunk once with a sentence-embedding model like `all-MiniLM-L6-v2`. Each chunk becomes a vector of 384 numbers. Save the vectors alongside the text.
3. **At search time**, embed the search query with the same model.
4. **Compare** the query vector to every chunk vector using cosine similarity.
5. **Return** the top few chunks with the highest similarity.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunks = ["Gradient descent steps downhill on the loss.",
          "Remember to buy milk.",
          "LoRA trains small adapter matrices."]
chunk_vecs = model.encode(chunks)

query_vec = model.encode("how do optimisers update weights?")
scores = util.cos_sim(query_vec, chunk_vecs)[0]
print(chunks[int(scores.argmax())])   # the gradient descent note
```

It finds the gradient descent note even though it shares no keywords with the query, because the embeddings capture meaning. Add an LLM that reads the top chunks and answers the question, and you have RAG.

</OpenQuestion>

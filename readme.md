# Intro

This package is just a little wrapper around [Sentence-BERT](https://www.sbert.net/).

# Installation

Install with `pip`:

```bash
pip install git+https://github.com/jrc03c/sentence-meaning-helper
```

Or install with `conda`:

```bash
conda install pip git
pip install git+https://github.com/jrc03c/sentence-meaning-helper
```

# Usage

A `SentenceMeaningHelper` instance requires a cache directory in which to work. The path to that directory must be passed into the constructor:

```python
from sentence_meaning_helper import SentenceMeaningHelper
cacheDir = "path/to/cache/folder"
helper = SentenceMeaningHelper(cacheDir)
```

# API

## Instance properties

**`cacheDir`**

The path to the cache directory.

**`model`**

The Sentence-BERT model; specifically a `SentenceTransformer` model from the Sentence-BERT `sentence_transformers` package.

## Instance methods

**`getSentenceEmbedding(sentence)`**

Returns the vector representing the embedding of the given sentence string.

**`getSimilarity(sentence1, sentence2)`**

Returns the cosine similarity between the given sentences.

**`getSimilarities(sentences, progress=lambda p: p)`**

Returns a pandas `DataFrame` containing the similarities of every sentence to every other sentence. The columns of the returned `DataFrame` are "Sentence 1", "Sentence 2", and "Cosine similarity". Optionally, a `progress` function can be passed as a way of monitoring the completion percentage of the function (since it can sometimes take a long time with long lists of sentences).

**`getNMostSimilarSentencesToTarget(targetSentence, otherSentences, n, shouldOnlyConsiderSimilarityMagnitude=False, progress=lambda p: p)`**

Given a target sentence, returns the _n_ most similar sentences from the `otherSentences` list. The `shouldOnlyConsiderSimilarityMagnitude` parameter determines whether cosine similarities are compared as values in the range [-1, 1] or absolute values in the range [0, 1]. Optionally, a `progress` function can be passed as a way of monitoring the completion percentage of the function (since it can sometimes take a long time with long lists of sentences).

**`clearCache()`**

Clears out the cache directory (which just holds sentence embeddings for quick recall).

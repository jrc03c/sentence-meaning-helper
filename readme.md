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

A `SentenceMeaningHelper` instance requires a cache directory in which to work. The path to that directory must be passed into the constructor. Optionally, a model name can be passed as well.

```python
from sentence_meaning_helper import SentenceMeaningHelper
cache_dir = "path/to/cache/folder"
model = "Salesforce/SFR-Embedding-Mistral"
helper = SentenceMeaningHelper(cache_dir, model=model)
```

# API

## Instance properties

**`cache_dir`**

The path to the cache directory.

**`model`**

The Sentence-BERT model; specifically a `SentenceTransformer` model from the Sentence-BERT `sentence_transformers` package.

## Instance methods

**`get_embedding(sentence)`**

Returns the vector representing the embedding of the given sentence string.

**`get_similarity(sentence1, sentence2)`**

Returns the cosine similarity between the given sentences.

**`get_similarities(sentences, progress=lambda p: p)`**

Returns a pandas `DataFrame` containing the similarities of every sentence to every other sentence. The columns of the returned `DataFrame` are "Sentence 1", "Sentence 2", and "Cosine similarity". Optionally, a `progress` function can be passed as a way of monitoring the completion percentage of the function (since it can sometimes take a long time with long lists of sentences).

**`get_n_most_similar_sentences_to_target(target_sentence, other_sentences, n, should_only_consider_similarity_magnitude=False, progress=lambda p: p)`**

Given a target sentence, returns the _n_ most similar sentences from the `other_sentences` list. The `should_only_consider_similarity_magnitude` parameter determines whether cosine similarities are compared as values in the range [-1, 1] or absolute values in the range [0, 1]. Optionally, a `progress` function can be passed as a way of monitoring the completion percentage of the function (since it can sometimes take a long time with long lists of sentences).

**`clear_cache()`**

Clears out the cache directory (which just holds sentence embeddings for quick recall).

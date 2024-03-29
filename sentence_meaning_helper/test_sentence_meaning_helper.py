import os
import shutil
from unittest import TestCase

from numpy import ndarray
from numpy.random import random
from pandas import DataFrame
from pyds import isEqual

from sentence_meaning_helper import SentenceMeaningHelper

SMALLEST_MODEL = "TaylorAI/bge-micro"


def random_string(n):
    alpha = "abcdef1234567890"
    out = ""

    for i in range(0, n):
        out += alpha[int(random() * len(alpha))]

    return out


class SentenceMeaningHelperTestCase(TestCase):
    def test_cache_dir(self):
        cache_dir = random_string(8)
        helper = SentenceMeaningHelper(cache_dir)
        self.assertTrue(cache_dir == helper.cache_dir)
        self.assertTrue(os.path.exists(cache_dir))
        shutil.rmtree(cache_dir)

    def test_model(self):
        cache_dir = random_string(8)
        model = SMALLEST_MODEL
        helper = SentenceMeaningHelper(cache_dir, model=model)
        self.assertTrue(model == helper.model.tokenizer.name_or_path)
        shutil.rmtree(cache_dir)

    def test_get_sentence_embedding(self):
        cache_dir = random_string(8)
        model = SMALLEST_MODEL
        sentence = "It was the best of times; it was the worst of times."
        helper = SentenceMeaningHelper(cache_dir, model=model)
        v1 = helper.get_sentence_embedding(sentence)
        v2 = helper.get_sentence_embedding(sentence)
        v3 = helper.get_sentence_embedding("Here's a different sentence!")
        self.assertTrue(type(v1) == ndarray)
        self.assertTrue(type(v2) == ndarray)
        self.assertTrue(isEqual(v1, v2))
        self.assertFalse(isEqual(v1, v3))
        shutil.rmtree(cache_dir)

    def test_get_similarity(self):
        cache_dir = random_string(8)
        model = SMALLEST_MODEL
        s1 = "From Italy they visited Germany and France."
        s2 = "God raises my weakness and gives me courage to endure the worst."
        helper = SentenceMeaningHelper(cache_dir, model=model)
        self.assertGreater(helper.get_similarity(s1, s1), 0.99)
        self.assertLess(helper.get_similarity(s1, s2), 0.30)
        shutil.rmtree(cache_dir)

    def test_get_similarities(self):
        cache_dir = random_string(8)
        model = SMALLEST_MODEL

        sentences = [
            "My favorite food is cake!",
            "My favorite food is pie!",
            "It was a dark and stormy night.",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        ]

        helper = SentenceMeaningHelper(cache_dir, model=model)
        similarities = helper.get_similarities(sentences)
        self.assertTrue(isinstance(similarities, DataFrame))
        self.assertTrue(similarities.shape[0] == 6)
        self.assertTrue(similarities.columns[0] == "Sentence 1")
        self.assertTrue(similarities.columns[1] == "Sentence 2")
        self.assertTrue(similarities.columns[2] == "Cosine similarity")

        for i in range(0, similarities.shape[0] - 1):
            self.assertGreater(
                similarities["Cosine similarity"].values[i],
                similarities["Cosine similarity"].values[i + 1],
            )

        shutil.rmtree(cache_dir)

    def test_get_n_most_similar_sentences_to_target(self):
        cache_dir = random_string(8)
        model = SMALLEST_MODEL
        helper = SentenceMeaningHelper(cache_dir, model=model)
        n = 3
        target = "My favorite food is pizza!"

        others = [
            "It was a dark and stormy night.",
            "My favorite food is macaroni and cheese!",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "It is a pleasure to burn.",
            "My favorite food is cake!",
            "Call me Ishmael.",
            "The story so far: in the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
            "My favorite food is pie!",
        ]

        results = helper.get_n_most_similar_sentences_to_target(target, others, n)
        self.assertTrue(isinstance(results, DataFrame))
        self.assertTrue(results.shape[0] == n)
        self.assertTrue(results.columns[0] == "Sentence")
        self.assertTrue(results.columns[1] == "Cosine similarity")
        most_similar_sentences = results["Sentence"].values.tolist()

        predicted = [
            "My favorite food is cake!",
            "My favorite food is macaroni and cheese!",
            "My favorite food is pie!",
        ]

        for sentence in predicted:
            self.assertTrue(sentence in most_similar_sentences)

        shutil.rmtree(cache_dir)

    def test_clear_cache(self):
        pass

import os
import shutil
from unittest import TestCase

from numpy import ndarray
from numpy.random import random
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
        sentence = "It was the best of times; it was the worst of times."
        helper = SentenceMeaningHelper(cache_dir, model=model)
        self.assertGreater(helper.get_similarity(sentence, sentence), 0.99)
        # self.assertLess(...)
        shutil.rmtree(cache_dir)

    def test_get_similarities(self):
        pass

    def test_get_n_most_similar_sentences_to_target(self):
        pass

    def test_clear_cache(self):
        pass

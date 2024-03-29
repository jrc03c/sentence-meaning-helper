import os
import shutil
from unittest import TestCase

from numpy.random import random

from sentence_meaning_helper import SentenceMeaningHelper


def random_string(n):
    alpha = "abcdef1234567890"
    out = ""

    for i in range(0, n):
        out += alpha[int(random() * len(alpha))]

    return out


class SentenceMeaningHelperTestCase(TestCase):
    def test_cache_dir(self):
        cache_dir = random_string(8)
        SentenceMeaningHelper(cache_dir)
        self.assertTrue(os.path.exists(cache_dir))
        shutil.rmtree(cache_dir)

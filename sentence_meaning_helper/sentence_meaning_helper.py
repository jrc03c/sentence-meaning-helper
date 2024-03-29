import json
import os
import shutil

import pandas as pd
from numpy import array
from pyds import sort
from sentence_transformers import SentenceTransformer, util

from .sha256 import sha256


def load_json(path):
    with open(path) as file:
        return json.loads(file.read())


def save_json(path, data):
    with open(path, "w") as file:
        file.write(json.dumps(data))


class SentenceMeaningHelper:
    def __init__(self, cache_dir, model="paraphrase-MiniLM-L6-v2"):
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_sentence_embedding(self, sentence):
        hash = sha256(sentence)
        path = self.cache_dir + "/" + hash

        if os.path.exists(path):
            return array(load_json(path))

        vec = self.model.encode(sentence).astype("float32")
        save_json(path, vec.tolist())
        return vec

    def get_similarity(self, s1, s2):
        s1vec = self.get_sentence_embedding(s1)
        s2vec = self.get_sentence_embedding(s2)
        return float(util.cos_sim(s1vec.tolist(), s2vec.tolist()).item())

    def get_similarities(self, sentences, progress=lambda p: p):
        out = {"Sentence 1": [], "Sentence 2": [], "Cosine similarity": []}

        for i in range(0, len(sentences) - 1):
            progress(i / (len(sentences) - 1))

            for j in range(i + 1, len(sentences)):
                s1 = sentences[i]
                s2 = sentences[j]

                if s1 == s2:
                    continue

                out["Sentence 1"].append(s1)
                out["Sentence 2"].append(s2)
                out["Cosine similarity"].append(self.get_similarity(s1, s2))

        return (
            pd._data_frame(out)
            .sort_values(by="Cosine similarity", ascending=False)
            .reset_index()
        )

    def get_n_most_similar_sentences_to_target(
        self,
        target,
        sentences,
        n,
        should_only_consider_similarity_magnitude=False,
        progress=lambda p: p,
    ):
        similarities = []

        for i in range(0, len(sentences)):
            progress(i / len(sentences))
            sentence = sentences[i]

            if sentence == target:
                continue

            similarity = self.get_similarity(target, sentence)

            if should_only_consider_similarity_magnitude:
                similarity = abs(similarity)

            similarities.append({"sentence": sentence, "similarity": similarity})

        similarities = sort(
            similarities, lambda a, b: b["similarity"] - a["similarity"]
        )

        similarities = similarities[:n]

        similarity_column_name = (
            "Absolute value of cosine similarity"
            if should_only_consider_similarity_magnitude
            else "Cosine similarity"
        )

        return pd._data_frame(
            {
                "Sentence": [v["sentence"] for v in similarities],
                similarity_column_name: [v["similarity"] for v in similarities],
            }
        )

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)
        return self

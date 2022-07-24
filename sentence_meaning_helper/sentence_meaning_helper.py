from .hmack import hmack
from pyds import loadJSON, saveJSON, sort
from sentence_transformers import SentenceTransformer, util
import os
import pandas as pd
import shutil


class SentenceMeaningHelper:
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        if not os.path.isdir(self.cacheDir):
            os.makedirs(self.cacheDir)

    def getSentenceEmbedding(self, sentence):
        macked = hmack(sentence)
        path = self.cacheDir + "/" + macked

        if os.path.isfile(path):
            return loadJSON(path)

        vec = self.model.encode(sentence).astype("float32")
        saveJSON(path, vec.tolist())
        return vec

    def getSimilarity(self, s1, s2):
        s1Vec = self.getSentenceEmbedding(s1)
        s2Vec = self.getSentenceEmbedding(s2)
        return float(util.cos_sim(s1Vec, s2Vec).item())

    def getSimilarities(self, sentences, progress=lambda p: p):
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
                out["Cosine similarity"].append(self.getSimilarity(s1, s2))

        return (
            pd.DataFrame(out)
            .sort_values(by="Cosine similarity", ascending=False)
            .reset_index()
        )

    def getNMostSimilarSentencesToTarget(
        self,
        target,
        sentences,
        n,
        shouldOnlyConsiderSimilarityMagnitude=False,
        progress=lambda p: p,
    ):
        similarities = []

        for i in range(0, len(sentences)):
            progress(i / len(sentences))
            sentence = sentences[i]

            if sentence == target:
                continue

            similarity = self.getSimilarity(target, sentence)

            if shouldOnlyConsiderSimilarityMagnitude:
                similarity = abs(similarity)

            similarities.append({"sentence": sentence, "similarity": similarity})

        similarities = sort(
            similarities, lambda a, b: b["similarity"] - a["similarity"]
        )

        similarities = similarities[:n]

        similarityColumnName = (
            "Absolute value of cosine similarity"
            if shouldOnlyConsiderSimilarityMagnitude
            else "Cosine similarity"
        )

        return pd.DataFrame(
            {
                "Sentence": [v["sentence"] for v in similarities],
                similarityColumnName: [v["similarity"] for v in similarities],
            }
        )

    def clearCache(self):
        shutil.rmtree(self.cacheDir)
        os.makedirs(self.cacheDir)
        return self

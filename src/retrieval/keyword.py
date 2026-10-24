import os

import bm25s

from scripts.index_corpus import read_passages

# built once from the corpus (~1 min), then loaded from disk
CACHE_DIR = ".cache/bm25"


class KeywordIndex:
    """
    BM25 over the same passages as Qdrant, run locally, so it costs no API
    calls. Catches exact words ("vets", "Section 8", "mediation") that
    embeddings blur.
    """

    def __init__(self, corpus: str = "govt"):
        path = f"{CACHE_DIR}/{corpus}"
        if os.path.exists(path):
            self._bm25 = bm25s.BM25.load(path, load_corpus=True)
            return

        passages = read_passages(f"dataset/corpus/{corpus}.jsonl")
        self._bm25 = bm25s.BM25()
        self._bm25.index(bm25s.tokenize([p["text"] for p in passages], stopwords="en"))
        # keep what search results need, same keys as the Qdrant payload
        self._bm25.corpus = [
            {"passage_id": p["id"], "title": p["title"], "text": p["text"]} for p in passages
        ]
        self._bm25.save(path, corpus=self._bm25.corpus)

    def search(self, query: str, k: int = 5) -> list[dict]:
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        docs, scores = self._bm25.retrieve(tokens, k=k, show_progress=False)
        return [{**doc, "score": float(s)} for doc, s in zip(docs[0], scores[0])]

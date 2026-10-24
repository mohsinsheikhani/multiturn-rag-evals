from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore
from src.retrieval.keyword import KeywordIndex

# how deep each query searches before reranking. gold passages for the
# failing turns sat at ranks 7 to 25 with full history, so 30 covers them
POOL_DEPTH = 30

# how many fused passages go to the reranker. on the 5 review turns, 20 beat
# 30, and a ~95 passage pool did worse than no BM25 at all
POOL_SIZE = 20

# standard RRF constant, keeps one list's rank 1 from dominating
RRF_K = 60


class Retriever:
    def __init__(self, collection: str = "mtrag_govt", hybrid: bool = False):
        self._embedder = Embedder()
        self._store = VectorStore()
        self._collection = collection
        # BM25 only feeds the rerank pool, plain search stays embeddings only
        self._keyword = KeywordIndex(collection.removeprefix("mtrag_")) if hybrid else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        hits = self._store.search(self._collection, self._embedder.embed_query(query), limit=k)
        return [
            {
                "passage_id": h.payload["passage_id"],
                "score": h.score,
                "title": h.payload["title"],
                "text": h.payload["text"],
            }
            for h in hits
        ]

    def pool(self, queries: list[str], depth: int = POOL_DEPTH, size: int = POOL_SIZE) -> list[dict]:
        # every query searches embeddings (and BM25 if hybrid), then the lists
        # are merged with reciprocal rank fusion: a passage near the top of
        # several lists beats one that is high in only one
        lists = []
        for query in queries:
            lists.append(self.search(query, k=depth))
            if self._keyword:
                lists.append(self._keyword.search(query, k=depth))

        scores, by_id = {}, {}
        for hits in lists:
            for rank, p in enumerate(hits):
                pid = p["passage_id"]
                scores[pid] = scores.get(pid, 0) + 1 / (RRF_K + rank + 1)
                by_id[pid] = p
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [by_id[pid] for pid in ranked[:size]]

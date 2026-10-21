import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.embedding.embedder import EMBEDDING_DIM

# fixed namespace so the same passage id always maps to the same point id,
# across processes and runs -- point ids can't be arbitrary strings in Qdrant.
_NAMESPACE = uuid.UUID("6f2c7c1e-2b8a-4c1a-9b3a-9a2f6e6d1a10")


def point_id(passage_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, passage_id))


class VectorStore:
    """
    One collection per MTRAG corpus (govt, cloud, ...). Payload keeps the
    original MTRAG passage id, since that's what gold contexts reference.
    """

    def __init__(self, url: str | None = None, api_key: str | None = None):
        self._client = QdrantClient(
            url=url or os.environ["QDRANT_CLUSTER_URL"],
            api_key=api_key or os.environ["QDRANT_API_KEY"],
            timeout=60,
            # 1536-float vectors as JSON over REST took ~11s per 128 points;
            # gRPC sends them as binary.
            prefer_grpc=True,
        )

    def ensure_collection(self, collection: str) -> None:
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, passages: list[dict], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=point_id(p["id"]),
                vector=vector,
                payload={
                    "passage_id": p["id"],
                    "title": p["title"],
                    "url": p["url"],
                    "text": p["text"],
                },
            )
            for p, vector in zip(passages, vectors)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def has_all(self, collection: str, passage_ids: list[str]) -> bool:
        ids = [point_id(pid) for pid in passage_ids]
        found = self._client.retrieve(collection, ids=ids, with_payload=False, with_vectors=False)
        return len(found) == len(ids)

    def count(self, collection: str) -> int:
        return self._client.count(collection_name=collection, exact=True).count

    def search(self, collection: str, query_vector: list[float], limit: int = 5):
        return self._client.query_points(
            collection_name=collection, query=query_vector, limit=limit
        ).points

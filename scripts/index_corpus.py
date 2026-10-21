"""
Index one MTRAG corpus into Qdrant.

The corpus is already chunked by MTRAG (512 tokens, 100 overlap), so each
jsonl row becomes exactly one point. No re-chunking here.

    uv run python -m scripts.index_corpus --corpus govt
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import batched

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.embedding.embedder import Embedder  # noqa: E402
from src.embedding.vector_store import VectorStore  # noqa: E402


def clean_text(text: str) -> str:
    # a few scraped govt pages carry huge runs of null bytes, zero-width
    # spaces or blank padding (one row is 187k chars for ~270 words), which
    # blows past the embedding model's 8192-token limit.
    text = re.sub("[\x00​﻿]", "", text)
    text = re.sub(r"[ \t\r\f\v\xa0]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def read_passages(path: str) -> list[dict]:
    with open(path) as f:
        passages = [json.loads(line) for line in f]
    for p in passages:
        p["text"] = clean_text(p["text"])
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="govt")
    parser.add_argument("--batch-size", type=int, default=128)
    # a 128 batch is ~70k tokens. 2 workers stays near the 1M TPM limit, 4 blows past it.
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="index only the first N rows (smoke test)")
    args = parser.parse_args()

    passages = read_passages(f"dataset/corpus/{args.corpus}.jsonl")[: args.limit]
    collection = f"mtrag_{args.corpus}"

    embedder = Embedder()
    store = VectorStore()
    store.ensure_collection(collection)

    def index_batch(batch: tuple[dict, ...], max_attempts: int = 4) -> None:
        # the network here drops both OpenAI and Qdrant calls now and then,
        # so retry the whole batch instead of special-casing each client's errors.
        for attempt in range(1, max_attempts + 1):
            try:
                # resume: don't pay to re-embed batches a previous run already stored
                if store.has_all(collection, [p["id"] for p in batch]):
                    return
                vectors = embedder.embed_documents([p["text"] for p in batch])
                store.upsert(collection, list(batch), vectors)
                return
            except Exception as e:
                if attempt == max_attempts:
                    raise
                tqdm.write(f"batch failed ({type(e).__name__}), retry {attempt}/{max_attempts - 1}")
                time.sleep(10 * attempt)

    # each upsert is mostly waiting on the network, so run a few at once.
    # point ids are deterministic, so re-running after a crash just overwrites.
    batches = list(batched(passages, args.batch_size))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(index_batch, batches), total=len(batches), desc=collection))

    print(f"{collection}: {store.count(collection)} points (expected {len(passages)})")


if __name__ == "__main__":
    main()

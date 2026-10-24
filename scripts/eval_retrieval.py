"""
Check 2 of Stage 2: grade retrieval on its own, in replay mode.

    uv run --no-sync python -m scripts.eval_retrieval --turns 10:4,38:5,72:6,101:5,101:7

Each turn is searched three ways (full history, last turn, rewrite), top 5
each, plus a fourth: the rewrite and full history searches (plus BM25 with
--hybrid) fused into one pool and reranked by an LLM down to 5. All are scored with Recall@5 and nDCG@5 against MTRAG's gold passages.
Unanswerable turns have no gold passages, so they're skipped.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client  # noqa: E402

from scripts.eval_rewrite import parse_turns  # noqa: E402
from scripts.run import gold_turn, load_conversation  # noqa: E402
from src.evals.retrieval_checks import (  # noqa: E402
    K,
    ndcg_at_k,
    recall_at_k,
    strategy_queries,
)
from src.rerank.reranker import Reranker  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.rewrite.rewriter import Rewriter  # noqa: E402


def eval_turn(rewriter, retriever, reranker, conv: int, turn: int) -> dict | None:
    messages = load_conversation(conv, "govt")
    user, gold = gold_turn(messages, turn)
    k = 2 * (turn - 1)
    history = [{"speaker": m["speaker"], "text": m["text"]} for m in messages[:k]]
    gold_ids = {c["document_id"] for c in gold.get("contexts") or []}

    print(f"\n=== conv {conv} turn {turn}  {user['enrichments'].get('Answerability')}")
    print(f"user:     {user['text']}")
    print(f"gold:     {len(gold_ids)} passages")
    if not gold_ids:
        print("skipped:  no gold passages")
        return None

    rewrite = rewriter.rewrite(history, user["text"])["rewritten_query"]
    queries = strategy_queries(history, user["text"], rewrite)
    results = {strategy: retriever.search(query) for strategy, query in queries.items()}

    # rewrite + full history fused pool, reranked by an LLM down to the top 5
    pool = retriever.pool([queries["rewrite"], queries["full_history"]])
    results["rerank"] = reranker.rerank(history, user["text"], rewrite, pool, k=K)
    queries["rerank"] = f"top {len(pool)} fused from rewrite + full history"

    scores = {}
    for strategy, query in queries.items():
        ids = [p["passage_id"] for p in results[strategy]]
        scores[strategy] = (recall_at_k(ids, gold_ids), ndcg_at_k(ids, gold_ids))
        marks = " ".join("HIT" if pid in gold_ids else "-" for pid in ids)
        shown = query if strategy != "full_history" else f"({len(query.split())} words)"
        print(f"\n  {strategy}: {shown}")
        print(f"  ranks 1-5: {marks}   recall={scores[strategy][0]:.2f} ndcg={scores[strategy][1]:.2f}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", required=True, help="comma separated conv:turn pairs")
    parser.add_argument("--prompt", default="mtrag", choices=["mtrag", "v2"], help="rewrite prompt")
    parser.add_argument("--rewrite-model", default="gpt-4o-mini")
    parser.add_argument("--rerank-model", default="gpt-4o-mini")
    parser.add_argument("--hybrid", action="store_true", help="add BM25 results to the rerank pool")
    args = parser.parse_args()

    rewriter = Rewriter(model=args.rewrite_model, prompt=args.prompt)
    retriever = Retriever(hybrid=args.hybrid)
    reranker = Reranker(model=args.rerank_model)
    results = [eval_turn(rewriter, retriever, reranker, c, t) for c, t in parse_turns(args.turns)]
    scored = [r for r in results if r]

    print(f"\n=== mean over {len(scored)} scored turns")
    for strategy in ("full_history", "last_turn", "rewrite", "rerank"):
        recall = sum(r[strategy][0] for r in scored) / len(scored)
        ndcg = sum(r[strategy][1] for r in scored) / len(scored)
        print(f"  {strategy:<13} recall@5={recall:.2f}  ndcg@5={ndcg:.2f}")

    get_client().flush()


if __name__ == "__main__":
    main()

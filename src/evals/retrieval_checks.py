import math


# the plan scores at 5, even though the bot only feeds 3 passages to the generator
K = 5


def recall_at_k(retrieved: list[str], gold: set[str], k: int = K) -> float:
    return len(set(retrieved[:k]) & gold) / len(gold)


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int = K) -> float:
    # MTRAG gold passages have no relevance grades, so relevance is 1 or 0.
    # a hit at rank 1 counts more than a hit at rank 5.
    dcg = sum(1 / math.log2(rank + 2) for rank, pid in enumerate(retrieved[:k]) if pid in gold)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
    return dcg / ideal


def strategy_queries(history: list[dict], user_message: str, rewrite: str) -> dict[str, str]:
    # the three things MTRAG compared sending to the retriever
    full = "\n".join([t["text"] for t in history] + [user_message])
    return {"full_history": full, "last_turn": user_message, "rewrite": rewrite}

"""
Lab 2 of Stage 2: all three per-turn checks on every turn of a few
conversations, in replay mode, then one table split into slices.

    uv run --no-sync python -m scripts.eval_lab2 --convs 72,92,44,47,108 --prompt v2 --rewrite-model gpt-4o --hybrid
    uv run --no-sync python -m scripts.eval_lab2 --table    # reprint the table, no API calls

One rewrite per turn feeds every check, so the checks grade the same output.
The bot is run piece by piece instead of through the graph: the reranker
picks 5 so Recall@5 can be scored, and the generator gets the first 3, the
same as the graph's top_k.

Per-turn results go to results/lab2.jsonl, so the table can be rebuilt
without paying for the run again.
"""

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client, propagate_attributes  # noqa: E402

from scripts.eval_rewrite import load_labels  # noqa: E402
from scripts.run import gold_turn, load_conversation  # noqa: E402
from src.evals.answer_checks import EXPECTED_IDK, AnswerJudge, idk_score  # noqa: E402
from src.evals.retrieval_checks import K, ndcg_at_k, recall_at_k, strategy_queries  # noqa: E402
from src.evals.rewrite_checks import RewriteJudge, class_correct  # noqa: E402
from src.generation.generator import Generator  # noqa: E402
from src.rerank.reranker import Reranker  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.rewrite.rewriter import Rewriter  # noqa: E402

RESULTS_PATH = "results/lab2.jsonl"
GENERATOR_TOP_K = 3
STRATEGIES = ("rewrite", "rerank")


class Bot:
    def __init__(self, prompt: str, rewrite_model: str, hybrid: bool, generate_prompt: str):
        self.rewriter = Rewriter(model=rewrite_model, prompt=prompt)
        self.retriever = Retriever(hybrid=hybrid)
        self.reranker = Reranker()
        self.generator = Generator(prompt=generate_prompt)
        self.rewrite_judge = RewriteJudge()
        self.answer_judge = AnswerJudge()


def eval_turn(bot: Bot, labels, conv: int, turn: int) -> dict:
    messages = load_conversation(conv, "govt")
    user, gold = gold_turn(messages, turn)
    k = 2 * (turn - 1)
    history = [{"speaker": m["speaker"], "text": m["text"]} for m in messages[:k]]
    gold_ids = {c["document_id"] for c in gold.get("contexts") or []}
    answerability = user["enrichments"]["Answerability"][0]
    # turn 1 has no history, so it is standalone by definition and has no label
    gold_class = labels.get((conv, turn), "standalone")

    row = {
        "conv": conv,
        "turn": turn,
        "user": user["text"],
        "answerability": answerability,
        "gold_class": gold_class,
    }

    # check 1: the rewrite
    out = bot.rewriter.rewrite(history, user["text"])
    rewrite = out["rewritten_query"]
    row["rewrite"] = rewrite
    row["bot_class"] = out["rewrite_class"]
    row["class_ok"] = class_correct(out["rewrite_class"], gold_class)
    if row["class_ok"] and gold_class != "standalone":
        entity = bot.rewrite_judge.entity_resolved(history, user["text"], rewrite)
        added = bot.rewrite_judge.nothing_added(history, user["text"], rewrite)
        row["entity"] = entity
        row["added"] = added
        row["rewrite_ok"] = entity["verdict"] == "pass" and added["verdict"] == "pass"
    else:
        row["rewrite_ok"] = row["class_ok"]

    # check 2: retrieval. unanswerable turns have no gold passages, so no score
    queries = strategy_queries(history, user["text"], rewrite)
    results = {"rewrite": bot.retriever.search(rewrite, k=K)}
    pool = bot.retriever.pool([queries["rewrite"], queries["full_history"]])
    results["rerank"] = bot.reranker.rerank(history, user["text"], rewrite, pool, k=K)
    row["retrieved"] = {s: [p["passage_id"] for p in r] for s, r in results.items()}
    row["gold_ids"] = sorted(gold_ids)
    if gold_ids:
        row["recall"] = {s: recall_at_k(ids, gold_ids) for s, ids in row["retrieved"].items()}
        row["ndcg"] = {s: ndcg_at_k(ids, gold_ids) for s, ids in row["retrieved"].items()}

    # check 3: the answer
    passages = results["rerank"][:GENERATOR_TOP_K]
    answer = bot.generator.generate(history, user["text"], passages)
    row["answer"] = answer
    row["gold_answer"] = gold["text"]
    idk = bot.answer_judge.idk(history, user["text"], answer)
    row["idk"] = idk
    row["idk_ok"] = idk_score(idk["idk"], answerability)
    if idk["idk"] != "full":
        texts = [p["text"] for p in passages]
        row["faithfulness"] = bot.answer_judge.faithfulness(history, user["text"], answer, texts)

    return row


def print_row(row: dict) -> None:
    print(f"\n=== conv {row['conv']} turn {row['turn']}  {row['answerability']}  {row['gold_class']}")
    print(f"user:     {row['user']}")
    print(f"rewrite:  [{row['bot_class']}] {row['rewrite']}")
    print(f"class:    {'PASS' if row['class_ok'] else 'FAIL'}   rewrite: {'PASS' if row['rewrite_ok'] else 'FAIL'}")
    if "recall" in row:
        print(f"recall@5: " + "  ".join(f"{s}={row['recall'][s]:.2f}" for s in STRATEGIES))
    else:
        print("recall@5: skipped, no gold passages")
    print(f"idk:      {row['idk']['idk']} (expected {EXPECTED_IDK[row['answerability']]})")
    faith = row.get("faithfulness")
    print(f"faithful: {faith['score'] if faith else 'skipped (full IDK)'}")


def mean(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{sum(values) / len(values):.2f}" if values else "-"


def print_table(rows: list[dict]) -> None:
    slices = {
        "all": rows,
        "turn 1": [r for r in rows if r["turn"] == 1],
        "later turns": [r for r in rows if r["turn"] > 1],
        "standalone": [r for r in rows if r["gold_class"] == "standalone"],
        "not standalone": [r for r in rows if r["gold_class"] != "standalone"],
        "answerable": [r for r in rows if r["answerability"] == "ANSWERABLE"],
        "partial": [r for r in rows if r["answerability"] == "PARTIAL"],
        "unanswerable": [r for r in rows if r["answerability"] == "UNANSWERABLE"],
    }
    header = (
        f"{'slice':<15}{'turns':>6}{'class':>7}{'rewrite':>9}"
        f"{'R@5 rw':>8}{'R@5 rr':>8}{'nDCG rr':>9}"
        f"{'IDK':>6}{'faith':>7}"
    )
    print("\n" + header)
    for name, group in slices.items():
        scored = [r for r in group if "recall" in r]
        faith = [r["faithfulness"]["score"] for r in group if r.get("faithfulness")]
        print(
            f"{name:<15}{len(group):>6}"
            f"{mean([r['class_ok'] for r in group]):>7}"
            f"{mean([r['rewrite_ok'] for r in group]):>9}"
            + "".join(f"{mean([r['recall'][s] for r in scored]):>{w}}"
                      for s, w in zip(STRATEGIES, (8, 8)))
            + f"{mean([r['ndcg']['rerank'] for r in scored]):>9}"
            f"{mean([r['idk_ok'] for r in group]):>6}"
            f"{mean(faith):>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convs", help="comma separated conversation indexes")
    parser.add_argument("--prompt", default="v2", choices=["mtrag", "v2"], help="rewrite prompt")
    parser.add_argument("--rewrite-model", default="gpt-4o")
    parser.add_argument("--generate-prompt", default="v2", choices=["mtrag", "v2"])
    parser.add_argument("--hybrid", action="store_true", help="add BM25 results to the rerank pool")
    parser.add_argument("--table", action="store_true", help="only reprint the table from saved results")
    args = parser.parse_args()

    if args.table:
        with open(RESULTS_PATH) as f:
            print_table([json.loads(line) for line in f])
        return

    labels = load_labels()
    bot = Bot(args.prompt, args.rewrite_model, args.hybrid, args.generate_prompt)
    langfuse = get_client()
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    rows = []
    with open(RESULTS_PATH, "w") as f:
        for conv in (int(c) for c in args.convs.split(",")):
            n_turns = len(load_conversation(conv, "govt")) // 2
            for turn in range(1, n_turns + 1):
                # one trace per turn, grouped by conversation like the bot's own runs
                with langfuse.start_as_current_observation(name="lab2-turn"):
                    with propagate_attributes(
                        session_id=f"conv-{conv}", trace_name=f"lab2-turn-{turn}", tags=["lab2"]
                    ):
                        row = eval_turn(bot, labels, conv, turn)
                print_row(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)

    print_table(rows)
    langfuse.flush()


if __name__ == "__main__":
    main()

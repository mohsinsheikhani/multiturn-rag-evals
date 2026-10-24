"""
Check 3 of Stage 2: grade the answer, in replay mode.

    uv run --no-sync python -m scripts.eval_answer --turns 10:4,38:5,72:6,101:5,101:7

Per turn, the whole bot runs (rewrite, retrieve, generate) with gold history, then:
  1. IDK judge (gpt-4o): did the answer say it can't answer, fully, partly or not at all?
     Scored against MTRAG's answerability label.
  2. faithfulness (RAGAS, gpt-4o): share of the answer's statements supported by
     the passages the bot saw or the earlier turns. Skipped on a full IDK,
     since there is nothing to check.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client  # noqa: E402

from scripts.eval_rewrite import parse_turns  # noqa: E402
from scripts.run import gold_turn, load_conversation  # noqa: E402
from src.evals.answer_checks import EXPECTED_IDK, AnswerJudge, idk_score  # noqa: E402
from src.graph.runner import run_turn  # noqa: E402
from src.graph.workflow import build_graph  # noqa: E402


def eval_turn(graph, judge, conv: int, turn: int) -> None:
    messages = load_conversation(conv, "govt")
    user, gold = gold_turn(messages, turn)
    k = 2 * (turn - 1)
    history = [{"speaker": m["speaker"], "text": m["text"]} for m in messages[:k]]
    answerability = user["enrichments"]["Answerability"][0]

    state = run_turn(graph, f"conv-{conv}", turn, history, user["text"], mode="replay")
    answer = state["answer"]

    print(f"\n=== conv {conv} turn {turn}  {answerability}")
    print(f"user:     {user['text']}")
    print(f"answer:   {answer}")

    idk = judge.idk(history, user["text"], answer)
    expected = EXPECTED_IDK[answerability]
    score = idk_score(idk["idk"], answerability)
    print(f"idk:      {idk['idk']} (expected {expected})  score={score}")
    print(f"  quote: {idk.get('idk_quote', '')!r}")
    print(f"  why: {idk.get('reasoning', '')}")

    if idk["idk"] == "full":
        print("faithful: skipped (full IDK)")
        return

    passages = [p["text"] for p in state["passages"]]
    faith = judge.faithfulness(history, user["text"], answer, passages)
    print(f"faithful: {faith['score']}")
    for s in faith["statements"]:
        mark = "ok " if s["verdict"] else "UNSUPPORTED"
        print(f"  {mark} {s['statement']}")
        print(f"      why: {s['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", required=True, help="comma separated conv:turn pairs")
    parser.add_argument("--prompt", default="mtrag", choices=["mtrag", "v2"], help="rewrite prompt")
    parser.add_argument("--rewrite-model", default="gpt-4o-mini")
    parser.add_argument("--rerank", action="store_true", help="pool rewrite + full history, LLM rerank")
    parser.add_argument("--hybrid", action="store_true", help="add BM25 results to the rerank pool")
    parser.add_argument("--top-k", type=int, default=3, help="passages the generator sees")
    args = parser.parse_args()

    graph = build_graph(
        top_k=args.top_k,
        rewrite_prompt=args.prompt,
        rewrite_model=args.rewrite_model,
        rerank=args.rerank,
        hybrid=args.hybrid,
    )
    judge = AnswerJudge()
    for conv, turn in parse_turns(args.turns):
        eval_turn(graph, judge, conv, turn)

    get_client().flush()


if __name__ == "__main__":
    main()

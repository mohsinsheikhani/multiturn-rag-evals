"""
Check 1 of Stage 2: grade the rewrite step on its own, in replay mode.

    uv run --no-sync python -m scripts.eval_rewrite --turns 10:4,38:5,72:6,101:5,101:7

Each turn is conv:turn. History is gold, so only the rewrite is being tested.

Per turn, in order, stopping at the first failure:
  1. class grader (code): bot's class vs our label in standalone_labels.jsonl
  2. entity judge (gpt-4o): only if gold and bot both say non-standalone
  3. nothing-added judge (gpt-4o): same turns as the entity judge
"""

import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client  # noqa: E402

from scripts.run import gold_turn, load_conversation  # noqa: E402
from src.evals.rewrite_checks import RewriteJudge, class_correct  # noqa: E402
from src.rewrite.rewriter import Rewriter  # noqa: E402

LABELS_PATH = "dataset/labels/standalone_labels.jsonl"


def load_labels() -> dict[tuple[int, int], str]:
    with open(LABELS_PATH) as f:
        rows = [json.loads(line) for line in f]
    return {(r["conv"], r["turn"]): r["label"] for r in rows}


def parse_turns(spec: str) -> list[tuple[int, int]]:
    pairs = [item.split(":") for item in spec.split(",")]
    return [(int(conv), int(turn)) for conv, turn in pairs]


def eval_turn(rewriter, judge, labels, conv: int, turn: int) -> None:
    messages = load_conversation(conv, "govt")
    user, _ = gold_turn(messages, turn)
    k = 2 * (turn - 1)
    history = [{"speaker": m["speaker"], "text": m["text"]} for m in messages[:k]]

    out = rewriter.rewrite(history, user["text"])
    gold = labels[(conv, turn)]

    print(f"\n=== conv {conv} turn {turn}")
    print(f"user:     {user['text']}")
    print(f"rewrite:  {out['rewritten_query']}")

    ok = class_correct(out["rewrite_class"], gold)
    print(f"class:    {'PASS' if ok else 'FAIL'}  bot={out['rewrite_class']} gold={gold}")

    # a wrong class is already counted above, and a correct standalone has
    # nothing to resolve, so the judges only see correctly flagged rewrites
    if not ok or gold == "standalone":
        print("judges:   skipped")
        return

    entity = judge.entity_resolved(history, user["text"], out["rewritten_query"])
    print(f"entity:   {entity['verdict'].upper()}")
    for ref in entity.get("references", []):
        mark = "ok " if ref["resolved_in_rewrite"] else "MISSING"
        print(f"  {mark} {ref['phrase']!r} -> {ref['resolves_to']}")
    print(f"  why: {entity.get('reasoning', '')}")

    added = judge.nothing_added(history, user["text"], out["rewritten_query"])
    print(f"added:    {added['verdict'].upper()}")
    print(f"  added_terms={added.get('added_terms')} meaning_changed={added.get('meaning_changed')}")
    print(f"  why: {added.get('reasoning', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", required=True, help="comma separated conv:turn pairs")
    args = parser.parse_args()

    labels = load_labels()
    rewriter = Rewriter()
    judge = RewriteJudge()

    for conv, turn in parse_turns(args.turns):
        eval_turn(rewriter, judge, labels, conv, turn)

    get_client().flush()


if __name__ == "__main__":
    main()

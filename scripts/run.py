"""
Run an MTRAG conversation through the bot, in replay or live mode.

    uv run python -m scripts.run --conv 10 --turn 5    # replay one turn
    uv run python -m scripts.run --conv 10 --live      # live, whole conversation

--conv is the conversation's position in conversations.json. --turn is 1-based.

Replay: turns 1..turn-1 go in as gold history (user and agent), the bot
answers one turn. Tests a turn in isolation, as if everything before it was right.

Live: the gold user messages go in one by one, but the history is the bot's
own earlier answers. Tests how errors pile up across turns.
"""

import argparse
import json
import uuid

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client  # noqa: E402

from src.graph.runner import run_turn  # noqa: E402
from src.graph.workflow import build_graph  # noqa: E402

CONVERSATIONS_PATH = "dataset/conversation/conversations.json"


def load_conversation(conv: int, corpus: str) -> list[dict]:
    with open(CONVERSATIONS_PATH) as f:
        conversations = json.load(f)
    domain = conversations[conv]["domain"]
    if corpus not in domain:
        valid = [i for i, c in enumerate(conversations) if corpus in c["domain"]]
        raise SystemExit(f"conv {conv} is {domain}, not {corpus}. {corpus} convs: {valid}")
    return conversations[conv]["messages"]


def gold_turn(messages: list[dict], turn: int) -> tuple[dict, dict]:
    # messages strictly alternate user, agent, so turn k is messages[2k-2], [2k-1]
    n_turns = len(messages) // 2
    if not 1 <= turn <= n_turns:
        raise SystemExit(f"conversation has turns 1..{n_turns}, got {turn}")
    k = 2 * (turn - 1)
    return messages[k], messages[k + 1]


def print_turn(turn: int, user: dict, gold: dict, state: dict, full: bool) -> None:
    gold_ids = {c["document_id"] for c in gold.get("contexts") or []}
    hits = sum(p["passage_id"] in gold_ids for p in state["passages"])
    print(f"\n--- turn {turn} {user['enrichments'].get('Multi-Turn')} {user['enrichments'].get('Answerability')}")
    print(f"user:      {user['text']}")
    print(f"rewrite:   [{state['rewrite_class']}] {state['rewritten_query']}")
    print(f"retrieved: {hits}/{len(gold_ids)} gold")
    if full:
        for p in state["passages"]:
            mark = "HIT " if p["passage_id"] in gold_ids else "miss"
            print(f"  {mark} {p['score']:.3f} {p['passage_id']}  {p['title'].strip()[:60]}")
        print(f"answer:    {state['answer']}")
        print(f"gold:      {gold['text']}")
    else:
        print(f"answer:    {state['answer'][:200]}")


def replay(graph, messages: list[dict], conv: int, turn: int) -> str:
    user, gold = gold_turn(messages, turn)
    k = 2 * (turn - 1)
    history = [{"speaker": m["speaker"], "text": m["text"]} for m in messages[:k]]
    state = run_turn(graph, f"conv-{conv}", turn, history, user["text"], mode="replay")
    print_turn(turn, user, gold, state, full=True)
    return state["trace_id"]


def live(graph, messages: list[dict], conv: int) -> str:
    # one session per live run, so repeated runs of the same conversation
    # (needed later for pass^k) don't merge into one session
    session_id = f"conv-{conv}-live-{uuid.uuid4().hex[:8]}"
    history: list[dict] = []
    trace_id = ""
    for turn in range(1, len(messages) // 2 + 1):
        user, gold = gold_turn(messages, turn)
        state = run_turn(graph, session_id, turn, history, user["text"], mode="live")
        print_turn(turn, user, gold, state, full=False)
        # the bot's answer, not gold, becomes the next turn's history
        history += [
            {"speaker": "user", "text": user["text"]},
            {"speaker": "agent", "text": state["answer"]},
        ]
        trace_id = state["trace_id"]
    return trace_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--turn", type=int, help="replay this one turn with gold history")
    mode.add_argument("--live", action="store_true", help="run every turn with the bot's own history")
    parser.add_argument("--corpus", default="govt")
    args = parser.parse_args()

    messages = load_conversation(args.conv, args.corpus)
    graph = build_graph(collection=f"mtrag_{args.corpus}")

    if args.live:
        trace_id = live(graph, messages, args.conv)
    else:
        trace_id = replay(graph, messages, args.conv, args.turn)

    langfuse = get_client()
    langfuse.flush()
    print(f"\ntrace:     {langfuse.get_trace_url(trace_id=trace_id)}")


if __name__ == "__main__":
    main()

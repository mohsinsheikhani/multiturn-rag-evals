from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler


def run_turn(
    graph,
    conversation_id: str,
    turn: int,
    history: list[dict],
    user_message: str,
    mode: str,
) -> dict:
    """
    Runs one turn as one Langfuse trace. All turns of a conversation share
    session_id, so Langfuse groups them into one session.
    """
    langfuse = get_client()

    with langfuse.start_as_current_observation(
        name="turn", input={"user_message": user_message, "history": history}
    ) as span:
        # propagated to every span in the trace, so it's filterable in the UI
        with propagate_attributes(
            session_id=conversation_id,
            trace_name=f"turn-{turn}",
            tags=[mode],
            metadata={"conversation_id": conversation_id, "turn": str(turn), "mode": mode},
        ):
            # the handler adds one span per graph node (rewrite, retrieve, generate)
            state = graph.invoke(
                {"history": history, "user_message": user_message},
                config={"callbacks": [CallbackHandler()]},
            )

        span.update(
            output={"answer": state["answer"]},
            metadata={
                "rewrite_class": state["rewrite_class"],
                "rewritten_query": state["rewritten_query"],
                "passages": [
                    {"passage_id": p["passage_id"], "score": round(p["score"], 4)}
                    for p in state["passages"]
                ],
            },
        )

    return {**state, "trace_id": span.trace_id}

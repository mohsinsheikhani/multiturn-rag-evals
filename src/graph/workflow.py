from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.generation.generator import Generator
from src.rerank.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.rewrite.rewriter import Rewriter


class TurnState(TypedDict, total=False):
    # inputs. history is passed in, never stored by the graph, so replay
    # can feed gold turns and live can feed the bot's own answers.
    history: list[dict]  # [{"speaker": "user" | "agent", "text": ...}]
    user_message: str

    # rewrite node
    rewrite_class: str
    rewritten_query: str

    # retrieve node
    passages: list[dict]  # [{"passage_id", "score", "title", "text"}]

    # generate node
    answer: str


# 3 matches what MTRAG's own bot fed the generator (max_count in conversations.json)
def build_graph(
    collection: str = "mtrag_govt",
    top_k: int = 3,
    rewrite_prompt: str = "mtrag",
    rewrite_model: str = "gpt-4o-mini",
    rerank: bool = False,
    hybrid: bool = False,
):
    rewriter = Rewriter(model=rewrite_model, prompt=rewrite_prompt)
    retriever = Retriever(collection, hybrid=hybrid)
    reranker = Reranker()
    generator = Generator()

    def rewrite(state: TurnState) -> dict:
        return rewriter.rewrite(state["history"], state["user_message"])

    def retrieve(state: TurnState) -> dict:
        if not rerank:
            return {"passages": retriever.search(state["rewritten_query"], k=top_k)}

        # the rewrite alone missed gold passages that the full history found
        # deeper down, so both feed one fused pool and an LLM picks the top_k
        full_history = "\n".join([t["text"] for t in state["history"]] + [state["user_message"]])
        pool = retriever.pool([state["rewritten_query"], full_history])
        passages = reranker.rerank(
            state["history"], state["user_message"], state["rewritten_query"], pool, k=top_k
        )
        return {"passages": passages}

    def generate(state: TurnState) -> dict:
        answer = generator.generate(state["history"], state["user_message"], state["passages"])
        return {"answer": answer}

    graph = StateGraph(TurnState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

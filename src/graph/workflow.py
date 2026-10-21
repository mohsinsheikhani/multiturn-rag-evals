from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore
from src.generation.generator import Generator
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
def build_graph(collection: str = "mtrag_govt", top_k: int = 3):
    rewriter = Rewriter()
    embedder = Embedder()
    store = VectorStore()
    generator = Generator()

    def rewrite(state: TurnState) -> dict:
        return rewriter.rewrite(state["history"], state["user_message"])

    def retrieve(state: TurnState) -> dict:
        query_vector = embedder.embed_query(state["rewritten_query"])
        hits = store.search(collection, query_vector, limit=top_k)
        passages = [
            {
                "passage_id": hit.payload["passage_id"],
                "score": hit.score,
                "title": hit.payload["title"],
                "text": hit.payload["text"],
            }
            for hit in hits
        ]
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

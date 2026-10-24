# drop-in wrapper: logs each call as a Langfuse generation with model and tokens
from langfuse.openai import OpenAI
from pydantic import BaseModel

from src.rewrite.rewriter import format_conversation

RERANK_PROMPT = """You are ranking search results for a chatbot that answers questions from government documents.

Below is a conversation, the user's last message rewritten as a standalone question, and numbered passages.
Pick the passages that best help answer the user's last message, in the context of the conversation.
Return up to {k} passage numbers, most useful first.

CONVERSATION:
{conversation}

STANDALONE QUESTION:
{rewrite}

PASSAGES:
{passages}"""

# passages are cut so a pool of ~60 fits in one cheap call
MAX_WORDS = 200


class Ranking(BaseModel):
    passage_numbers: list[int]


class Reranker:
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = OpenAI()
        self._model = model

    def rerank(
        self, history: list[dict], user_message: str, rewrite: str, pool: list[dict], k: int
    ) -> list[dict]:
        block = "\n\n".join(
            f"[{i}] {p['title'].strip()}\n{' '.join(p['text'].split()[:MAX_WORDS])}"
            for i, p in enumerate(pool)
        )
        prompt = RERANK_PROMPT.format(
            k=k,
            conversation=format_conversation(history, user_message),
            rewrite=rewrite,
            passages=block,
        )
        response = self._client.chat.completions.parse(
            name="rerank-llm",
            model=self._model,
            temperature=0,
            response_format=Ranking,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # refusal: keep the embedding order rather than return nothing
            return pool[:k]

        # drop out-of-range and repeated numbers the model may return
        seen, ranked = set(), []
        for n in parsed.passage_numbers:
            if 0 <= n < len(pool) and n not in seen:
                seen.add(n)
                ranked.append(pool[n])
        return ranked[:k]

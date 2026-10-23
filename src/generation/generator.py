# drop-in wrapper: logs each call as a Langfuse generation with model and tokens
from langfuse.openai import OpenAI

# MTRAG Appendix D.2. The refusal string is what answerability checks look for.
REFUSAL = "I do not have specific information"

GENERATION_PROMPT = (
    "Given one or more documents and a user query, generate a response to the query using less "
    "than 150 words that is grounded in the provided documents. If no answer can be found in the "
    f'documents, say, "{REFUSAL}"'
)

SPEAKER_LABELS = {"user": "User", "agent": "Agent"}


class Generator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = OpenAI()
        self._model = model

    def generate(self, history: list[dict], user_message: str, passages: list[dict]) -> str:
        # generation sees the full conversation, not the rewritten query.
        # the rewrite is for the retriever only.
        passage_block = "\n\n".join(
            f"PASSAGE {i + 1}\n{p['text']}" for i, p in enumerate(passages)
        )
        turns = [f"{SPEAKER_LABELS[t['speaker']]}: {t['text']}" for t in history]
        turns.append(f"User: {user_message}")

        prompt = f"{GENERATION_PROMPT}\n\n{passage_block}\n\n" + "\n".join(turns)
        response = self._client.chat.completions.create(
            name="generate-llm",
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

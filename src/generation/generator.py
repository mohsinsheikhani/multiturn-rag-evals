# drop-in wrapper: logs each call as a Langfuse generation with model and tokens
from langfuse.openai import OpenAI

# MTRAG Appendix D.2. The refusal string is what answerability checks look for.
REFUSAL = "I do not have specific information"

GENERATION_PROMPT = (
    "Given one or more documents and a user query, generate a response to the query using less "
    "than 150 words that is grounded in the provided documents. If no answer can be found in the "
    f'documents, say, "{REFUSAL}"'
)

# our own version, from the Lab 2 partial slice: with MTRAG's prompt the bot
# refused outright on 6 of 7 partial failures, even when a gold passage was in
# front of it (47/4, 108/4, 108/7). MTRAG's partial gold answers give what the
# documents have and say what they don't. run 2 showed the rule has to be
# limited to what the user asked, or the bot adds "the documents do not
# cover..." to complete answers too (44/1, 47/2).
GENERATION_RULES = (
    " If the user asked for several things and the documents cover only some of them, answer those "
    "and say in one sentence which of the things the user asked for the documents do not cover. "
    "If the documents answer everything the user asked, do not add notes about what the documents "
    "leave out. Use the refusal alone only when nothing in the documents helps. "
    "If the user is reacting to an earlier answer rather than asking something new, reply to that "
    "using the documents and the conversation. Do not add facts that are not in the documents."
)

PROMPTS = {"mtrag": GENERATION_PROMPT, "v2": GENERATION_PROMPT + GENERATION_RULES}

SPEAKER_LABELS = {"user": "User", "agent": "Agent"}


class Generator:
    def __init__(self, model: str = "gpt-4o-mini", prompt: str = "mtrag"):
        self._client = OpenAI()
        self._model = model
        self._prompt = PROMPTS[prompt]

    def generate(self, history: list[dict], user_message: str, passages: list[dict]) -> str:
        # generation sees the full conversation, not the rewritten query.
        # the rewrite is for the retriever only.
        passage_block = "\n\n".join(
            f"PASSAGE {i + 1}\n{p['text']}" for i, p in enumerate(passages)
        )
        turns = [f"{SPEAKER_LABELS[t['speaker']]}: {t['text']}" for t in history]
        turns.append(f"User: {user_message}")

        prompt = f"{self._prompt}\n\n{passage_block}\n\n" + "\n".join(turns)
        response = self._client.chat.completions.create(
            name="generate-llm",
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

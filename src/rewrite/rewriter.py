from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

# MTRAG Appendix C.1 (Figure 5), kept verbatim so our rewrite numbers
# are comparable with the paper's.
REWRITE_PROMPT = (
    "Given the following conversation, please reword the final utterance from the user into a "
    "single utterance that does not need the history to understand the user's intent. Output in "
    'proper JSON format indicating the "class" (standalone or non-standalone) and the "reworded '
    'version" of the last utterance. Use this format: {{"class": "type of last utterance", '
    '"reworded version": "the last utterance rewritten into a standalone question, IF NEEDED"}}.\n\n'
    "In your rewording of the last utterance, do not do any unnecessary rephrasing or introduction "
    "of new terms or concepts that were not mentioned in the prior part of the conversation. Be "
    "minimal, by staying as close as possible to the shape and meaning of the last user utterance. "
    "If the last user utterance is already clear and standalone, the reworded version should be "
    "THE SAME as the last user utterance, and the class should be 'standalone'.\n\n"
    "{conversation} ASSISTANT:"
)

SPEAKER_LABELS = {"user": "USER", "agent": "ASSISTANT"}


class RewriteOutput(BaseModel):
    # aliases match the key names the prompt asks for, so the schema and
    # the prompt never disagree.
    model_config = ConfigDict(populate_by_name=True)

    rewrite_class: Literal["standalone", "non-standalone"] = Field(alias="class")
    rewritten_query: str = Field(alias="reworded version")


def format_conversation(history: list[dict], user_message: str) -> str:
    lines = [f"{SPEAKER_LABELS[turn['speaker']]}: {turn['text']}" for turn in history]
    lines.append(f"USER: {user_message}")
    return "\n".join(lines)


class Rewriter:
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = OpenAI()
        self._model = model

    def rewrite(self, history: list[dict], user_message: str) -> dict:
        prompt = REWRITE_PROMPT.format(conversation=format_conversation(history, user_message))
        response = self._client.chat.completions.parse(
            model=self._model,
            temperature=0,
            response_format=RewriteOutput,
            messages=[{"role": "user", "content": prompt}],
        )

        # the schema is enforced, so the only way to get nothing back is a
        # model refusal. keep it visible in evals and fall back to the raw message.
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return {"rewrite_class": "refused", "rewritten_query": user_message}
        return parsed.model_dump()

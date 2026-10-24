from typing import Literal

# drop-in wrapper: logs each call as a Langfuse generation with model and tokens
from langfuse.openai import OpenAI
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

# our own version: MTRAG's prompt plus one rule per failure found in the
# Stage 2 rewrite review (10/4, 38/5, 72/6). The MTRAG one stays the baseline.
REWRITE_RULES = (
    "Rules:\n"
    "1. A full sentence can still depend on the history. If a word in it, such as \"my expenses\", "
    "\"the forms\" or \"the fee\", only makes clear sense because of earlier turns, the class is "
    "'non-standalone'.\n"
    "2. Keep the question as open as the user left it. If the user only names a topic, such as "
    "\"How about mediation?\" or \"Discrimination\", do not narrow it to one aspect (process, cost, "
    "eligibility) the user did not ask about.\n"
    "3. If the user repeats an earlier question with a new subject, such as \"any advantage to a "
    "women owned company?\" followed by \"anything for vets?\", carry over the earlier question and "
    "its setting, and read the new subject the way the conversation uses it (here: veteran-owned "
    "companies, not veterans in general).\n\n"
)

PROMPTS = {
    "mtrag": REWRITE_PROMPT,
    "v2": REWRITE_PROMPT.replace("{conversation}", REWRITE_RULES + "{conversation}"),
}

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
    def __init__(self, model: str = "gpt-4o-mini", prompt: str = "mtrag"):
        self._client = OpenAI()
        self._model = model
        self._prompt = PROMPTS[prompt]

    def rewrite(self, history: list[dict], user_message: str) -> dict:
        # turn 1 has nothing to depend on, so it is standalone by definition.
        # sent to the model anyway, it sometimes answered the question instead (44/1, 47/1)
        if not history:
            return {"rewrite_class": "standalone", "rewritten_query": user_message}

        prompt = self._prompt.format(conversation=format_conversation(history, user_message))
        # reasoning models (gpt-5 and later, o-series) only accept the default temperature
        reasoning = self._model.startswith(("gpt-5", "gpt-6", "o"))
        response = self._client.chat.completions.parse(
            name="rewrite-llm",
            model=self._model,
            **({} if reasoning else {"temperature": 0}),
            response_format=RewriteOutput,
            messages=[{"role": "user", "content": prompt}],
        )

        # the schema is enforced, so the only way to get nothing back is a
        # model refusal. keep it visible in evals and fall back to the raw message.
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return {"rewrite_class": "refused", "rewritten_query": user_message}
        return parsed.model_dump()

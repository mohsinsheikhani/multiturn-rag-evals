from typing import Literal

# drop-in wrapper: logs each judge call as a Langfuse generation with model and tokens
from langfuse.openai import OpenAI
from pydantic import BaseModel

from src.rewrite.rewriter import format_conversation

# a stronger model than the bot (gpt-4o-mini), so the judge doesn't go easy
# on mistakes it would make itself
JUDGE_MODEL = "gpt-4o"


def class_correct(predicted: str, gold: str) -> bool:
    # code grader: plain match against our own standalone labels
    return predicted == gold


ENTITY_PROMPT = """You are grading a query rewriter in a multi-turn chat.
The rewriter takes the user's last message and rewrites it so it can be understood without the conversation history.

Your job: check that every reference in the last user message that depends on the history is resolved in the rewrite.

A reference depends on history when its meaning comes from earlier turns, for example:
- pronouns and pointers: "it", "that", "they", "the other one", "there", "this program"
- ellipsis: "what about veterans?", "and in Texas?" (what is being asked comes from earlier)
- bare topics: "Renters" when the topic (renters in California housing) was set earlier
- definite nouns that only make sense with history: "the forms", "the filing fee"

Steps:
1. List each such reference in the last user message, and what it refers to according to the history.
2. For each one, check whether the rewrite states it explicitly. Wording can differ, the meaning must be there.
3. verdict is "pass" only if every reference is resolved. If there are no references, verdict is "pass".

Only judge resolution. Do not judge whether the rewrite added extra things or changed the meaning, another check covers that.

CONVERSATION:
{conversation}

REWRITE:
{rewrite}"""


ADDED_PROMPT = """You are grading a query rewriter in a multi-turn chat.
The rewriter takes the user's last message and rewrites it so it can be understood without the conversation history.
It is told to be minimal: stay close to the user's wording and not introduce new terms or concepts.

Your job: check that the rewrite keeps the user's intent and adds nothing new.

Steps:
1. List terms or concepts in the rewrite that appear nowhere in the conversation, neither in the last user message nor in the history. Pulling a term in from the history is allowed, that is what a rewrite is for. Plain function words and harmless rephrasing ("can you tell me" vs "what is") do not count.
2. Decide whether the rewrite asks something different from what the user asked: narrower, broader, a different question, or a different entity.
3. verdict is "pass" only if added_terms is empty and meaning_changed is false.

Only judge additions and meaning. Do not judge whether references were resolved, another check covers that.

CONVERSATION:
{conversation}

REWRITE:
{rewrite}"""


class Reference(BaseModel):
    phrase: str
    resolves_to: str
    resolved_in_rewrite: bool


# evidence fields come before the verdict, so the model writes the evidence
# first and the verdict is built on it
class EntityVerdict(BaseModel):
    references: list[Reference]
    reasoning: str
    verdict: Literal["pass", "fail"]


class AddedVerdict(BaseModel):
    added_terms: list[str]
    meaning_changed: bool
    reasoning: str
    verdict: Literal["pass", "fail"]


class RewriteJudge:
    def __init__(self, model: str = JUDGE_MODEL):
        self._client = OpenAI()
        self._model = model

    def _judge(self, name: str, prompt: str, schema: type[BaseModel]) -> dict:
        response = self._client.chat.completions.parse(
            name=name,
            model=self._model,
            temperature=0,
            response_format=schema,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # a refusal is not a pass or a fail, keep it visible
            return {"verdict": "refused"}
        return parsed.model_dump()

    def entity_resolved(self, history: list[dict], user_message: str, rewrite: str) -> dict:
        prompt = ENTITY_PROMPT.format(
            conversation=format_conversation(history, user_message), rewrite=rewrite
        )
        return self._judge("judge-entity", prompt, EntityVerdict)

    def nothing_added(self, history: list[dict], user_message: str, rewrite: str) -> dict:
        prompt = ADDED_PROMPT.format(
            conversation=format_conversation(history, user_message), rewrite=rewrite
        )
        return self._judge("judge-added", prompt, AddedVerdict)

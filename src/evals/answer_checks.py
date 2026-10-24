import asyncio
from typing import Literal

# drop-in wrapper: logs each judge call as a Langfuse generation with model and tokens
from langfuse.openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from src.evals.rewrite_checks import JUDGE_MODEL
from src.rewrite.rewriter import format_conversation

IDK_PROMPT = """You are grading the answer of a chatbot that answers questions from documents.

Your job: decide whether the answer says it cannot answer the question.

- "full": the answer says it cannot answer and gives no real answer, for example "I do not have specific information about that".
- "partial": the answer gives some of what was asked, but says it cannot answer another part.
- "none": the answer tries to answer everything and never says it lacks the information.

Only judge whether the answer admits it can't answer. Do not judge whether the answer is correct.

CONVERSATION:
{conversation}

ANSWER:
{answer}"""


class IdkVerdict(BaseModel):
    # the sentence that admits not knowing, quoted, or empty if there is none
    idk_quote: str
    reasoning: str
    idk: Literal["full", "partial", "none"]


# what a right answer looks like for each MTRAG answerability label.
# unanswerable: only a full IDK is right. answerable: any IDK is wrong.
# partial: answer what you can and say what you can't.
EXPECTED_IDK = {"UNANSWERABLE": "full", "PARTIAL": "partial", "ANSWERABLE": "none"}


def idk_score(idk: str, answerability: str) -> int:
    return int(idk == EXPECTED_IDK[answerability])


class AnswerJudge:
    def __init__(self, model: str = JUDGE_MODEL):
        self._client = OpenAI()
        self._model = model
        llm = llm_factory(model, client=AsyncOpenAI(), temperature=0)
        self._faithfulness = Faithfulness(llm=llm)

    def idk(self, history: list[dict], user_message: str, answer: str) -> dict:
        prompt = IDK_PROMPT.format(
            conversation=format_conversation(history, user_message), answer=answer
        )
        response = self._client.chat.completions.parse(
            name="judge-idk",
            model=self._model,
            temperature=0,
            response_format=IdkVerdict,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # a refusal is not a verdict, keep it visible
            return {"idk": "refused"}
        return parsed.model_dump()

    def faithfulness(
        self, history: list[dict], user_message: str, answer: str, passages: list[str]
    ) -> dict:
        return asyncio.run(self._faithfulness_steps(history, user_message, answer, passages))

    async def _faithfulness_steps(self, history, user_message, answer, passages) -> dict:
        # RAGAS only checks against passages. The plan also allows earlier
        # turns, so the history goes in as one extra context.
        contexts = passages + [format_conversation(history, user_message)]
        question = format_conversation(history, user_message)

        # same two steps as Faithfulness.ascore, called one by one so each
        # statement's verdict can be reviewed, not just the final ratio
        statements = await self._faithfulness._create_statements(question, answer)
        if not statements:
            return {"score": None, "statements": []}
        verdicts = await self._faithfulness._create_verdicts(statements, "\n".join(contexts))
        rows = [s.model_dump() for s in verdicts.statements]
        return {"score": self._faithfulness._compute_score(verdicts), "statements": rows}

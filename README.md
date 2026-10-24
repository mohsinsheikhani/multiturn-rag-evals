# Multi-Turn RAG Evals

A multi-turn RAG bot built to be tested, not shipped. In a chat, the user's later messages lean on earlier ones ("How much does it cost?"), so a bot can fail on turn 4 because of something it got wrong on turn 2. This repo measures where that happens: at each step of a turn (rewrite, retrieve, generate) and, next, across the whole conversation.

Data is IBM's [MTRAG benchmark](https://arxiv.org/abs/2501.03468): human-written conversations with gold passages and reference answers for every turn. This repo uses the **Govt** set (28 conversations, 49,607 passages).

## How to evaluate a multi-turn chatbot

This flow works for any chatbot, not only RAG. The steps inside a turn change (a RAG bot rewrites, retrieves and generates; an agent might call tools), but the order of the work stays the same. It follows Hamel Husain's [guide to debugging multi-turn traces](https://hamel.dev/blog/posts/evals-faq/how-do-i-debug-multi-turn-conversation-traces.html).

1. **Get conversations.** In production, these are real user chats. Before launch, a simulated user (an LLM given a goal, what it knows and when to stop) chats with the bot. The bot answers from its own earlier replies, so its mistakes carry forward like they would with a real user.
2. **Judge each whole conversation.** An LLM judge reads the conversation and gives pass or fail: did the user get what they came for? The expected outcome is written before the run. Turns taken to get there are tracked separately.
3. **Read the failing conversations.** Read the user and bot messages first, then the steps underneath them.
4. **Mark only the first failure.** Later mistakes often come from the first one, so fixing it can fix the rest.
5. **Turn that failure into a single-turn test.** Rerun only that turn, with a correct history before it. If it still fails, the problem isn't the conversation. It's one of the steps inside the turn.
6. **Check each step on that turn.** One check per step shows which one broke. For RAG: was the message rewritten right, were the right passages found, and was the answer right and backed by them.
7. **Fix one thing and rerun.** The single-turn tests become a cheap regression set, and the conversation pass rate shows whether users are better off.

**Where this repo is:** steps 5 to 7 are built and run on 41 turns (results below). Steps 1 to 4, with simulated users, come next.

## Results

41 turns from 5 conversations (72, 92, 44, 47, 108), each turn run with the gold history before it. Every number is the average over the turns in that group.

| turns | count | rewrite correct | Recall@5 | "I don't know" correct | faithfulness |
|---|---|---|---|---|---|
| all | 41 | 0.61 | 0.39 | 0.59 | 0.91 |
| turn 1 | 5 | 1.00 | 0.33 | 1.00 | 0.78 |
| later turns | 36 | 0.56 | 0.40 | 0.53 | 0.94 |
| answerable | 26 | 0.65 | 0.40 | 0.73 | 0.93 |
| partly answerable | 9 | 0.78 | 0.48 | 0.33 | 0.78 |
| unanswerable | 6 | 0.17 | 0.19 | 0.33 | 1.00 |

- **Rewrite correct:** the bot labelled the message as standalone or not correctly and, when it needed a rewrite, two LLM judges passed it (the reference was resolved and nothing was added).
- **Recall@5:** share of the gold passages found in the top 5 retrieved.
- **"I don't know" correct:** the bot answered, partly answered or refused, matching what MTRAG says the turn needs.
- **Faithfulness:** share of the answer's claims backed by the passages or the conversation (RAGAS, gpt-4o).

Retrieval is the weakest step. The full run history, including every config tried on the way here, is in [docs/experiments.md](docs/experiments.md).

## What I found

- **One missed rewrite takes down the whole turn.** In conversation 10, turn 4, the user asked "Who is responsible for covering my expenses?". It reads like a full question, so the bot left it unchanged. But "my expenses" means the appeal costs from turn 3. Retrieval found 0 of the gold passages and the bot said it had no information. Grading only the final answer would have blamed the generator.
- **The judge was grading the wrong text.** The "I don't know" judge kept quoting refusals from earlier turns in the history and grading the answer by them. In 72/6 the bot answered "Yes, there are specific advantages for Veteran-owned businesses when bidding on government contracts." and the judge called it a refusal. Fixing the judge prompt moved the score from 0.46 to 0.71 without touching the bot. Judges need checking before their numbers mean anything.
- **On turn 1 the rewriter answered instead of rewording.** With no history, the prompt looked like a plain chat waiting for a reply. "How do I know I am on a secured government website before sharing sensitive information?" came back as "Look for 'https://' in the URL and a padlock icon in the address bar to ensure you are on a secured government website." Turn 1 now skips the rewriter.
- **A prompt fix moved the failure instead of removing it.** The original generator prompt only allowed a full refusal, so partly answerable turns were refused whole. A new rule to answer what the documents cover raised that group from 0.22 to 0.33. But overall "I don't know" correct fell from 0.71 to 0.59, because the bot started answering unanswerable turns from loosely related passages.
- **Some gold passages can't be found at all.** For 3 of the failing partly answerable turns, I wrote the best standalone query by hand. None of the gold passages reached the top 5. Better rewrites won't fix those turns. Retrieval has to.

## How a turn runs

The bot is a LangGraph pipeline with three steps. History is passed in on every turn, never stored, so the same bot can be fed gold history or its own earlier answers.

- **Rewrite:** labels the last message as standalone or not, and rewrites it into a standalone search query. gpt-4o with structured output. Skipped on turn 1.
- **Retrieve:** the rewrite and the full conversation are each searched with embeddings (text-embedding-3-small in Qdrant) and keyword search (BM25). The four result lists are merged with reciprocal rank fusion, the top 20 go to a gpt-4o-mini reranker, and the top 3 go to the generator.
- **Generate:** gpt-4o-mini answers in under 150 words from the 3 passages and the full conversation, or says "I do not have specific information".

Every turn is one Langfuse trace, grouped into one session per conversation.

## Checks used for each turn

- **One check per step.** Rewrite: two LLM judges. Retrieval: Recall@5 and nDCG@5 against MTRAG's gold passage ids. Answer: an LLM judge for "I don't know" correctness, and RAGAS faithfulness.
- **Sliced, not averaged.** Results are split by turn 1 against later turns, standalone against not, and answerability. The overall average hid that partly answerable turns were failing.
- **Read before fixing.** Every failure is read by hand, and one thing is changed per rerun.

## Run it

```
uv sync
git lfs pull    # dataset/corpus/govt.jsonl is stored with Git LFS
```

`.env` needs:

```
OPENAI_API_KEY=
QDRANT_CLUSTER_URL=
QDRANT_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
```

Index the corpus (one time, resumable):

```
uv run python -m scripts.index_corpus --corpus govt
```

Run the per-turn evals behind the results table:

```
uv run --no-sync python -m scripts.eval_lab2 --convs 72,92,44,47,108 --prompt v2 --rewrite-model gpt-4o --hybrid --generate-prompt v2
uv run --no-sync python -m scripts.eval_lab2 --table    # reprint the table from results/, no API calls
```

Replay one turn with gold history, or run a whole conversation on the bot's own answers:

```
uv run python -m scripts.run --conv 10 --turn 5
uv run python -m scripts.run --conv 10 --live
```

`scripts.run` still uses the first setup (MTRAG prompts, gpt-4o-mini rewriter, embeddings only).

## Project layout

```
src/
  rewrite/rewriter.py         rewrite step, MTRAG and v2 prompts
  retrieval/retriever.py      embedding and hybrid search, rank fusion
  retrieval/keyword.py        BM25
  rerank/reranker.py          LLM reranker
  generation/generator.py     generate step, MTRAG and v2 prompts
  graph/workflow.py           LangGraph pipeline
  graph/runner.py             runs one turn as one Langfuse trace
  evals/                      rewrite, retrieval and answer checks
  embedding/                  OpenAI embeddings, Qdrant client
scripts/
  eval_lab2.py                per-turn evals with slices
  run.py                      replay one turn, or run a conversation live
  index_corpus.py             indexes a corpus into Qdrant
results/                      saved per-turn results, one file per run
docs/experiments.md           every run, with full tables and failure write-ups
```

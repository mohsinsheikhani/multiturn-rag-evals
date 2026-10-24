# Experiments log

Every run so far, in the order it happened, with the full tables and the failure write-ups. The [README](../README.md) has the summary.

## Part 1: tuning on 5 turns

These runs used 5 hand-picked turns (10/4, 38/5, 72/6, 101/5, 101/7) to try one change at a time cheaply.

### Baseline results (before any improvement)

Stage 2, replay mode, on 5 turns: 10/4, 38/5, 72/6, 101/5, 101/7. All 5 are answerable in MTRAG.

Retrieval, searching with the bot's rewrite, top 5:

| turn | gold passages | Recall@5 | nDCG@5 |
|---|---|---|---|
| 10/4 | 2 | 0.00 | 0.00 |
| 38/5 | 6 | 0.50 | 0.72 |
| 72/6 | 3 | 0.00 | 0.00 |
| 101/5 | 2 | 0.00 | 0.00 |
| 101/7 | 2 | 0.00 | 0.00 |
| **mean** | | **0.10** | **0.14** |

Answer:

| turn | answer | IDK judge | expected | IDK score | faithfulness |
|---|---|---|---|---|---|
| 10/4 | "I do not have specific information." | full | none | 0 | skipped |
| 38/5 | overview of mediation in small claims | none | none | 1 | 1.0 |
| 72/6 | VA services for veterans | none | none | 1 | 1.0 |
| 101/5 | "I do not have specific information." | full | none | 0 | skipped |
| 101/7 | housing discrimination, Section 8 case | none | none | 1 | 1.0 |

Faithfulness is RAGAS with gpt-4o, checked against the passages the bot saw plus the earlier turns. It's skipped when the answer is a full "I don't know".

### v2 rewrite prompt results

v2 is the MTRAG prompt plus three rules, one per rewrite failure found below: a full sentence can still depend on the history; keep a bare-topic question as open as the user left it; when the user repeats an earlier question with a new subject, carry the earlier question over. Same 5 turns, replay mode. Switch with `--prompt mtrag|v2`.

Rewrite:

| turn | v2 rewrite | class | entity | added |
|---|---|---|---|---|
| 10/4 | Who is responsible for covering the expenses related to my case? | PASS | PASS | FAIL |
| 38/5 | What is the process for mediation in small claims court? | PASS | PASS | FAIL |
| 72/6 | Is there any program or advantage for veteran-owned companies? | PASS | PASS | FAIL |
| 101/5 | Do people who receive welfare assistance also get help with housing? | FAIL | skipped | skipped |
| 101/7 | What assistance is available for discrimination? | PASS | PASS | FAIL |

Class 4 of 5, same as MTRAG (10/4 fixed, 101/5 broke). Added failed 4 of 4, MTRAG failed 3 of 3.

Retrieval, rewrite only, top 5:

| turn | Recall@5 | nDCG@5 |
|---|---|---|
| 10/4 | 0.00 | 0.00 |
| 38/5 | 0.50 | 0.72 |
| 72/6 | 0.00 | 0.00 |
| 101/5 | 0.00 | 0.00 |
| 101/7 | 0.00 | 0.00 |
| **mean** | **0.10** | **0.14** |

Same as MTRAG.

Answer:

| turn | answer | IDK score | faithfulness |
|---|---|---|---|
| 10/4 | "I do not have specific information." | 0 | skipped |
| 38/5 | overview of mediation in small claims | 1 | 1.0 |
| 72/6 | VetCert, SBA benefits for veteran-owned businesses on government contracts | 1 | 1.0 |
| 101/5 | "I do not have specific information." | 0 | skipped |
| 101/7 | federal employment discrimination, EEO counselor | 1 | 1.0 |

Same scores as MTRAG.

### v2 prompt with gpt-4o as the rewriter

Same 5 turns, replay mode. `--prompt v2 --rewrite-model gpt-4o`.

Rewrite:

| turn | rewrite | class | entity | added |
|---|---|---|---|---|
| 10/4 | Who is responsible for covering my legal research expenses for the appeal? | PASS | PASS | FAIL |
| 38/5 | What is the role of mediation in small claims court? | PASS | PASS | PASS |
| 72/6 | Are there any advantages for veteran-owned companies in state contracting opportunities? | PASS | PASS | PASS |
| 101/5 | Do people who receive welfare in California also receive housing assistance? | FAIL | skipped | skipped |
| 101/7 | Is there any assistance available for homeowners facing discrimination? | PASS | PASS | FAIL |

Retrieval, rewrite only, top 5:

| turn | Recall@5 | nDCG@5 |
|---|---|---|
| 10/4 | 0.00 | 0.00 |
| 38/5 | 0.67 | 0.85 |
| 72/6 | 0.00 | 0.00 |
| 101/5 | 0.00 | 0.00 |
| 101/7 | 0.00 | 0.00 |
| **mean** | **0.13** | **0.17** |

Answer:

| turn | answer | IDK score | faithfulness |
|---|---|---|---|
| 10/4 | "I do not have specific information." | 0 | skipped |
| 38/5 | overview of mediation in small claims | 1 | 1.0 |
| 72/6 | VetCert, VA help for veteran-owned businesses with government contracts | 1 | 1.0 |
| 101/5 | "I do not have specific information." | 0 | skipped |
| 101/7 | "I do not have specific information." | 0 | skipped |

### v2 prompt with gpt-5.5 as the rewriter

Same 5 turns, replay mode. `--prompt v2 --rewrite-model gpt-5.5`. gpt-5.5 only accepts the default temperature, so it runs without temperature 0.

Rewrite:

| turn | rewrite | class | entity | added |
|---|---|---|---|---|
| 10/4 | Who is responsible for covering my expenses for an appeal? | PASS | PASS | FAIL |
| 38/5 | How about mediation for my small claims case? | PASS | PASS | PASS |
| 72/6 | Is there any advantage for veteran-owned companies? | PASS | PASS | PASS |
| 101/5 | Do people who receive welfare also receive housing? | PASS | skipped | skipped |
| 101/7 | What about discrimination in housing? | PASS | PASS | PASS |

Retrieval, rewrite only, top 5:

| turn | Recall@5 | nDCG@5 |
|---|---|---|
| 10/4 | 0.00 | 0.00 |
| 38/5 | 0.67 | 0.87 |
| 72/6 | 0.00 | 0.00 |
| 101/5 | 0.00 | 0.00 |
| 101/7 | 0.00 | 0.00 |
| **mean** | **0.13** | **0.17** |

Answer:

| turn | answer | IDK score | faithfulness |
|---|---|---|---|
| 10/4 | "I do not have specific information." | 0 | skipped |
| 38/5 | overview of mediation in small claims | 1 | 1.0 |
| 72/6 | VetCert, VA support for veteran-owned businesses | 1 | 0.75 |
| 101/5 | "I do not have specific information." | 0 | skipped |
| 101/7 | "I do not have specific information." | 0 | skipped |

### v2 + gpt-4o rewriter, with reranking

Retrieval change: the rewrite and the full history each search the top 30, the two lists are pooled (about 50 passages, no repeats), and gpt-4o-mini picks the best ones from the pool. Generator still gets 3 passages. `--rerank` on `eval_answer`; `eval_retrieval` always reports it as the `rerank` strategy.

Retrieval, top 5 after reranking:

| turn | Recall@5 | nDCG@5 |
|---|---|---|
| 10/4 | 0.50 | 0.24 |
| 38/5 | 0.50 | 0.70 |
| 72/6 | 0.33 | 0.18 |
| 101/5 | 0.00 | 0.00 |
| 101/7 | 0.00 | 0.00 |
| **mean** | **0.27** | **0.22** |

gpt-4o as the reranker gave 0.17 / 0.19 and 0.27 / 0.27 on two runs, so it's not clearly better than gpt-4o-mini and costs much more.

Answer, generator sees the top 3:

| turn | answer | IDK score | faithfulness |
|---|---|---|---|
| 10/4 | appeal court fees and fee waivers | 1 | 1.0 |
| 38/5 | overview of mediation in small claims | 1 | 1.0 |
| 72/6 | VetCert, VA support for veteran-owned businesses with contracts | 1 | 1.0 |
| 101/5 | "I do not have specific information." | 0 | skipped |
| 101/7 | EEOC charge of discrimination at work | 1 | 0.86 |

With the generator seeing the top 5 instead, IDK correct dropped to 3/5 (10/4 and 101/7 became partial "I don't know" answers).

### v2 + gpt-4o rewriter, hybrid search (embeddings + BM25), RRF, rerank

Retrieval change: the rewrite and the full history each search the top 30 with embeddings and with BM25 (run locally, no API cost). The four lists are merged with reciprocal rank fusion, the top 20 go to the gpt-4o-mini reranker, and it picks the final passages. `--rerank --hybrid` on `eval_answer`, `--hybrid` on `eval_retrieval`.

What led here (first two rows come from `eval_retrieval` runs, the last three from one script with the gpt-4o rewrites held fixed):

| pool sent to the reranker | Recall@5 | nDCG@5 |
|---|---|---|
| embeddings only, ~50 passages | 0.27 | 0.22 |
| embeddings + BM25 joined, ~95 passages | 0.17 | 0.18 |
| embeddings + BM25, RRF, no reranker | 0.27 | 0.23 |
| embeddings + BM25, RRF top 30, reranked | 0.33 | 0.31 |
| embeddings + BM25, RRF top 20, reranked | 0.40 | 0.35 |

Retrieval, final setup, top 5:

| turn | Recall@5 | nDCG@5 |
|---|---|---|
| 10/4 | 0.50 | 0.31 |
| 38/5 | 0.50 | 0.51 |
| 72/6 | 1.00 | 0.62 |
| 101/5 | 0.00 | 0.00 |
| 101/7 | 0.00 | 0.00 |
| **mean** | **0.40** | **0.29** |

For 101/5 and 101/7, none of the gold passages make it into the fused pool.

Answer, generator sees the top 3:

| turn | answer | IDK score | faithfulness |
|---|---|---|---|
| 10/4 | "I do not have specific information." | 0 | skipped |
| 38/5 | overview of mediation in small claims | 1 | 1.0 |
| 72/6 | VetCert, VA small business liaisons for veteran-owned businesses | 1 | 1.0 |
| 101/5 | "I do not have specific information." | 0 | skipped |
| 101/7 | California housing discrimination law, CRD complaints | 1 | 1.0 |

### Runs compared

| | MTRAG + gpt-4o-mini | v2 + gpt-4o-mini | v2 + gpt-4o | v2 + gpt-5.5 | v2 + gpt-5.4-mini | v2 + gpt-4o + rerank | v2 + gpt-4o + hybrid + RRF + rerank |
|---|---|---|---|---|---|---|---|
| class correct | 4/5 | 4/5 | 4/5 | 5/5 | 2/5 | same rewriter as v2 + gpt-4o | same rewriter as v2 + gpt-4o |
| entity pass | 3/3 | 4/4 | 4/4 | 4/4 | 0/2 | same rewriter as v2 + gpt-4o | same rewriter as v2 + gpt-4o |
| added pass | 0/3 | 0/4 | 2/4 | 3/4 | 2/2 | same rewriter as v2 + gpt-4o | same rewriter as v2 + gpt-4o |
| Recall@5 (mean) | 0.10 | 0.10 | 0.13 | 0.13 | 0.00 | 0.27 | **0.40** |
| nDCG@5 (mean) | 0.14 | 0.14 | 0.17 | 0.17 | 0.00 | 0.22 | **0.29** |
| IDK correct | 3/5 | 3/5 | 2/5 | 2/5 | 2/5 | **4/5** | 3/5 |
| faithfulness (mean of scored turns) | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 0.96 | **1.00** |

gpt-5.4-mini returned all 5 user messages unchanged, including 10/4 and 72/6, where it said non-standalone.

## Part 2: Lab 2, all checks on 41 turns


Setup: every turn of conversations 72, 92, 44, 47 and 108, in replay mode. That's 41 turns: 26 answerable, 9 partial and 6 unanswerable. The bot is the best setup from the 5-turn runs: v2 rewrite prompt, gpt-4o rewriter, hybrid search with reranking. One rewrite per turn feeds all three checks, so every check grades the same output. Per-turn results are saved in `results/`, one file per run.

Command:

```
uv run --no-sync python -m scripts.eval_lab2 --convs 72,92,44,47,108 --prompt v2 --rewrite-model gpt-4o --hybrid --generate-prompt v2
uv run --no-sync python -m scripts.eval_lab2 --table    # reprint the table from saved results, no API calls
```

How to read the slices: each slice is a group of turns, and each number is the average over the turns in that group. "All" is the average over all 41 turns. Each turn sits in three slices at once: turn 1 or later, standalone or not, and its answerability. "Rewrite" means the class was right and, for non-standalone turns, both rewrite judges passed. "Recall@5 rewrite" searches with the rewrite alone. "Recall@5 rewrite with reranking" is what the bot actually uses: the rewrite and the full history go into one pool, and an LLM picks the top 5. The generator sees the first 3 of those. Unanswerable turns also have gold passages in MTRAG, so they are scored on retrieval too.

Two fixes were made before the slice table was read, because both made the numbers wrong:

1. Turn 1 skipped the rewriter (see "Failure write-ups"). Only the 5 first turns were rerun.
2. The IDK judge was grading refusals from earlier turns in the history, not the bot's answer (see "Failure write-ups"). All 41 saved answers were graded again with the fixed judge. The bot was not rerun.

### Run 1 (MTRAG generator prompt), after those two fixes

| slice | turns | class | rewrite | Recall@5 rewrite | Recall@5 rewrite with reranking | nDCG@5 rewrite with reranking | IDK correct | faithfulness |
|---|---|---|---|---|---|---|---|---|
| all | 41 | 0.73 | 0.61 | 0.31 | 0.39 | 0.37 | 0.71 | 0.88 |
| turn 1 | 5 | 1.00 | 1.00 | 0.27 | 0.33 | 0.35 | 1.00 | 0.93 |
| later turns | 36 | 0.69 | 0.56 | 0.32 | 0.39 | 0.37 | 0.67 | 0.87 |
| standalone | 20 | 0.45 | 0.45 | 0.40 | 0.34 | 0.35 | 0.85 | 0.90 |
| not standalone | 21 | 1.00 | 0.76 | 0.23 | 0.42 | 0.39 | 0.57 | 0.85 |
| answerable | 26 | 0.81 | 0.65 | 0.32 | 0.41 | 0.40 | 0.88 | 0.93 |
| partial | 9 | 0.78 | 0.78 | 0.31 | 0.44 | 0.43 | **0.22** | **0.53** |
| unanswerable | 6 | 0.33 | 0.17 | 0.28 | 0.19 | 0.17 | 0.67 | 0.75 |

### Worst slice: partial

Partial has the lowest IDK correct (0.22) and the lowest faithfulness (0.53). It has 9 turns and 7 of them fail the IDK check, so all 7 were checked, not 10. For each failure: did the rewrite pass, did a gold passage reach the 3 passages the generator sees, and what did the bot answer.

| turn | rewrite | gold passages in the generator's 3 | bot's answer type (expected partial) |
|---|---|---|---|
| 92/2 | pass | 0 of 3 | full refusal |
| 47/3 | pass | 0 of 3 | full refusal |
| 47/4 | pass | 1 of 3 | full refusal |
| 47/6 | pass | 0 of 3 | full refusal |
| 47/8 | pass | 2 of 3 | answered everything (faithfulness 0.18) |
| 108/4 | pass | 2 of 2 | full refusal |
| 108/7 | class wrong | 2 of 4 | full refusal |

Debug trick on the three turns with no gold passage in front of the generator (92/2, 47/3, 47/6). A hand-written standalone query was searched:

- 92/2: "How do I file a complaint with a state other than New York, such as California?" No gold passage in the top 5, and 1 in the fused pool of 20.
- 47/3: "Are children under 18 allowed to submit personal information on state government websites?" No gold passage in the top 5, and 1 in the fused pool of 20.
- 47/6: "If government websites get hacked, how is my personal information protected?" No gold passage in the top 5, and none in the fused pool of 20.

So even a good query doesn't find these gold passages. Retrieval misses them, not the rewrite.

Where the partial failures come from: the rewrite passed on 6 of the 7. Retrieval missed gold on 3 of them. The answer step is to blame on all 7: 6 times it gave the full refusal and once it answered everything. That includes 47/4, 108/4 and 108/7, where gold passages were in front of it. The MTRAG generator prompt only offers one way out, the full refusal, so a turn the documents half answer gets refused whole.

Fix: a v2 generator prompt (`--generate-prompt v2`). It keeps the MTRAG prompt and adds rules: if the user asked for several things and the documents cover only some, answer those and say which ones they don't cover. Don't add notes about gaps when the answer is complete. Use the refusal alone only when nothing helps. When the user is reacting to an earlier answer, reply to that.

### Run 2: first version of the v2 generator prompt

The first version of the rule said "if the documents answer only part of the query, answer that part and say which part they do not cover". Partial went up: IDK correct from 0.22 to 0.56, faithfulness from 0.53 to 0.88. But the bot started adding a gap note to complete answers too. For example, 44/1 ended with "The documents do not cover specific web extensions beyond the ".gov" domain." Answerable IDK correct fell from 0.88 to 0.50. The rule was then limited to things the user actually asked for.

### Run 3: final

| slice | turns | class | rewrite | Recall@5 rewrite | Recall@5 rewrite with reranking | nDCG@5 rewrite with reranking | IDK correct | faithfulness |
|---|---|---|---|---|---|---|---|---|
| all | 41 | 0.73 | 0.61 | 0.31 | 0.39 | 0.38 | 0.59 | 0.91 |
| turn 1 | 5 | 1.00 | 1.00 | 0.27 | 0.33 | 0.35 | 1.00 | 0.78 |
| later turns | 36 | 0.69 | 0.56 | 0.31 | 0.40 | 0.38 | 0.53 | 0.94 |
| standalone | 20 | 0.45 | 0.45 | 0.39 | 0.35 | 0.35 | 0.70 | 0.86 |
| not standalone | 21 | 1.00 | 0.76 | 0.23 | 0.42 | 0.41 | 0.48 | 0.97 |
| answerable | 26 | 0.81 | 0.65 | 0.31 | 0.40 | 0.39 | 0.73 | 0.93 |
| partial | 9 | 0.78 | 0.78 | 0.31 | 0.48 | 0.47 | 0.33 | 0.78 |
| unanswerable | 6 | 0.33 | 0.17 | 0.28 | 0.19 | 0.17 | 0.33 | 1.00 |

Run 1 against run 3, IDK correct and faithfulness:

| slice | IDK correct, run 1 | IDK correct, run 3 | faithfulness, run 1 | faithfulness, run 3 |
|---|---|---|---|---|
| all | 0.71 | 0.59 | 0.88 | 0.91 |
| answerable | 0.88 | 0.73 | 0.93 | 0.93 |
| partial | 0.22 | 0.33 | 0.53 | 0.78 |
| unanswerable | 0.67 | 0.33 | 0.75 | 1.00 |

What the final numbers say: the partial slice got better (IDK correct from 0.22 to 0.33, faithfulness from 0.53 to 0.78), and faithfulness went up overall. But IDK correct went down overall, from 0.71 to 0.59. Answerable turns still get a gap note 5 times out of 26. On unanswerable turns the bot now answers from loosely related passages more often, with a refusal on only 2 of 6. The rule trades refusals for answers, and that only pays off on partial turns. Rewrite and retrieval numbers barely moved between runs, which is expected, since only the generator prompt changed.

Retrieval is the other weak spot. Recall@5 with reranking is 0.39 overall, and the debug trick showed that for 3 of the 7 partial failures, even a hand-written query can't find the gold passages.

## Failure write-ups


- **conv 10, turn 4: rewrite class wrong.** "Who is responsible for covering my expenses?" reads like a full question, so the bot called it standalone and left it unchanged. But "my expenses" means the costs of the appeal from turn 3. As written the query is unanswerable, and once it's rewritten ("Who is responsible for covering my costs of an appeal?") the corpus has the answer. A complete sentence isn't the same as a standalone query.
- **conv 38, turn 5: rewrite narrowed the question.** "How about mediation?" came after a filing fee question, and the bot rewrote it to "What is the process for mediation in small claims court?". The gold answer is a general overview of mediation: you can mediate before or after filing, the court offers it for free, it lets both sides settle without a trial, and if you settle you file a Request For Dismissal form. "Process" is the bot's own word and covers only part of that. The class was right and the entity was resolved, but the rewrite still changed the question. The nothing-added judge caught it.
- **conv 72, turn 6: rewrite lost the context, and the entity judge missed it.** The turn before asked if women-owned companies get an advantage in state contracts. "anything for vets?" asks the same about veteran-owned companies. The bot rewrote it to "Is there any support or programs available for veterans?", which drops companies and state contracts. The entity judge resolved "vets" to "veterans" instead of "veteran-owned businesses", so it missed the reference and passed a bad rewrite. The nothing-added judge also passed it.
- **conv 101, turn 7: nothing-added judge too strict.** "Discrimination" was rewritten to "What are the laws regarding discrimination in housing?", which is a good rewrite. The judge failed it for two reasons, and both are judge mistakes:
  - It said discrimination could mean any kind, not only housing. But every turn in the conversation is about housing in California, so "housing" comes from the history, which is what a rewrite is for.
  - It flagged "regarding" and "laws" as added terms. "Regarding" is a plain connecting word the prompt already says to ignore. "Laws" is a choice the rewriter made, since the user typed only a keyword, but it's a reasonable one, and it fits the gold answer (the agency that enforces the Fair Employment and Housing Act).

  Rule this sets: when the user types only a keyword, the rewriter has to turn it into a question, and a reasonable framing word like "laws" doesn't count as an addition.
- **Turn 1: the rewriter gets called with no history, and sometimes answers instead of rewording.** In the Lab 2 run (conversations 72, 92, 44, 47, 108), the rewriter called 4 of the 5 first turns non-standalone, even though there is nothing before turn 1 for a message to depend on. In 72, 92 and 108 it gave back almost the same question, so only the class was wrong. In 44 and 47 it answered the question instead of rewording it:
  - 44/1: "What is the Web extension for federal government websites and how do we ensure the site is secure?" was rewritten to "The Web extension for federal government websites is '.gov', and to ensure the site is secure, look for 'https://' in the URL and a padlock icon in the browser's address bar."
  - 47/1: "How do I know I am on a secured government website before sharing sensitive information?" was rewritten to "Look for 'https://' in the URL and a padlock icon in the address bar to ensure you are on a secured government website."

  Likely reason: the MTRAG prompt ends with the user's message followed by "ASSISTANT:", which looks like the spot where the assistant replies. With no history above it, the prompt reads like a plain chat waiting for an answer. Our code also sends turn 1 to the rewriter at all, when it could skip the rewrite and pass the message through unchanged.

  Fix: when there is no history, the rewriter now skips the model and returns the message unchanged as standalone. After rerunning the 5 first turns, class and rewrite went from 1 of 5 correct to 5 of 5. Across all 41 turns, class went from 0.63 to 0.73.
- **IDK judge graded the history, not the answer.** On 41 turns, the judge often quoted a refusal from an earlier agent turn in the history and graded the answer by it. In 72/6, the bot answered "Yes, there are specific advantages for Veteran-owned businesses when bidding on government contracts." and the judge called it a full refusal. It quoted "I do not have specific information on the process for becoming a contractor with the Office of General Services.", which is from an earlier turn. Fix: the prompt now labels the conversation as context only, says to grade only the answer, and says the quote must come word for word from the answer. Graded again, 10 of 41 verdicts changed, and every quote now comes from the answer. IDK correct over all 41 turns went from 0.46 to 0.71.


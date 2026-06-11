# agentic-rag-pipeline

A retrieval-augmented generation pipeline that actually retrieves, actually grounds
its answers, and runs entirely offline. It implements BM25 from scratch, a dense
retriever built on deterministic hashing TF-IDF vectors (no embedding model, no
downloads), reciprocal rank fusion of the two, a query-rewriting loop driven by
retrieval gaps, a tool-using agent with a hard step budget, an answer-grounding check
that traces claims back to retrieved chunks, and an offline evaluation set reporting
recall@k and answer coverage.

The corpus ships with the repository: eight short technical pages about the
techniques the pipeline itself uses. Answers are extractive (sentences drawn from the
retrieved passages), which keeps every step deterministic and inspectable.

## What is in the box

| Module | Job |
|---|---|
| `src/ragpipe/config.py` | Frozen dataclasses for chunking, retrieval, rewriting, the agent, grounding and the whole pipeline; YAML loading with unknown-key rejection; parameter sweeps |
| `src/ragpipe/corpus.py` | `Document`, the built-in eight-page corpus, loading `.md`/`.txt` directories, writing a corpus back out, corpus statistics |
| `src/ragpipe/chunking.py` | `Chunk`/`Chunker`: fixed character windows with overlap and sentence-aware packing with sentence overlap; chunk statistics |
| `src/ragpipe/bm25.py` | Tokenisation, inverted index, smoothed IDF, exact BM25 scoring, `k1`/`b` parameters, ranked search, per-term score breakdown |
| `src/ragpipe/dense.py` | BLAKE2b hashing TF-IDF embedder with sign hashing, L2 normalisation, cosine search, nearest neighbours |
| `src/ragpipe/fusion.py` | Reciprocal rank fusion with weights, `FusionHit` provenance (which list ranked each chunk where), `HybridIndex` over both retrievers |
| `src/ragpipe/rewrite.py` | Coverage measurement, gap-term extraction, bounded query-rewriting loop with explicit stop reasons |
| `src/ragpipe/tools.py` | The `retrieve` / `lookup` / `summarise` tools, a registry that validates arguments and records failures instead of raising, extractive sentence selection |
| `src/ragpipe/agent.py` | The agent loop: rewrite → retrieve → conditional lookup → summarise → ground, with a step budget and a full trace |
| `src/ragpipe/grounding.py` | Claim extraction and token-overlap support scoring, `GroundingReport` with unsupported claims listed |
| `src/ragpipe/evaluation.py` | The 16-question offline evaluation set, recall@k, answer coverage, `EvalResult` |
| `src/ragpipe/cli.py` | `--stats`, `--query ...`, `--eval`, JSON output |

## Quickstart

```bash
pip install -r requirements.txt

# corpus and index statistics as JSON
python -m src.ragpipe.cli --stats

# ask a question
python -m src.ragpipe.cli --query "How does reciprocal rank fusion combine ranked lists?"

# several questions at once
python -m src.ragpipe.cli \
  --query "What does the k1 parameter control in BM25?" \
  --query "Why does chunk overlap protect against a split answer?"

# offline evaluation -> artifacts/eval.json
make eval

# test suite
make test        # or: python -m pytest -q
```

`make install / test / lint / stats / ask / eval / clean` are the targets. To run on
your own documents, point `corpus_path` in `config.yaml` at a directory of `.md` or
`.txt` files; the corpus is then read from disk instead of the built-in pages.

## Design notes

**Reciprocal rank fusion instead of score blending.** BM25 scores are unbounded and
depend on corpus statistics, while hashing cosine similarities live in `[-1, 1]`.
Adding them would need a normalisation step that is itself a tuning decision, and
would break the moment the corpus changed size. RRF only needs ranks, so the two
retrievers can be developed independently and the fusion constant `k=60` damps the
top ranks so neither retriever can dominate through a single confident hit.

**Sign hashing, because bucket collisions are the whole problem with hashing
embeddings.** Each token maps to `blake2b(token) % dim` with a sign taken from
another hash bit. Without the sign, colliding tokens always add to the same
coordinate and inflate similarity between unrelated texts; with it, collisions tend to
cancel. `blake2b` is used rather than Python's `hash()` because the latter is
randomised per process, which would make retrieval non-reproducible between runs.

**The rewriting loop stops on coverage or on a budget, never on a score threshold.**
Scores are not comparable across queries; coverage (fraction of the question's terms
that appear in the retrieved passages) and the presence of gap terms are. The loop
therefore has two honest stop reasons — `coverage` and `no_gaps` — plus `budget` and
`no_change`, and every stop reason is recorded in the result. Each round costs a
retrieval call, which is why the budget is a config value rather than a loop guard
inside the code.

**Grounding is labelled as a heuristic.** The check splits an answer into claims and
measures the fraction of each claim's content tokens that appear in some cited chunk.
It catches a fabricated sentence with no overlap, which is the common failure, but it
cannot detect a claim that reuses the right words while stating the wrong
relationship, and it calls a good paraphrase with no shared tokens unsupported. The
report's `method` field says so, so nothing downstream can mistake it for entailment.

**Extractive answers, so the pipeline can be trusted end to end.** The `summarise`
tool ranks sentences from the retrieved chunks by query overlap and returns them
verbatim. That makes answers reproducible and makes the grounding score meaningful;
the limitation section explains what this rules out.

## Tests

The suite has **135 tests** (`python -m pytest -q` → `135 passed in 0.38s`), all
offline and deterministic:

| File | Tests | What it asserts |
|---|---|---|
| `tests/test_config.py` | 13 | YAML round-trips, defaults for partial configs, unknown keys and out-of-range values raise, override helpers, sweep expansion |
| `tests/test_corpus.py` | 9 | Corpus shape and uniqueness, document validation, write/load round-trip, statistics arithmetic, duplicate-id and empty-corpus errors |
| `tests/test_chunking.py` | 14 | Exact fixed-window text/spans, overlap repetition, short-tail dropping, sentence packing, overlap repetition, long-sentence fallback, id prefixing, duplicate-id rejection |
| `tests/test_bm25.py` | 13 | Tokenisation, exact IDF (`ln(2.6667)`, `ln(1.6)`), exact scores (`0.6714`, `0.4700`) computed from the formula, `b` lowering long-document scores, ranking order, zero-score dropping, error paths |
| `tests/test_dense.py` | 11 | Pinned bucket/sign values, exact IDF including the unseen-term case, unit norms, cosine 1.0 for identical and 0.0 for disjoint texts, exact-match retrieval, shape validation |
| `tests/test_fusion.py` | 12 | Exact RRF values (`1/(k+rank)`), duplicate handling, weight-driven order flips, provenance maps, `top_n`, hybrid index sources/agreement/stats, validation |
| `tests/test_rewrite.py` | 11 | Coverage and gap-frequency arithmetic, term appending, idempotence, each stop reason, budget enforcement, helper guards |
| `tests/test_tools.py` | 12 | Registry registration/duplicates/unknown tools, failure recorded not raised, retrieve payload shape, exact-term lookup, snippet extraction, summarise limits, tool summary counts |
| `tests/test_agent.py` | 12 | Answer assembly with sources, missing-term detection, step numbering, budget enforcement, rewrite-disabled path costs no retrieval, lookup branch, grounding caveat on a non-extractive sentence |
| `tests/test_grounding.py` | 10 | Claim extraction dropping the `Sources:` suffix, exact support fractions (`2/3`), thresholds changing the verdict, report coverage arithmetic, best-evidence selection |
| `tests/test_evaluation.py` | 11 | Exact recall@k cases, coverage arithmetic, the 16-question set materialised against real chunk ids, pipeline report floors (`recall@5 ≥ 0.8`, `coverage ≥ 0.7`) |
| `tests/test_cli.py` | 7 | Pipeline assembly, stats/ask/eval payloads, JSON output to a file, agent/index wiring |

Expected values are computed by hand where the formula makes that possible (BM25
scores, RRF sums, hashing buckets, recall fractions) and use lower bounds only where
the number depends on corpus statistics. No test touches the network, downloads a
model, or writes outside `tmp_path`.

Measured on the built-in corpus with the default configuration: 8 documents, 16
chunks, **recall@5 = 0.875** and **answer coverage = 0.875** over the 16-question set
(12 questions retrieve every gold chunk). Those numbers are reproducible with
`make eval`.

## Limitations

* Answers are extractive: the pipeline selects and cites sentences, it does not write
  new prose. There is no language model in the loop, so there is no generation step to
  evaluate.
* Dense retrieval is lexical hashing TF-IDF, not a neural encoder. It has no synonymy
  beyond shared tokens, and word order is discarded entirely.
* There is no stemming or lemmatisation, so `retrieve` and `retrieval` are different
  terms. A Porter stemmer would be the next improvement to BM25.
* Grounding is token overlap. It cannot detect a claim that reuses the right words in
  the wrong relation, and it marks correct paraphrases as unsupported.
* The evaluation set has 16 questions over 8 documents, all written by hand against
  those pages. It is a regression check, not a benchmark, and it rewards wording
  overlap rather than correctness.
* Query rewriting uses term frequency in retrieved passages as its only signal; it
  does not use a language model, feedback clicks, or pseudo-relevance scoring beyond
  that.
* Single-process, in-memory index: a corpus that does not fit in memory would need a
  different storage layer.

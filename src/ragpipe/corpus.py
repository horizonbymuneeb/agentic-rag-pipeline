"""Documents and the offline corpus the pipeline runs on.

Two sources: a deterministic synthetic corpus that ships with the repository (so
tests and demos need no downloads) and plain-text files read from a local
directory.  Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class CorpusError(ValueError):
    """Raised when a corpus is empty or malformed."""


@dataclass(frozen=True)
class Document:
    """One source document: a stable id, a title and its body text."""

    doc_id: str
    title: str
    text: str
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise CorpusError("document needs a non-empty doc_id")
        if not self.text.strip():
            raise CorpusError(f"document {self.doc_id!r} has an empty body")

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated words."""
        return len(self.text.split())

    def preview(self, chars: int = 80) -> str:
        """Short single-line preview for logs."""
        flat = " ".join(self.text.split())
        return flat[:chars] + ("..." if len(flat) > chars else "")

    def as_dict(self) -> dict[str, Any]:
        """Serialisable mapping."""
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "source": self.source}


# A small hand-written corpus about this project family.  It is deliberately
# written as documentation prose (not keyword soup) so retrieval and grounding
# behave the way they would on real technical notes.
SYNTHETIC_PAGES: tuple[tuple[str, str, str], ...] = (
    (
        "bm25-scoring",
        "BM25 scoring and term saturation",
        "BM25 ranks documents with a sum over query terms of an inverse document frequency "
        "weight multiplied by a saturated term frequency. The k1 parameter controls how "
        "quickly term frequency saturates: larger k1 means repeated occurrences keep adding "
        "score for longer. The b parameter controls length normalisation, so long documents "
        "are not rewarded purely for containing more words. A value of b near one applies "
        "full normalisation by document length, while b equal to zero disables it. "
        "Saturation is why the third occurrence of a term adds much less than the first.",
    ),
    (
        "rank-fusion",
        "Reciprocal rank fusion of lexical and dense results",
        "Reciprocal rank fusion combines several ranked lists without needing comparable "
        "scores. Each document receives the sum of one over the constant k plus its rank in "
        "every list it appears in. The constant k, typically sixty, damps the influence of "
        "the very top ranks so a single list cannot dominate. Fusion is attractive because "
        "lexical scores and cosine similarities live on different scales and cannot be added "
        "directly. Documents absent from a list simply contribute nothing to their fusion score.",
    ),
    (
        "hashing-embeddings",
        "Hashing TF-IDF embeddings without a model download",
        "A hashing embedding maps tokens to fixed-size buckets using a deterministic hash, "
        "which keeps the vector width constant without learning a vocabulary. Term frequency "
        "inverse document frequency weighting down-weights common words that appear in most "
        "documents. Sign hashing assigns each token a positive or negative sign so collisions "
        "tend to cancel instead of always inflating the same coordinate. Normalising the "
        "vector to unit length makes the dot product equal to cosine similarity. The approach "
        "loses word order and cannot capture synonymy beyond shared stems.",
    ),
    (
        "chunking",
        "Chunking strategies and overlap",
        "Chunking splits long documents into passages small enough to retrieve and quote. "
        "Fixed-size chunking cuts on character windows and is trivial to reason about, but it "
        "can split a sentence in half. Sentence-aware chunking accumulates whole sentences up "
        "to a character budget and keeps a configurable number of trailing sentences as "
        "overlap. Overlap protects against the case where the answer spans a boundary: the "
        "same sentence appears in two neighbouring chunks, so at least one of them is fully "
        "retrievable. Oversized overlap wastes storage and produces near-duplicate results.",
    ),
    (
        "query-rewriting",
        "Query rewriting from retrieval gaps",
        "A query rewriting loop reformulates the question using evidence from the previous "
        "retrieval round. One useful signal is a gap term: a word that appears repeatedly in "
        "the retrieved passages but not in the question. Adding a small number of high "
        "frequency gap terms expands the query towards the vocabulary of the relevant "
        "documents without discarding the original wording. The loop needs a budget because "
        "each round costs a retrieval call, and unbounded rewriting tends to drift away from "
        "the user's intent.",
    ),
    (
        "grounding",
        "Answer grounding with token overlap",
        "Grounding checks whether the claims in a generated answer can be traced to retrieved "
        "passages. A cheap lexical version splits the answer into claims and measures the "
        "fraction of each claim's content tokens that appear in some cited chunk. A claim "
        "above the support threshold is reported as supported and otherwise as unsupported. "
        "This is a lexical heuristic, not entailment: it cannot detect a claim that reuses the "
        "right words with the wrong relationship, and it will mark a paraphrase with no shared "
        "tokens as unsupported even when it is correct.",
    ),
    (
        "agent-loop",
        "Tool-using agent loops and step budgets",
        "An agent loop interleaves planning, tool calls and observation until it can answer. "
        "A step budget bounds the loop so a confused agent cannot spend unbounded time. Each "
        "tool call consumes steps, and when the budget runs out the agent must answer with "
        "whatever evidence it has and flag that the budget was exhausted. Tools should "
        "validate their arguments and return structured results rather than raising "
        "stringly-typed errors. Logging the full trace makes failures debuggable.",
    ),
    (
        "evaluation",
        "Offline retrieval evaluation",
        "Offline evaluation compares retrieved chunk identifiers against a gold set. Recall at "
        "k is the fraction of gold chunks that appear in the first k results. Answer coverage "
        "measures the fraction of expected key terms that the produced answer contains. Both "
        "metrics are computed on a fixed question set with stable expected answers, so a "
        "change in retrieval configuration can be compared without a human in the loop. These "
        "metrics are proxies: they reward wording overlap and do not prove correctness.",
    ),
)


def synthetic_documents() -> list[Document]:
    """The built-in corpus as :class:`Document` objects."""
    return [
        Document(doc_id=doc_id, title=title, text=text, source="synthetic")
        for doc_id, title, text in SYNTHETIC_PAGES
    ]


def build_corpus(seed: int = 0, extra_documents: Sequence[Document] | None = None) -> list[Document]:
    """Return the offline corpus, optionally extended with caller documents.

    ``seed`` is accepted (and validated) because callers treat the corpus as
    generated data; the built-in pages are fixed text, so the seed only affects
    nothing today and is kept for interface stability.
    """
    if seed < 0:
        raise CorpusError("seed must be >= 0")
    documents = synthetic_documents()
    if extra_documents:
        documents.extend(extra_documents)
    if not documents:
        raise CorpusError("corpus is empty")
    ids = [document.doc_id for document in documents]
    if len(set(ids)) != len(ids):
        raise CorpusError("document ids must be unique")
    return documents


def load_corpus(directory: str | Path, suffixes: Sequence[str] = (".txt", ".md")) -> list[Document]:
    """Read every matching file in ``directory`` as one document.

    Files are sorted by name so the doc ids are stable across machines. The
    document id is the file stem, which is what the evaluation set keys on.
    """
    root = Path(directory)
    if not root.is_dir():
        raise CorpusError(f"corpus directory does not exist: {root}")
    documents: list[Document] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in suffixes or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].lstrip("# ").strip()
        documents.append(
            Document(
                doc_id=path.stem,
                title=first_line or path.stem,
                text=text,
                source=str(path),
            )
        )
    if not documents:
        raise CorpusError(f"no {'/'.join(suffixes)} documents found in {root}")
    return documents

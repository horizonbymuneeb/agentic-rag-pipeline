"""Answer grounding: can each claim in the answer be traced to a retrieved chunk?

The check is lexical: a claim is *supported* when the fraction of its content tokens
that appear in some cited chunk is at least ``support_threshold``.  That is a
heuristic, not entailment -- it cannot see that "BM25 uses k1 to normalise length"
borrows the right words while stating the wrong relationship, and it will call a
correct paraphrase with no shared tokens unsupported.  The report labels itself
accordingly so downstream code does not mistake it for a fact check.
"""

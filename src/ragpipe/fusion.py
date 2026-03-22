"""Hybrid retrieval by reciprocal rank fusion of the lexical and dense indexes.

RRF is used instead of score blending because BM25 scores are unbounded and
corpus-dependent while cosine similarities live in ``[-1, 1]``; adding them would
require a normalisation step that is itself a tuning decision.  Ranks are
comparable, so fusion only needs one constant::

    score(d) = sum over lists L containing d of 1 / (k + rank_L(d))

with ``k = 60`` by default.  The constant damps the top ranks so one retriever
cannot dominate on its own.
"""

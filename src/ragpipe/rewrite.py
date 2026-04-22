"""Query rewriting driven by retrieval gaps, with a hard round budget.

The signal used here is a *gap term*: a token that appears repeatedly in the
passages the previous round retrieved but not in the question.  Adding a couple of
those high-frequency terms pulls the query towards the vocabulary of the relevant
documents without throwing away the original wording.  The loop stops early when
the question's terms are already covered or when a rewrite would change nothing,
and it can never run more rounds than ``RewriteConfig.max_rounds``.
"""

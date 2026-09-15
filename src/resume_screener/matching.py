"""Keyword matching helpers shared by eligibility and scoring.

Matching uses non-alphanumeric boundaries so short tokens like ``rag`` do not
match inside ``storage``. Callers pass an already-lower-cased corpus.
"""

from __future__ import annotations

import re


def contains(corpus: str, term: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
    return re.search(pattern, corpus) is not None


def distinct_matches(corpus: str, terms: list[str]) -> list[str]:
    """Return the terms present in ``corpus``, de-duplicated, order preserved."""

    return [term for term in dict.fromkeys(terms) if contains(corpus, term)]

"""Grounding guardrail: does the generated answer stay inside the evidence?

A RAG answer is only trustworthy if it is *supported* by the retrieved
context. This module implements a cheap, deterministic lexical check: at least
``min_overlap`` of the answer's content words must appear in the context, and
the answer must not be too short to judge. When the check fails, the pipeline
replies "I don't know." instead of surfacing a fabrication.

The heuristic is intentionally conservative: short answers and abstentions
("I don't know") are always accepted, so a terse-but-correct answer is never
nuked. It is a first line of defense, not a substitute for a reranker or a
trained faithfulness model.
"""

from __future__ import annotations

from string import punctuation

from app.config import settings

_STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have he her his i if in is it its
    of on or she so that the their them there they this to was were will with you your
    not no do does did would could should may might must can could than then very just
    about into over under again further once here when where why how all any both each
    few more most other some such only own same too very s t don now""".split()
)

_TRANSLATE = str.maketrans({ch: " " for ch in punctuation})


def content_tokens(text: str) -> list[str]:
    """Lowercase content words (no punctuation/stopwords, length > 2)."""
    words = []
    for word in text.lower().translate(_TRANSLATE).split():
        if len(word) > 2 and word not in _STOPWORDS:
            words.append(word)
    return words


def is_abstention(answer: str) -> bool:
    return answer.strip().lower() == "i don't know"


def grounding_supported(
    answer: str,
    context: str,
    min_overlap: float | None = None,
    min_tokens: int | None = None,
) -> bool:
    """True when the answer is grounded in ``context`` (or not judgeable).

    ``min_overlap`` is the minimum fraction of answer content words that must
    appear in the context; ``min_tokens`` is the minimum answer length before
    the check kicks in.
    """
    min_overlap = settings.GROUNDING_MIN_OVERLAP if min_overlap is None else min_overlap
    min_tokens = settings.GROUNDING_MIN_TOKENS if min_tokens is None else min_tokens
    if not answer or is_abstention(answer):
        return True  # abstentions and empty answers are always fine
    answer_words = content_tokens(answer)
    if len(answer_words) < min_tokens:
        return True  # too short to judge - don't discard terse correct answers
    context_words = set(content_tokens(context))
    if not context_words:
        return False  # there is no evidence at all
    hits = sum(1 for w in answer_words if w in context_words)
    return (hits / len(answer_words)) >= min_overlap

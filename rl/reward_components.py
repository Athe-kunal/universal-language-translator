"""Reward penalty signals for RL translation training."""

import re

_HARD_REPETITION_RE = re.compile(r"(\S+(?:\s+\S+){0,4})(?:\s+\1){3,}")
_WORD_RE = re.compile(r"\S+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_NUMERIC_RE = re.compile(r"^[0-9०-९.,%:/+\-]+$")
_NGRAM_SIZE = 4
_MIN_TOKENS_FOR_NGRAM = 8
_STRIP_CHARS = ".,!?;:\"'()[]{}—-।"


def repetition_penalty(text: str) -> float:
    """Returns a degenerate/repeated-generation penalty in [0, 1].

    Args:
        text: Candidate translation text.

    Returns:
        1.0 for a short-phrase loop; otherwise a 4-gram repetition ratio,
        0.0 if there's not enough text to judge.
    """
    if _HARD_REPETITION_RE.search(text):
        return 1.0

    tokens = _WORD_RE.findall(text)
    if len(tokens) < _MIN_TOKENS_FOR_NGRAM:
        return 0.0

    ngrams = [tuple(tokens[i : i + _NGRAM_SIZE]) for i in range(len(tokens) - _NGRAM_SIZE + 1)]
    if not ngrams:
        return 0.0
    return max(0.0, 1.0 - len(set(ngrams)) / len(ngrams))


def language_switch_penalty(text: str) -> float:
    """Returns a non-numeric Latin-script penalty in [0, 1].

    Args:
        text: Candidate translation text.

    Returns:
        Fraction of alphabetic tokens (Devanagari + Latin) that are
        Latin-script; numeric tokens are always exempt.
    """
    tokens = _WORD_RE.findall(text)
    alphabetic = switched = 0
    for tok in tokens:
        stripped = tok.strip(_STRIP_CHARS)
        if not stripped or _NUMERIC_RE.match(stripped):
            continue
        has_devanagari = bool(_DEVANAGARI_RE.search(stripped))
        has_latin = bool(_LATIN_LETTER_RE.search(stripped))
        if not has_devanagari and not has_latin:
            continue
        alphabetic += 1
        if has_latin and not has_devanagari:
            switched += 1
    return switched / alphabetic if alphabetic else 0.0

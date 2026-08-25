"""Pure-Python reward penalty signals for RL translation training.

Split out from `rl/reward.py` so the repetition and language-switch checks
can be unit-tested (see tests/test_rl_reward.py) without pulling in
sentence-transformers or a GPU.
"""

import re

# Same "short phrase repeated 4+ times in a row" degeneration signature
# data_gen/translate_reasoning.py's has_repetition() rejects outright during
# SFT data generation - see that module's comment for the manual-review
# history behind the 1-5 token / 4-repeat thresholds. Reused verbatim here
# as the "unambiguously degenerate" hard case; the RL reward also wants a
# softer, continuous signal for texts that are repetitive without matching
# this exact pattern, which _ngram_repetition_ratio below provides.
_HARD_REPETITION_RE = re.compile(r"(\S+(?:\s+\S+){0,4})(?:\s+\1){3,}")

_WORD_RE = re.compile(r"\S+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
# A token counts as "numeric" (and is exempt from the language-switch
# penalty) if, once punctuation is stripped, every remaining character is a
# Western or Devanagari digit or common numeric connective punctuation
# (decimal point, thousands separator, percent, ratio/date separators).
_NUMERIC_RE = re.compile(r"^[0-9०-९.,%:/+\-]+$")

_NGRAM_SIZE = 4
_NGRAM_PENALTY_FLOOR_TOKENS = 8  # too short for n-gram stats to mean anything

_STRIP_CHARS = ".,!?;:\"'()[]{}—-।"  # । is the Devanagari danda (sentence stop)


def repetition_penalty(text: str) -> float:
    """Degenerate/repeated-generation penalty in [0, 1] (0 = clean).

    1.0 for the hard short-phrase-loop signature also used to reject SFT
    data outright; otherwise a softer signal from how much of the text is
    made of repeated n-grams, so a model stuck in a slightly longer or
    less-regular loop than the hard regex catches still gets penalized in
    proportion to how repetitive it is, rather than an all-or-nothing 0/1.
    """
    if _HARD_REPETITION_RE.search(text):
        return 1.0

    tokens = _WORD_RE.findall(text)
    if len(tokens) < _NGRAM_PENALTY_FLOOR_TOKENS:
        return 0.0

    ngrams = [tuple(tokens[i : i + _NGRAM_SIZE]) for i in range(len(tokens) - _NGRAM_SIZE + 1)]
    if not ngrams:
        return 0.0
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return max(0.0, 1.0 - unique_ratio)


def language_switch_penalty(text: str) -> float:
    """Penalty in [0, 1] for Latin-script words outside numeric values.

    Numbers (Western or Devanagari digits, with connective punctuation like
    `.`, `,`, `%`, `:`, `/`) are exempt regardless of script, since a
    translation correctly leaves numerals as-is. Any other token containing
    a Latin letter is treated as an English/code-switch leak. The penalty is
    the fraction such tokens make up of all alphabetic (Devanagari + Latin)
    tokens - 0 for a fully-Hindi (plus numbers/punctuation) response, up to
    1 for a fully-Latin-script one.
    """
    tokens = _WORD_RE.findall(text)
    alphabetic = 0
    switched = 0
    for tok in tokens:
        stripped = tok.strip(_STRIP_CHARS)
        if not stripped or _NUMERIC_RE.match(stripped):
            continue
        has_devanagari = bool(_DEVANAGARI_RE.search(stripped))
        has_latin = bool(_LATIN_LETTER_RE.search(stripped))
        if not has_devanagari and not has_latin:
            continue  # punctuation-only or another script entirely; not scored
        alphabetic += 1
        if has_latin and not has_devanagari:
            switched += 1
    if alphabetic == 0:
        return 0.0
    return switched / alphabetic

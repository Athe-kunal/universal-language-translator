"""Splits a long step into balanced pieces that each fit a token cap, without breaking mid-sentence.

Boundaries are chosen in order of preference: paragraph breaks, sentence ends, then single newlines. Code fences and
`$...$` / `$$...$$` math are masked while splitting so a boundary never lands inside one. Pieces are balanced
(each close to total / n_pieces tokens) so they decode in same-length batches. A single unit longer than the cap
(e.g. a huge code fence with no newlines) stays whole in its own piece.

    pieces, seps = split_text(text, count_tokens, cap=450)
    text == join_pieces(pieces, seps)     # with `seps[0]` the leading and `seps[i]` the gap after piece i-1
"""

import math
import re
from typing import Callable

_PROTECT = re.compile(r"```.*?```|\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
_PARA = re.compile(r"(\n[ \t]*\n\s*)")
_SENT = re.compile(r"((?<=[.!?।])\s+)")
_LINE = re.compile(r"(\n)")


def _units(text: str, count: Callable[[str], int], cap: int) -> list[tuple[str, str]]:
    """Returns [(unit_text, whitespace_after)] where no unit is splittable further within `cap` if avoidable."""
    saved: list[str] = []

    def mask(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"

    masked = _PROTECT.sub(mask, text)
    restore = lambda s: re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], s)

    def pairs(parts: list[str]) -> list[tuple[str, str]]:
        # re.split with a capture group alternates text, separator, text, ...
        return [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]

    out: list[tuple[str, str]] = []
    for para, para_sep in pairs(_PARA.split(masked)):
        if count(restore(para)) <= cap:
            out.append((restore(para), para_sep))
            continue
        sents = pairs(_SENT.split(para))
        for k, (sent, sent_sep) in enumerate(sents):
            sep = sent_sep if k < len(sents) - 1 else para_sep
            if count(restore(sent)) <= cap:
                out.append((restore(sent), sep))
                continue
            lines = pairs(_LINE.split(sent))
            for j, (line, line_sep) in enumerate(lines):
                out.append((restore(line), line_sep if j < len(lines) - 1 else sep))
    return out


def split_text(text: str, count: Callable[[str], int], cap: int) -> tuple[list[str], list[str]]:
    """Returns (pieces, seps): `seps` has len(pieces) + 1 entries: leading whitespace, the gap after each piece
    (the last being trailing whitespace). Every piece is stripped and non-empty."""
    if count(text) <= cap:
        return [text.strip()], [text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]]
    # normalise: every unit is (core, whitespace_after); stray whitespace moves into the neighbouring separator
    merged: list[tuple[str, str]] = []
    lead = ""
    for t, sep in _units(text, count, cap):
        core = t.strip()
        pre, post = t[: len(t) - len(t.lstrip())], t[len(t.rstrip()) :] if core else ""
        if not core:
            pre, post = t, ""
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + pre)
        else:
            lead += pre
        if core:
            merged.append((core, post + sep))
        elif merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + sep)
        else:
            lead += sep
    counts = [count(t) for t, _ in merged]
    n = max(1, math.ceil(sum(counts) / cap))
    target = sum(counts) / n
    groups: list[list[int]] = [[]]
    cur = 0
    for i, c in enumerate(counts):
        # start a new piece when adding this unit would overshoot the balanced target more than stopping short
        if groups[-1] and (cur + c > cap or cur + c - target > target - cur):
            groups.append([])
            cur = 0
        groups[-1].append(i)
        cur += c
    pieces, seps = [], [lead]
    for g in groups:
        body = "".join(merged[i][0] + (merged[i][1] if i != g[-1] else "") for i in g)
        pieces.append(body)
        seps.append(merged[g[-1]][1])
    return pieces, seps


def join_pieces(pieces: list[str], seps: list[str]) -> str:
    return seps[0] + "".join(p + s for p, s in zip(pieces, seps[1:]))

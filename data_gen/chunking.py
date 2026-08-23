"""Markdown-aware chunking of reasoning traces into translation units.

Segments an English reasoning trace / assistant response into markdown-aware
translation units for block-level English->Hindi translation, with byte-exact
provenance so translated units splice back into a reconstructed document.

No translation, no embeddings, no scoring here. Segment, classify, protect.
"""

import re
import threading
from dataclasses import dataclass, field, replace
from typing import Any

import tiktoken
from loguru import logger
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin

_MD = (
    MarkdownIt("commonmark")
    .use(dollarmath_plugin, allow_labels=True, double_inline=True)
    .use(amsmath_plugin)
    .enable("table")
)

# node.type -> (kind, translate)
_NODE_KIND: dict[str, tuple[str, bool]] = {
    "heading": ("heading", True),
    "paragraph": ("paragraph", True),
    "bullet_list": ("list", True),
    "ordered_list": ("list", True),
    "fence": ("code", False),
    "code_block": ("code", False),
    "math_block": ("math", False),
    "amsmath": ("math", False),
    "table": ("table", True),
    "blockquote": ("quote", True),
    "hr": ("rule", False),
}


@dataclass
class ProtectedSpan:
    """A zero-translation-freedom span protected behind a placeholder token.

    Attributes:
        placeholder: Per-unit placeholder token, e.g. "⟦0⟧".
        original: The original source text of the span.
        kind: One of "math_inline", "math_block", "amsmath", "code_inline",
            "link_url", "number".
        start: Offset of the span's start into the unit's text_raw.
        end: Offset of the span's end into the unit's text_raw.
    """

    placeholder: str
    original: str
    kind: str
    start: int
    end: int


@dataclass
class Fingerprint:
    """Structural signature of a unit, used for lossless-translation checks.

    Attributes:
        bullet_count: Number of list_item nodes in the unit.
        max_nest_depth: Deepest list/blockquote nesting depth.
        bold_span_count: Number of strong_open nodes.
        italic_span_count: Number of em_open nodes.
        heading_level: Heading level (1-6), or None if not a heading.
        placeholder_ids: Sorted list of placeholder tokens present in the unit.
        sentence_count: Number of sentences in text_raw.
        ends_with_colon: Whether text_raw ends with ":" (ignoring trailing
            whitespace).
    """

    bullet_count: int
    max_nest_depth: int
    bold_span_count: int
    italic_span_count: int
    heading_level: int | None
    placeholder_ids: list[str]
    sentence_count: int
    ends_with_colon: bool


@dataclass
class Unit:
    """A single translation unit with provenance and protection metadata.

    Attributes:
        unit_id: f"{doc_id}:{index:04d}".
        doc_id: Identifier of the parent document.
        source: "openthoughts" | "naturalreasoning".
        index: Contiguous index of this unit within the document, from 0.
        kind: One of heading/paragraph/list/code/math/table/quote/rule.
        text_raw: Exact source slice for this unit.
        text_protected: text_raw with protected spans replaced by placeholders.
        spans: Protected spans found in text_raw.
        char_start: Start offset of text_raw into the document.
        char_end: End offset of text_raw into the document.
        translate: Whether this unit should be sent to the translation model.
        token_count: cl100k_base token count of text_raw.
        fingerprint: Structural fingerprint of this unit.
        merged_from: Node types this unit was merged from, if any.
        split_index: Position of this unit within its split group, if split.
        is_recap: Whether this unit is a closing recap/summary (naturalreasoning).
        has_discourse_markers: Whether this unit contains discourse markers
            such as "wait", "hold on", "actually" (openthoughts).
    """

    unit_id: str
    doc_id: str
    source: str
    index: int
    kind: str
    text_raw: str
    text_protected: str
    spans: list[ProtectedSpan]
    char_start: int
    char_end: int
    translate: bool
    token_count: int
    fingerprint: Fingerprint
    merged_from: list[str] | None = None
    split_index: int | None = None
    is_recap: bool = False
    has_discourse_markers: bool = False


@dataclass
class ChunkedDocument:
    """A document broken into translation units, with enough state to splice
    translations back into a byte-exact reconstruction of the original text.

    Attributes:
        doc_id: Identifier of the document.
        source: "openthoughts" | "naturalreasoning".
        units: Ordered translation units.
        gaps: Whitespace between unit[i] and unit[i+1]; len(gaps) == len(units) - 1.
        original_length: len(text) of the source document.
        stats: Counts by kind, split rate, merge rate, translatable token total.
    """

    doc_id: str
    source: str
    units: list[Unit]
    gaps: list[str]
    original_length: int
    stats: dict[str, Any] = field(default_factory=dict)


def _line_starts(text: str) -> list[int]:
    """Builds a line-index -> char-offset table for O(1) node span lookups.

    Args:
        text: The full source document.

    Returns:
        List where entry i is the char offset of the start of line i.
    """
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _node_span(node: SyntaxTreeNode, line_starts: list[int]) -> tuple[int, int]:
    """Converts a block node's line map into a char offset span.

    Args:
        node: A block-level AST node with a populated `.map`.
        line_starts: Table from `_line_starts`.

    Returns:
        (char_start, char_end) offsets into the original document text.
    """
    assert node.map is not None, "_node_span requires a block node with a populated map"
    start_line, end_line = node.map
    return line_starts[start_line], line_starts[end_line]


def _top_level_nodes(text: str) -> tuple[SyntaxTreeNode, list[int]]:
    """Parses text and returns its top-level AST children plus the line table.

    Args:
        text: The source document.

    Returns:
        (tree, line_starts) where tree.children are the top-level block nodes.
    """
    tree = SyntaxTreeNode(_MD.parse(text))
    return tree, _line_starts(text)


_INLINE_PROTECT_KINDS = {"math_inline": "math_inline", "code_inline": "code_inline"}
_BLOCK_PROTECT_KINDS = {"math_block": "math_block", "amsmath": "amsmath"}

# Integers, decimals, percentages, and simple ranges (e.g. "42", "3.14", "50%", "10-20").
_NUMBER_RE = re.compile(
    r"\d+(?:,\d{3})*(?:\.\d+)?%?(?:\s*-\s*\d+(?:,\d{3})*(?:\.\d+)?%?)?"
)


def _find_protectable_nodes(
    node: SyntaxTreeNode,
    text_raw: str,
    line_starts: list[int],
    unit_char_start: int,
    cursor: int,
) -> tuple[list[ProtectedSpan], int]:
    """Walks a unit's subtree in document order, locating protectable spans.

    Handles math_inline/code_inline (inline tokens, no `.map`; located via a
    forward-moving cursor search) and math_block/amsmath (block tokens with a
    `.map`, e.g. a math block nested inside a list item or blockquote) and the
    `href` of link_open nodes. Link text itself is left untranslated-visible
    (not protected) but still walked, in case it nests protectable content.

    Args:
        node: Subtree root to walk (typically a unit's top-level AST node).
        text_raw: The unit's exact source text, for cursor-based lookups.
        line_starts: Table from `_line_starts`, for block-node offsets.
        unit_char_start: Offset of this unit's text_raw into the full document.
        cursor: Forward-moving search position into text_raw.

    Returns:
        (spans, cursor) — spans found in document order, and the cursor
        position after the last one found.
    """
    spans: list[ProtectedSpan] = []
    for child in node.children or []:
        if child.type in _INLINE_PROTECT_KINDS:
            needle = f"{child.markup}{child.content}{child.markup}"
            # CommonMark normalizes code-span content (line endings -> spaces,
            # a single leading/trailing space stripped), so markup+content+
            # markup doesn't always reconstruct an exact substring of the
            # original text_raw (observed on real data: a lone literal "`"
            # character paired with a distant backtick elsewhere produced
            # content that didn't match). Skip protecting this one node
            # rather than crash the whole unit over an unprotected span.
            idx = text_raw.find(needle, cursor)
            if idx != -1:
                spans.append(
                    ProtectedSpan(
                        placeholder="",
                        original=needle,
                        kind=_INLINE_PROTECT_KINDS[child.type],
                        start=idx,
                        end=idx + len(needle),
                    )
                )
                cursor = idx + len(needle)
        elif child.type in _BLOCK_PROTECT_KINDS and child.map is not None:
            block_start, block_end = _node_span(child, line_starts)
            start = block_start - unit_char_start
            end = block_end - unit_char_start
            spans.append(
                ProtectedSpan(
                    placeholder="",
                    original=text_raw[start:end],
                    kind=_BLOCK_PROTECT_KINDS[child.type],
                    start=start,
                    end=end,
                )
            )
            cursor = max(cursor, end)
        else:
            if child.type == "link_open":
                href_attr = child.attrGet("href")
                href = str(href_attr) if href_attr is not None else ""
                # Reference-style links ([text][ref]) or percent-encoding
                # normalization mean href doesn't always appear verbatim in
                # text_raw — skip protecting it rather than crash the unit.
                idx = text_raw.find(href, cursor) if href else -1
                if idx != -1:
                    spans.append(
                        ProtectedSpan(
                            placeholder="",
                            original=href,
                            kind="link_url",
                            start=idx,
                            end=idx + len(href),
                        )
                    )
                    cursor = idx + len(href)
            child_spans, cursor = _find_protectable_nodes(
                child, text_raw, line_starts, unit_char_start, cursor
            )
            spans.extend(child_spans)
    return spans, cursor


def _find_number_spans(
    text_raw: str, existing: list[ProtectedSpan]
) -> list[ProtectedSpan]:
    """Finds standalone numbers outside of already-protected spans.

    Args:
        text_raw: The unit's exact source text.
        existing: Spans already found (math/code/link), used to skip ranges
            that are already protected.

    Returns:
        Additional ProtectedSpan entries for numbers, in document order.
    """
    spans = []
    for match in _NUMBER_RE.finditer(text_raw):
        start, end = match.span()
        if any(start < e.end and end > e.start for e in existing):
            continue
        spans.append(
            ProtectedSpan(
                placeholder="",
                original=match.group(0),
                kind="number",
                start=start,
                end=end,
            )
        )
    return spans


def protect_unit(
    node: SyntaxTreeNode,
    text_raw: str,
    line_starts: list[int],
    unit_char_start: int,
) -> tuple[str, list[ProtectedSpan]]:
    """Replaces protectable spans in a unit's text with numbered placeholders.

    Args:
        node: The unit's top-level AST node.
        text_raw: The unit's exact source text.
        line_starts: Table from `_line_starts`, for block-node offsets.
        unit_char_start: Offset of this unit's text_raw into the full document.

    Returns:
        (text_protected, spans) — text_raw with each span replaced by its
        placeholder token (e.g. "⟦0⟧"), and the finalized spans (placeholder
        ids assigned in left-to-right document order).
    """
    node_spans, _ = _find_protectable_nodes(node, text_raw, line_starts, unit_char_start, 0)
    spans = sorted(node_spans + _find_number_spans(text_raw, node_spans), key=lambda s: s.start)

    for i, span in enumerate(spans):
        span.placeholder = f"⟦{i}⟧"

    text_protected = []
    cursor = 0
    for span in spans:
        text_protected.append(text_raw[cursor:span.start])
        text_protected.append(span.placeholder)
        cursor = span.end
    text_protected.append(text_raw[cursor:])
    return "".join(text_protected), spans


_ENCODING = tiktoken.get_encoding("cl100k_base")
_HEADING_TAG_RE = re.compile(r"^h([1-6])$")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")
_NEST_DEPTH_TYPES = {"bullet_list", "ordered_list", "blockquote"}


def _count_tokens(text: str) -> int:
    """Counts tokens via the cl100k_base encoding (a stable proxy, independent
    of any particular teacher model's tokenizer).

    Args:
        text: Text to count.

    Returns:
        Token count.
    """
    return len(_ENCODING.encode(text))


def _count_sentences(text: str) -> int:
    """Counts sentences via terminal punctuation, as a cheap fingerprint signal.

    Args:
        text: Text to count.

    Returns:
        Sentence count (0 for empty/whitespace-only text).
    """
    if not text.strip():
        return 0
    return max(1, len(_SENTENCE_END_RE.findall(text)))


def _find_heading_level(node: SyntaxTreeNode) -> int | None:
    """Finds the first heading level anywhere in a subtree, depth-first.

    Args:
        node: Subtree root to search.

    Returns:
        Heading level 1-6, or None if the subtree contains no heading.
    """
    if node.type == "heading" and node.tag:
        match = _HEADING_TAG_RE.match(node.tag)
        if match:
            return int(match.group(1))
    for child in node.children or []:
        level = _find_heading_level(child)
        if level is not None:
            return level
    return None


def _max_nest_depth(node: SyntaxTreeNode, depth: int = 0) -> int:
    """Finds the deepest list/blockquote nesting depth in a subtree.

    Args:
        node: Subtree root to search.
        depth: Current depth (for recursion).

    Returns:
        Maximum nesting depth found.
    """
    deepest = depth
    for child in node.children or []:
        child_depth = depth + 1 if child.type in _NEST_DEPTH_TYPES else depth
        deepest = max(deepest, _max_nest_depth(child, child_depth))
    return deepest


def _count_node_type(node: SyntaxTreeNode, node_type: str) -> int:
    """Counts occurrences of a node type anywhere in a subtree.

    Args:
        node: Subtree root to search.
        node_type: The `node.type` value to count.

    Returns:
        Number of matching nodes.
    """
    count = 1 if node.type == node_type else 0
    for child in node.children or []:
        count += _count_node_type(child, node_type)
    return count


def _compute_fingerprint(
    node: SyntaxTreeNode, text_raw: str, spans: list[ProtectedSpan]
) -> Fingerprint:
    """Computes a unit's structural fingerprint from its AST subtree.

    Args:
        node: The unit's top-level AST node (or a synthetic root wrapping
            several, for a merged unit).
        text_raw: The unit's exact source text.
        spans: The unit's protected spans.

    Returns:
        A Fingerprint capturing counts a downstream verifier can check
        against the translated text without an embedding call.
    """
    return Fingerprint(
        bullet_count=_count_node_type(node, "list_item"),
        max_nest_depth=_max_nest_depth(node),
        bold_span_count=_count_node_type(node, "strong_open"),
        italic_span_count=_count_node_type(node, "em_open"),
        heading_level=_find_heading_level(node),
        placeholder_ids=sorted(span.placeholder for span in spans),
        sentence_count=_count_sentences(text_raw),
        ends_with_colon=text_raw.rstrip().endswith(":"),
    )


def _build_unit(
    node: SyntaxTreeNode,
    text: str,
    line_starts: list[int],
    doc_id: str,
    source: str,
    index: int,
    char_start_override: int | None = None,
    char_end_override: int | None = None,
) -> Unit:
    """Builds a single Unit from one top-level AST node.

    Args:
        node: The top-level AST node for this unit.
        text: The full source document (unit text is sliced from this).
        line_starts: Table from `_line_starts`.
        doc_id: Identifier of the parent document.
        source: "openthoughts" | "naturalreasoning".
        index: This unit's contiguous index within the document.
        char_start_override: If given, use this instead of the node's own
            span start — used to pull the very first unit back to 0, since
            a block node's `.map` excludes leading blank lines and those
            would otherwise be lost (no preceding unit's gap covers them).
        char_end_override: Same, for the very last unit and trailing blank
            lines at the end of the document.

    Returns:
        The assembled Unit, protected and fingerprinted.
    """
    kind, translate = _NODE_KIND.get(node.type, (node.type, True))
    char_start, char_end = _node_span(node, line_starts)
    if char_start_override is not None:
        char_start = char_start_override
    if char_end_override is not None:
        char_end = char_end_override
    text_raw = text[char_start:char_end]
    if translate:
        text_protected, spans = protect_unit(node, text_raw, line_starts, char_start)
    else:
        text_protected, spans = text_raw, []
    return Unit(
        unit_id=f"{doc_id}:{index:04d}",
        doc_id=doc_id,
        source=source,
        index=index,
        kind=kind,
        text_raw=text_raw,
        text_protected=text_protected,
        spans=spans,
        char_start=char_start,
        char_end=char_end,
        translate=translate,
        token_count=_count_tokens(text_raw),
        fingerprint=_compute_fingerprint(node, text_raw, spans),
    )


def _absorb_code_islands(units: list[Unit]) -> list[Unit]:
    """Reclassifies a translatable unit sandwiched between two non-translatable
    code units as code itself, not translatable prose.

    A stray unfenced fragment (e.g. a bare "else:" line) can end up as its
    own "paragraph" node purely because the source document's code fencing
    is inconsistent — if a fence appears immediately before AND after it,
    it's almost certainly part of the same code, not real prose, regardless
    of what the parser classified it as.

    Args:
        units: Unmerged Units in document order, from the per-node build loop.

    Returns:
        Units with any such code-island unit reclassified to translate=False.
    """
    result = list(units)
    for i in range(1, len(result) - 1):
        unit = result[i]
        if not unit.translate or result[i - 1].translate or result[i + 1].translate:
            continue
        if result[i - 1].kind != "code" or result[i + 1].kind != "code":
            continue
        result[i] = replace(unit, kind="code", translate=False, text_protected=unit.text_raw, spans=[])
    return result


def build_units(text: str, doc_id: str, source: str) -> list[Unit]:
    """Segments a document's top-level AST nodes into unmerged, unsplit Units.

    Args:
        text: The source document.
        doc_id: Identifier of the document.
        source: "openthoughts" | "naturalreasoning".

    Returns:
        One Unit per top-level markdown block, in document order.
    """
    tree, line_starts = _top_level_nodes(text)
    children = tree.children
    last = len(children) - 1
    units = [
        _build_unit(
            node,
            text,
            line_starts,
            doc_id,
            source,
            index,
            char_start_override=0 if index == 0 else None,
            char_end_override=len(text) if index == last else None,
        )
        for index, node in enumerate(children)
    ]
    return _absorb_code_islands(units)


_SHORT_HEADER_TOKENS = 8


def _is_lead_in_pair(first: Unit, second: Unit) -> bool:
    """Whether `first` is a paragraph ending in ':' immediately followed by
    a list `second` — the bullets are grammatically dependent on the lead-in.
    """
    return (
        first.kind == "paragraph"
        and second.kind == "list"
        and first.text_raw.rstrip().endswith(":")
    )


def _is_short_header_pair(first: Unit, second: Unit, min_tokens: int) -> bool:
    """Whether `first` is a short heading immediately followed by a short
    first paragraph `second`, which should read as one unit.
    """
    return (
        first.kind == "heading"
        and second.kind == "paragraph"
        and first.token_count < _SHORT_HEADER_TOKENS
        and second.token_count < min_tokens
    )


def _merge_group(group: list[Unit], text: str) -> Unit:
    """Merges consecutive sibling Units into one, re-parsing and re-protecting
    the merged span as a standalone document fragment.

    Args:
        group: Two or more consecutive Units to merge (all translate=True).
        text: The full source document the units were sliced from.

    Returns:
        A single merged Unit; `merged_from` records the original kinds.
    """
    first, last = group[0], group[-1]
    char_start, char_end = first.char_start, last.char_end
    text_raw = text[char_start:char_end]
    sub_tree, sub_line_starts = _top_level_nodes(text_raw)
    text_protected, spans = protect_unit(sub_tree, text_raw, sub_line_starts, 0)
    return Unit(
        unit_id=first.unit_id,
        doc_id=first.doc_id,
        source=first.source,
        index=first.index,
        kind=first.kind,
        text_raw=text_raw,
        text_protected=text_protected,
        spans=spans,
        char_start=char_start,
        char_end=char_end,
        translate=True,
        token_count=_count_tokens(text_raw),
        fingerprint=_compute_fingerprint(sub_tree, text_raw, spans),
        merged_from=[unit.kind for unit in group],
    )


def merge_units(units: list[Unit], text: str, min_tokens: int) -> list[Unit]:
    """Merges dependent sibling nodes per the chunking spec's Step 3:
    lead-in + list, short header + first paragraph, and consecutive short
    paragraphs. Never merges across code/math/rule units, since those never
    match a merge trigger's kind checks.

    Args:
        units: Unmerged Units in document order, from `build_units`.
        text: The full source document the units were sliced from.
        min_tokens: Token threshold below which paragraphs are considered
            short enough to merge.

    Returns:
        Units with dependent siblings merged, reindexed contiguously from 0.
    """
    groups: list[list[Unit]] = []
    i = 0
    while i < len(units):
        current = units[i]
        j = i + 1
        if j < len(units) and _is_lead_in_pair(current, units[j]):
            groups.append([current, units[j]])
            i = j + 1
            continue
        if j < len(units) and _is_short_header_pair(current, units[j], min_tokens):
            groups.append([current, units[j]])
            i = j + 1
            continue
        if current.kind == "paragraph" and current.token_count < min_tokens:
            group = [current]
            total = current.token_count
            k = j
            while (
                k < len(units)
                and units[k].kind == "paragraph"
                and units[k].token_count < min_tokens
                and total < min_tokens
            ):
                group.append(units[k])
                total += units[k].token_count
                k += 1
            groups.append(group)
            i = k
            continue
        groups.append([current])
        i = j

    groups = _absorb_orphan_groups(groups, min_tokens)
    merged = [_merge_group(g, text) if len(g) > 1 else g[0] for g in groups]
    for index, unit in enumerate(merged):
        unit.index = index
        unit.unit_id = f"{unit.doc_id}:{index:04d}"
    return merged


def _absorb_orphan_groups(groups: list[list[Unit]], min_tokens: int) -> list[list[Unit]]:
    """Second merge pass: folds a standalone, sub-`min_tokens` translatable
    group into an adjacent translatable group (previous preferred, else
    next), regardless of kind mismatch.

    The kind-specific rules above (lead-in+list, short-header+paragraph,
    consecutive short paragraphs) only merge specific kind pairs and only
    ever look forward — a short fragment next to a `list`, or one whose
    forward neighbor doesn't itself qualify, falls through every rule and
    ships alone (e.g. a lone "AND" between two list units, or "Hmm."
    followed by a paragraph that isn't itself short). This pass catches
    what's left over, using the same `min_tokens` floor as the rest of
    Step 3 rather than a separate threshold.

    Args:
        groups: Groups from the main merge pass, each a list of one or more
            consecutive Units to be flattened into a single merged Unit.
        min_tokens: Token threshold below which a standalone unit is
            considered too short to ship on its own.

    Returns:
        Groups with remaining orphans folded into a translatable neighbor,
        or left alone if no translatable neighbor exists at all (e.g. a
        fragment sandwiched between two code fences — see
        `_absorb_code_islands`, which handles that case earlier).
    """
    result: list[list[Unit]] = []
    i = 0
    n = len(groups)
    while i < n:
        group = groups[i]
        is_orphan = len(group) == 1 and group[0].translate and group[0].token_count < min_tokens
        if not is_orphan:
            result.append(group)
            i += 1
            continue
        if result and result[-1][-1].translate:
            result[-1] = result[-1] + group
            i += 1
        elif i + 1 < n and groups[i + 1][0].translate:
            result.append(group + groups[i + 1])
            i += 2
        else:
            result.append(group)
            i += 1
    return result


_SPLIT_RATE_WARN_THRESHOLD = 0.15
_SAT_MODEL_NAME = "sat-3l-sm"
_sat_model = None
_sat_lock = threading.Lock()
_SAT_DEVICE = "cpu"


def set_sat_device(device: str) -> None:
    """Sets the device wtpsplit's SaT model loads onto on first use.

    CPU inference for this model is ~20-40x slower than GPU for the same
    input (measured: a batch of split-heavy documents that took 3+ minutes
    on CPU took well under 1s/doc on GPU) — worth setting explicitly to a
    free GPU for any batch run, since split_units is on the hot path for
    every oversized unit. Must be called before the first split-triggering
    chunk_document() call; has no effect once the model is already loaded.

    Args:
        device: e.g. "cpu", "cuda:0".
    """
    global _SAT_DEVICE
    _SAT_DEVICE = device


def _get_sat_model():
    """Lazily loads and caches the wtpsplit SaT sentence-splitting model, so
    importing this module doesn't pay the model-load cost unless a paragraph
    actually needs splitting.

    Guarded by a real (non-async) lock: `chunk_document` can be invoked from
    many `asyncio.to_thread` calls concurrently, so without this, multiple
    threads could race to construct the model on its first use (same class
    of bug fixed for the LaBSE model in translate_reasoning.py).

    Returns:
        A loaded `wtpsplit.SaT` instance, moved to `_SAT_DEVICE` if it isn't "cpu".
    """
    global _sat_model
    if _sat_model is None:
        with _sat_lock:
            if _sat_model is None:
                from wtpsplit import SaT

                model = SaT(_SAT_MODEL_NAME)
                if _SAT_DEVICE != "cpu":
                    model.to(_SAT_DEVICE)
                _sat_model = model
    return _sat_model


def _sentence_offsets(text: str) -> list[tuple[int, int]]:
    """Splits text into sentences via wtpsplit SaT and returns their offsets.

    Args:
        text: Text to split. SaT's segments normally concatenate back to
            `text` exactly, so offsets are derived by cumulative length
            rather than a search; a trailing shortfall (SaT can drop
            trailing whitespace) is folded into the last offset so the
            returned offsets always cover `text` exactly. Zero-length
            pieces are dropped.

    Returns:
        (start, end) offsets covering `text` contiguously and completely,
        one per non-empty sentence.
    """
    cursor = 0
    offsets = []
    for piece in _get_sat_model().split(text):
        if piece:
            offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    if offsets and offsets[-1][1] < len(text):
        offsets[-1] = (offsets[-1][0], len(text))
    elif not offsets and len(text) > 0:
        offsets.append((0, len(text)))
    return offsets


def _fix_sentence_boundaries(
    offsets: list[tuple[int, int]], spans: list[ProtectedSpan]
) -> list[tuple[int, int]]:
    """Merges sentence boundaries that fall inside a protected span, so a
    later split never cuts through protected content.

    Args:
        offsets: Sentence (start, end) offsets from `_sentence_offsets`.
        spans: The unit's protected spans (offsets in the same coordinate
            space as `offsets`).

    Returns:
        Offsets with any span-violating boundaries merged away.
    """
    if not spans or not offsets:
        return offsets
    fixed = [offsets[0]]
    for start, end in offsets[1:]:
        boundary = fixed[-1][1]
        if any(span.start < boundary < span.end for span in spans):
            fixed[-1] = (fixed[-1][0], end)
        else:
            fixed.append((start, end))
    return fixed


def _group_by_token_budget(
    text: str, offsets: list[tuple[int, int]], max_tokens: int
) -> list[tuple[int, int]]:
    """Greedily groups contiguous (start, end) offsets so each group's token
    count stays within `max_tokens` where possible.

    Args:
        text: The text the offsets index into.
        offsets: Contiguous (start, end) offsets covering `text`.
        max_tokens: Token budget per group.

    Returns:
        Merged (start, end) groups, each ideally <= max_tokens tokens (a
        single oversized piece that alone exceeds the budget is kept whole).
    """
    groups = [offsets[0]]
    group_tokens = _count_tokens(text[offsets[0][0] : offsets[0][1]])
    for start, end in offsets[1:]:
        piece_tokens = _count_tokens(text[start:end])
        if group_tokens > 0 and group_tokens + piece_tokens > max_tokens:
            groups.append((start, end))
            group_tokens = piece_tokens
        else:
            groups[-1] = (groups[-1][0], end)
            group_tokens += piece_tokens
    return groups


def _rebuild_unit(unit: Unit, start: int, end: int, split_index: int) -> Unit:
    """Builds a split-off sub-unit from a slice of a unit's text_raw,
    re-parsing and re-protecting the slice as a standalone fragment.

    Args:
        unit: The oversized unit being split.
        start: Slice start offset into `unit.text_raw`.
        end: Slice end offset into `unit.text_raw`.
        split_index: This piece's position within the split group.

    Returns:
        A new Unit covering `unit.text_raw[start:end]`.
    """
    text_raw = unit.text_raw[start:end]
    sub_tree, sub_line_starts = _top_level_nodes(text_raw)
    text_protected, spans = protect_unit(sub_tree, text_raw, sub_line_starts, 0)
    return Unit(
        unit_id=unit.unit_id,
        doc_id=unit.doc_id,
        source=unit.source,
        index=unit.index,
        kind=unit.kind,
        text_raw=text_raw,
        text_protected=text_protected,
        spans=spans,
        char_start=unit.char_start + start,
        char_end=unit.char_start + end,
        translate=True,
        token_count=_count_tokens(text_raw),
        fingerprint=_compute_fingerprint(sub_tree, text_raw, spans),
        merged_from=unit.merged_from,
        split_index=split_index,
    )


def _split_paragraph(unit: Unit, max_tokens: int) -> list[Unit]:
    """Splits an oversized paragraph at sentence boundaries via wtpsplit SaT,
    never cutting inside a protected span.

    Args:
        unit: The oversized paragraph unit.
        max_tokens: Token budget per resulting piece.

    Returns:
        Split sub-units, or [unit] unchanged if it's already a single
        sentence (nothing to split on).
    """
    offsets = _fix_sentence_boundaries(_sentence_offsets(unit.text_raw), unit.spans)
    groups = _group_by_token_budget(unit.text_raw, offsets, max_tokens)
    if len(groups) <= 1:
        return [unit]
    return [
        _rebuild_unit(unit, start, end, split_index)
        for split_index, (start, end) in enumerate(groups)
    ]


def _split_list(unit: Unit, max_tokens: int) -> list[Unit]:
    """Splits an oversized list at top-level list_item boundaries, grouping
    consecutive items to stay within `max_tokens`. Never orphans a nested
    sub-list, since each list_item's span already covers its full subtree.

    Args:
        unit: The oversized list unit.
        max_tokens: Token budget per resulting piece.

    Returns:
        Split sub-units, or [unit] unchanged if it can't be split further
        (a single item, or a merged lead-in + list unit — splitting a merged
        unit would orphan the lead-in, so it's left oversized instead).
    """
    if unit.merged_from is not None:
        logger.warning(
            "split: unit {} is a merged lead-in + list; splitting would "
            "orphan the lead-in, emitting oversized ({} tokens > {})",
            unit.unit_id,
            unit.token_count,
            max_tokens,
        )
        return [unit]

    tree, line_starts = _top_level_nodes(unit.text_raw)
    list_node = tree.children[0]
    items = [child for child in list_node.children if child.type == "list_item"]
    if len(items) <= 1:
        logger.warning(
            "split: unit {} has a single list_item, cannot split further, "
            "emitting oversized ({} tokens > {})",
            unit.unit_id,
            unit.token_count,
            max_tokens,
        )
        return [unit]

    item_spans = [_node_span(item, line_starts) for item in items]
    offsets = _group_by_token_budget(unit.text_raw, item_spans, max_tokens)
    if len(offsets) <= 1:
        return [unit]
    return [
        _rebuild_unit(unit, start, end, split_index)
        for split_index, (start, end) in enumerate(offsets)
    ]


def split_units(units: list[Unit], max_tokens: int) -> list[Unit]:
    """Splits oversized units per the chunking spec's Step 4 (fallback only):
    lists split at item boundaries, tables are never split (emitted oversized
    with a warning), paragraphs split at sentence boundaries via wtpsplit.
    Logs the split rate; warns above a 15% rate (max_tokens likely too low).

    Args:
        units: Merged units, in document order, from `merge_units`.
        max_tokens: Token budget above which a unit is considered oversized.

    Returns:
        Units with oversized ones split where possible, reindexed
        contiguously from 0.
    """
    result: list[Unit] = []
    split_count = 0
    for unit in units:
        if not unit.translate or unit.token_count <= max_tokens:
            result.append(unit)
            continue
        if unit.kind == "list":
            parts = _split_list(unit, max_tokens)
        elif unit.kind == "table":
            logger.warning(
                "split: table unit {} exceeds max_tokens ({} > {}); "
                "tables are never split, emitting oversized",
                unit.unit_id,
                unit.token_count,
                max_tokens,
            )
            parts = [unit]
        elif unit.kind == "paragraph":
            parts = _split_paragraph(unit, max_tokens)
        else:
            parts = [unit]
        if len(parts) > 1:
            split_count += 1
        result.extend(parts)

    if units:
        split_rate = split_count / len(units)
        logger.info(
            "split: {}/{} units split ({:.1f}%)",
            split_count,
            len(units),
            split_rate * 100,
        )
        if split_rate > _SPLIT_RATE_WARN_THRESHOLD:
            logger.warning(
                "split: split rate {:.1f}% exceeds {:.0f}%; max_tokens={} "
                "may be too low for this corpus",
                split_rate * 100,
                _SPLIT_RATE_WARN_THRESHOLD * 100,
                max_tokens,
            )

    for index, unit in enumerate(result):
        unit.index = index
        unit.unit_id = f"{unit.doc_id}:{index:04d}"
    return result


_RECAP_PHRASES = ("in conclusion", "in summary", "to summarize", "to conclude")
_RECAP_POSITION_THRESHOLD = 0.85  # final 15% of the document
_RECAP_OVERLAP_THRESHOLD = 0.5
_DISCOURSE_MARKER_RE = re.compile(
    r"\b(wait|hold on|actually|let me reconsider|hmm|that's not right)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _word_set(text: str) -> set[str]:
    """Lowercased word set of a unit's text, for a cheap overlap heuristic."""
    return set(_WORD_RE.findall(text.lower()))


def _flag_recap_units(units: list[Unit]) -> None:
    """Flags naturalreasoning units that look like a closing recap, mutating
    `is_recap` in place. Heuristic per the spec: unit falls in the final 15%
    of the document (by cumulative token share) AND opens with a recap
    phrase ("in conclusion", "in summary", ...) AND has high word overlap
    with earlier units. Flags, doesn't delete — downstream caller decides.

    Args:
        units: A document's units, in order.
    """
    if not units:
        return
    total_tokens = sum(u.token_count for u in units) or 1
    cumulative = 0
    earlier_words: set[str] = set()
    for unit in units:
        position = cumulative / total_tokens
        cumulative += unit.token_count
        if position < _RECAP_POSITION_THRESHOLD:
            earlier_words |= _word_set(unit.text_raw)
            continue
        stripped = unit.text_raw.strip().lower()
        if not any(stripped.startswith(phrase) for phrase in _RECAP_PHRASES):
            continue
        unit_words = _word_set(unit.text_raw)
        overlap = len(unit_words & earlier_words) / len(unit_words) if unit_words else 0.0
        if overlap >= _RECAP_OVERLAP_THRESHOLD:
            unit.is_recap = True


def _flag_discourse_markers(units: list[Unit]) -> None:
    """Flags openthoughts units containing discourse markers ("wait", "hold
    on", "actually", ...), mutating `has_discourse_markers` in place. These
    need register-preserving translation instructions downstream — formal
    Hindi renders them as stiff written prose and destroys reasoning texture.

    Args:
        units: A document's units, in order.
    """
    for unit in units:
        if _DISCOURSE_MARKER_RE.search(unit.text_raw):
            unit.has_discourse_markers = True


def apply_source_flags(units: list[Unit], source: str) -> list[Unit]:
    """Applies the chunking spec's Step 6 source-specific flags in place.

    Args:
        units: A document's units, in order.
        source: "openthoughts" | "naturalreasoning".

    Returns:
        The same `units` list, for chaining.
    """
    if source == "naturalreasoning":
        _flag_recap_units(units)
    elif source == "openthoughts":
        _flag_discourse_markers(units)
    return units


def _compute_stats(units: list[Unit]) -> dict[str, Any]:
    """Computes summary stats for a ChunkedDocument.

    Args:
        units: A document's final units, in order.

    Returns:
        Dict with counts by kind, split rate, merge rate, and total
        translatable token count.
    """
    kind_counts: dict[str, int] = {}
    split_count = 0
    merge_count = 0
    translatable_tokens = 0
    for unit in units:
        kind_counts[unit.kind] = kind_counts.get(unit.kind, 0) + 1
        if unit.split_index is not None:
            split_count += 1
        if unit.merged_from is not None:
            merge_count += 1
        if unit.translate:
            translatable_tokens += unit.token_count
    total = len(units) or 1
    return {
        "kind_counts": kind_counts,
        "split_rate": split_count / total,
        "merge_rate": merge_count / total,
        "translatable_token_total": translatable_tokens,
    }


def _assert_invariants(text: str, doc: ChunkedDocument) -> None:
    """Asserts the chunking spec's hard invariants against a built document.

    Args:
        text: The original source document.
        doc: The ChunkedDocument built from `text`.

    Raises:
        AssertionError: If any invariant is violated — units overlapping or
            non-monotonic, text_raw + gaps not reproducing `text` exactly,
            or a unit's placeholders not restoring to its text_raw exactly.
    """
    reconstructed = []
    prev_end = 0
    for i, unit in enumerate(doc.units):
        assert unit.char_start >= prev_end, f"unit {unit.unit_id} overlaps the previous unit"
        assert unit.char_end >= unit.char_start, f"unit {unit.unit_id} has a negative-length span"
        reconstructed.append(unit.text_raw)
        if i < len(doc.gaps):
            reconstructed.append(doc.gaps[i])
        prev_end = unit.char_end

        restored = unit.text_protected
        seen_placeholders: set[str] = set()
        for span in unit.spans:
            assert span.placeholder not in seen_placeholders, (
                f"unit {unit.unit_id}: duplicate placeholder {span.placeholder}"
            )
            seen_placeholders.add(span.placeholder)
            restored = restored.replace(span.placeholder, span.original, 1)
        assert restored == unit.text_raw, (
            f"unit {unit.unit_id}: placeholder restoration does not equal text_raw"
        )

    assert "".join(reconstructed) == text, (
        "units' text_raw + gaps do not reproduce the input document byte-for-byte"
    )


def chunk_document(
    text: str,
    doc_id: str,
    source: str,
    min_tokens: int = 60,
    max_tokens: int = 400,
) -> ChunkedDocument:
    """Segments a document into markdown-aware translation units.

    Runs the full pipeline: node-kind segmentation, protection, dependent-
    node merging, oversized-node splitting (fallback), and source-specific
    flagging. Asserts the spec's hard invariants before returning.

    Args:
        text: The source document (English reasoning trace / response).
        doc_id: Identifier for this document.
        source: "openthoughts" | "naturalreasoning".
        min_tokens: Token threshold below which paragraphs get merged.
        max_tokens: Token threshold above which units get split (fallback).

    Returns:
        A ChunkedDocument ready for translation and later splice-back via
        `reconstruct`.
    """
    units = build_units(text, doc_id, source)
    units = merge_units(units, text, min_tokens)
    units = split_units(units, max_tokens)
    apply_source_flags(units, source)

    gaps = [
        text[units[i].char_end : units[i + 1].char_start] for i in range(len(units) - 1)
    ]

    doc = ChunkedDocument(
        doc_id=doc_id,
        source=source,
        units=units,
        gaps=gaps,
        original_length=len(text),
        stats=_compute_stats(units),
    )
    _assert_invariants(text, doc)
    return doc


_PLACEHOLDER_RE = re.compile(r"⟦\d+⟧")


def reconstruct(doc: ChunkedDocument, translations: dict[str, str]) -> str:
    """Splices translated units back into a full, byte-exact document.

    Args:
        doc: A ChunkedDocument from `chunk_document`.
        translations: Map of unit_id -> translated text_protected (with
            placeholders still present, to be restored to their original
            protected content). A unit missing from this map falls back to
            its original `text_raw` unchanged (e.g. non-translatable units,
            or units the caller chose not to translate).

    Returns:
        The full reconstructed document text.

    Raises:
        AssertionError: If a translated unit's placeholder set doesn't match
            its fingerprint's placeholder_ids (a dropped or invented
            placeholder — a lossy translation).
    """
    pieces = []
    for i, unit in enumerate(doc.units):
        translated = translations.get(unit.unit_id)
        if translated is None:
            piece = unit.text_raw
        else:
            found = sorted(set(_PLACEHOLDER_RE.findall(translated)))
            expected = sorted(set(unit.fingerprint.placeholder_ids))
            assert found == expected, (
                f"unit {unit.unit_id}: translated placeholders {found} do not "
                f"match expected {expected}"
            )
            piece = translated
            for span in unit.spans:
                piece = piece.replace(span.placeholder, span.original)
        pieces.append(piece)
        if i < len(doc.gaps):
            pieces.append(doc.gaps[i])
    return "".join(pieces)

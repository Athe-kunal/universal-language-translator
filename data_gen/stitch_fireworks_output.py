"""Stitches a completed Fireworks Batch Inference job's output back into the
final {"id", "steps": [{"en","hi"}, ...]} dataset (the same shape
reasoning_hi_train.jsonl already uses), joining on custom_id against
`translate_fireworks.py`'s units.jsonl.

Filters at the STEP level, not the document level: train_translation.py
treats each {"en","hi"} pair as an independent chat-format example (not a
sequential document continuation - see CLAUDE.md), so a document keeps its
good steps even if some of its siblings get flagged, rather than losing an
otherwise-fine document over one bad step.

A step is flagged (written to --flagged_out instead of --out) if either:
  - it dropped or altered an expected ⟦N⟧ placeholder (the model summarized/
    paraphrased instead of translating - observed on real output: some units
    get a wholly different, hallucinated response instead of a translation,
    not just a missing placeholder), or
  - its translation didn't finish naturally (finish_reason != "stop" - hit
    the request's max_tokens, i.e. truncated mid-sentence), or
  - it has no translation at all (missing from the output / in the error file).

Confirmed output row schema (not documented by Fireworks - verified against
a real completed job's BIJOutputSet.jsonl):
    {"custom_id": "<doc_id>:<0000>", "response": {"choices": [
        {"message": {"content": "<translated text, ⟦N⟧ placeholders intact>"},
         "finish_reason": "stop"}], "usage": {...}}}
A separate error-data file (JSONL, same directory) holds failed rows - see
`--error_file`.

Usage:
    uv run python -m data_gen.stitch_fireworks_output \
        --units_file fireworks_batch_30k/units.jsonl \
        --output_file fireworks_batch_30k/fireworks_output/dataset/reasoning-hi-30k-output/BIJOutputSet.jsonl \
        --error_file fireworks_batch_30k/fireworks_output/dataset/reasoning-hi-30k-output/error-data \
        --out reasoning_hi_from_fireworks.jsonl \
        --flagged_out reasoning_hi_from_fireworks_flagged.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("stitch_fireworks_output")


def restore_placeholders(text: str, spans: list[dict]) -> str:
    """Substitutes each span's placeholder back to its original content -
    same replace-based logic as chunking.reconstruct(), just operating on
    a translated unit's text directly rather than a whole document.

    Args:
        text: Translated text, with ⟦N⟧ placeholders still present.
        spans: [{"placeholder", "original"}, ...] from units.jsonl.

    Returns:
        Text with every placeholder replaced by its original content.
    """
    for span in spans:
        text = text.replace(span["placeholder"], span["original"])
    return text


def load_units(units_file: Path) -> dict[str, dict]:
    """Loads units.jsonl into a dict keyed by custom_id.

    Args:
        units_file: Path to translate_fireworks.py's units.jsonl.

    Returns:
        {custom_id: unit_row}.
    """
    units = {}
    with open(units_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            units[row["custom_id"]] = row
    return units


def load_translations(output_file: Path) -> dict[str, tuple[str, str]]:
    """Loads a Fireworks batch output file into {custom_id: (hi, finish_reason)}.

    Args:
        output_file: Path to the job's BIJOutputSet.jsonl.

    Returns:
        {custom_id: (translated_text_with_placeholders, finish_reason)}.
    """
    translations = {}
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            choice = row["response"]["choices"][0]
            translations[row["custom_id"]] = (choice["message"]["content"], choice["finish_reason"])
    return translations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True, help="The job's BIJOutputSet.jsonl.")
    parser.add_argument("--error_file", type=Path, default=None, help="The job's error-data file, if non-empty.")
    parser.add_argument("--out", type=Path, required=True, help="Clean {id, steps} dataset to write.")
    parser.add_argument(
        "--flagged_out", type=Path, required=True, help="Flagged steps (for later review/re-translation)."
    )
    args = parser.parse_args()

    log.info(f"Loading units from {args.units_file}...")
    units = load_units(args.units_file)
    log.info(f"Loaded {len(units)} units")

    log.info(f"Loading translations from {args.output_file}...")
    translations = load_translations(args.output_file)
    log.info(f"Loaded {len(translations)} translations")

    if args.error_file and args.error_file.exists() and args.error_file.stat().st_size > 0:
        n_errors = sum(1 for _ in open(args.error_file, encoding="utf-8"))
        log.warning(f"{args.error_file} has {n_errors} failed rows.")

    clean_docs: dict[str, list[tuple[int, dict]]] = {}
    flagged_rows: list[dict] = []
    n_missing = n_truncated = n_placeholder_mismatch = 0

    for custom_id, unit in units.items():
        doc_id, index_str = custom_id.rsplit(":", 1)
        index = int(index_str)

        if custom_id not in translations:
            n_missing += 1
            flagged_rows.append({"custom_id": custom_id, "doc_id": doc_id, "en": unit["en"], "reason": "missing"})
            continue

        hi_raw, finish_reason = translations[custom_id]
        expected = sorted({s["placeholder"] for s in unit["spans"]})
        found = sorted(set(p for p in expected if p in hi_raw))
        placeholder_ok = found == expected
        truncated = finish_reason != "stop"

        if not placeholder_ok:
            n_placeholder_mismatch += 1
        if truncated:
            n_truncated += 1

        if not placeholder_ok or truncated:
            reasons = [r for r, bad in (("placeholder_mismatch", not placeholder_ok), ("truncated", truncated)) if bad]
            flagged_rows.append(
                {
                    "custom_id": custom_id,
                    "doc_id": doc_id,
                    "en": unit["en"],
                    "hi_raw": hi_raw,
                    "reason": "+".join(reasons),
                }
            )
            continue

        hi = restore_placeholders(hi_raw, unit["spans"])
        clean_docs.setdefault(doc_id, []).append((index, {"en": unit["en"], "hi": hi}))

    log.info(
        f"missing={n_missing} truncated={n_truncated} placeholder_mismatch={n_placeholder_mismatch} "
        f"({100 * len(flagged_rows) / len(units):.1f}% of {len(units)} units flagged)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_clean_steps = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for doc_id, indexed_steps in clean_docs.items():
            steps = [step for _, step in sorted(indexed_steps, key=lambda pair: pair[0])]
            n_clean_steps += len(steps)
            f.write(json.dumps({"id": doc_id, "steps": steps}, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(clean_docs)} documents ({n_clean_steps} clean steps) to {args.out}")

    args.flagged_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.flagged_out, "w", encoding="utf-8") as f:
        for row in flagged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(flagged_rows)} flagged steps to {args.flagged_out}")


if __name__ == "__main__":
    main()

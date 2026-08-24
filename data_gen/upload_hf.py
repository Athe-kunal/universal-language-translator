"""Uploads the EN-HI reasoning translation dataset to the HuggingFace Hub.

Reads `reasoning_translation_train.jsonl` / `reasoning_translation_val.jsonl`,
rewrites `source` from the internal short code to the full origin HF dataset
path (for provenance), and pushes to the Hub as a `DatasetDict` with a
generated dataset card.

Usage:
    uv run python -m data_gen.upload_hf
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard, DatasetCardData

SOURCE_MAP = {
    "openthoughts": "open-thoughts/OpenThoughts3-1.2M",
    "naturalreasoning": "facebook/natural_reasoning",
    "opencodereasoning": "nvidia/OpenCodeReasoning",
}

DEFAULT_TRAIN_FILE = Path("reasoning_translation_train.jsonl")
DEFAULT_VAL_FILE = Path("reasoning_translation_val.jsonl")
DEFAULT_REPO_ID = "Athekunal/english-hindi-reasoning-dataset"


def load_split(path: Path) -> list[dict]:
    """Loads one split, remapping `source` to its full origin dataset path.

    Args:
        path: Path to a `reasoning_translation_{train,val}.jsonl` file.

    Returns:
        List of records with `source` rewritten to the full HF dataset id.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["source"] = SOURCE_MAP[row["source"]]
            records.append(row)
    return records


def build_card(repo_id: str, train_counts: dict, val_counts: dict) -> str:
    """Builds the dataset card markdown body.

    Args:
        repo_id: Target Hub repo id, used in the title.
        train_counts: {full_source_name: (docs, steps)} for the train split.
        val_counts: {full_source_name: (docs, steps)} for the val split.

    Returns:
        Markdown text for the card body.
    """
    def table(counts: dict) -> str:
        lines = ["| Source | Documents | Steps |", "|---|---|---|"]
        for src, (docs, steps) in counts.items():
            lines.append(f"| `{src}` | {docs:,} | {steps:,} |")
        total_docs = sum(d for d, _ in counts.values())
        total_steps = sum(s for _, s in counts.values())
        lines.append(f"| **Total** | **{total_docs:,}** | **{total_steps:,}** |")
        return "\n".join(lines)

    return f"""# {repo_id.split('/')[-1]}

English→Hindi translation dataset for chain-of-thought reasoning, segmented
into PRM-style reasoning steps. Built for training a masked diffusion language
model (MDLM) to translate reasoning traces step-by-step.

## Sources

Reasoning traces were sampled from three upstream datasets and translated
English→Hindi at the reasoning-step level (each step: one coherent chunk of a
chain-of-thought, bounded by heading/discourse-marker/semantic-break
detection). The `source` field on every document records exactly which
upstream dataset it came from.

- [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) — diverse-domain CoT
- [`facebook/natural_reasoning`](https://huggingface.co/datasets/facebook/natural_reasoning) — general reasoning
- [`nvidia/OpenCodeReasoning`](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) — code reasoning

## Splits

**Train**

{table(train_counts)}

**Validation**

{table(val_counts)}

Split is deterministic per-document (md5 hash of `doc_id`), so a document's
steps never cross the train/val boundary.

## Schema

Each row is one document:

```json
{{
  "doc_id": "md5 hash of the original English question, stable across reruns",
  "source": "full HF dataset id the document was sampled from",
  "question": "original English question/prompt",
  "num_steps": "int, number of reasoning steps",
  "steps": [
    {{
      "step_index": "int, 0-based order within the document",
      "boundary_reason": "why this step starts here (heading / discourse marker / semantic break / token budget)",
      "token_count": "int, English token count for this step",
      "en": "English step text",
      "hi": "Hindi translation of this step",
      "has_missing_translation": "bool, true if translation was unavailable for this step"
    }}
  ]
}}
```

Math, code, numbers, and links inside each step are protected during
translation and reinserted as `<placeholder-N>` tags in both `en` and `hi` —
a translation model is expected to reproduce the tag verbatim in its Hindi
output, and the original content is spliced back in afterward.
"""


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_records = load_split(args.train_file)
    val_records = load_split(args.val_file)

    def counts(records: list[dict]) -> dict:
        c: dict[str, list[int]] = {}
        for r in records:
            entry = c.setdefault(r["source"], [0, 0])
            entry[0] += 1
            entry[1] += r["num_steps"]
        return {k: tuple(v) for k, v in c.items()}

    train_counts = counts(train_records)
    val_counts = counts(val_records)

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "validation": Dataset.from_list(val_records),
        }
    )

    print(f"Pushing to {args.repo_id} (private={args.private})...")
    ds.push_to_hub(args.repo_id, private=args.private)

    card_data = DatasetCardData(
        language=["en", "hi"],
        task_categories=["translation"],
        pretty_name="English-Hindi Reasoning Translation Dataset",
    )
    card = DatasetCard.from_template(
        card_data,
        template_str="{{ body }}",
        body=build_card(args.repo_id, train_counts, val_counts),
    )
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print("Done.")


if __name__ == "__main__":
    main()

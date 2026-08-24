"""Spot-checks translated units via multilingual embedding similarity.

A QA aid, not a pipeline gate: cosine similarity between a unit's English
text_protected and its Hindi translation via the same embedding model the
pipeline validates translations with (`data_gen.embeddings`, currently
jina-embeddings-v3 — see `reward_metric_experiment.md` for why it replaced
LaBSE). Flags the lowest-scoring units for manual review — it does not retry
or reject anything itself.

Caveats (see the chunking-fallback design discussion this was born from):
- Similarity is pooled over the whole unit, so a single dropped clause inside
  an otherwise-good multi-sentence unit is diluted, not reliably caught.
  Smaller units make this signal more sensitive.
- A subtly wrong translation (flipped sign, swapped number in free-text
  prose, wrong word for a specific term) can still score high similarity,
  since embedding similarity tracks topic/theme, not precise facts. Treat
  low scores as "worth a human look," not as ground truth on correctness,
  and expect real, complete translations to still score well below 1.0.

Usage:
    uv run python -m data_gen.quality_check --units_file translated_reasoning_units.jsonl --bottom_n 30
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from data_gen.embeddings import DEFAULT_EMBEDDING_MODEL, get_embedding_model

DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL
DEFAULT_BATCH_SIZE = 64


def load_units(units_file: Path) -> list[dict]:
    """Loads translated per-unit records, skipping ones with no translation.

    Args:
        units_file: Path to a `translated_reasoning_units.jsonl`-style file.

    Returns:
        Records with both `en` and a non-null `hi`.
    """
    records = []
    with open(units_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("hi"):
                records.append(record)
    return records


def compute_similarities(records: list[dict], model, batch_size: int = DEFAULT_BATCH_SIZE) -> list[float]:
    """Computes cosine similarity between each record's `en` and `hi` text.

    Args:
        records: Unit records with `en` and `hi` fields.
        model: A loaded SentenceTransformer (e.g. jina-embeddings-v3).
        batch_size: Encoding batch size.

    Returns:
        Cosine similarities, one per record, same order as `records`.
    """
    en_texts = [r["en"] for r in records]
    hi_texts = [r["hi"] for r in records]
    encode_kwargs = {"batch_size": batch_size, "normalize_embeddings": True, "show_progress_bar": True}
    try:
        en_emb = model.encode(en_texts, task="text-matching", **encode_kwargs)
        hi_emb = model.encode(hi_texts, task="text-matching", **encode_kwargs)
    except TypeError:
        # Models without task-specific LoRA heads (e.g. LaBSE) don't accept `task`.
        en_emb = model.encode(en_texts, **encode_kwargs)
        hi_emb = model.encode(hi_texts, **encode_kwargs)
    return (en_emb * hi_emb).sum(axis=1).tolist()


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units_file", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the embedding model (default: cpu, to avoid contending with any GPU "
        "already busy serving the translation model).",
    )
    parser.add_argument(
        "--bottom_n", type=int, default=30, help="Print the N lowest-similarity units for manual review."
    )
    parser.add_argument(
        "--flagged_output_file",
        type=Path,
        default=None,
        help="Optional path to write the bottom_n flagged records as JSON, for later review.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_units(args.units_file)
    logger.info(f"Loaded {len(records)} translated units from {args.units_file}")

    logger.info(f"Loading {args.model} on {args.device}...")
    model = get_embedding_model(args.model, args.device)

    similarities = compute_similarities(records, model, args.batch_size)
    for record, sim in zip(records, similarities):
        record["embedding_similarity"] = sim

    ranked = sorted(records, key=lambda r: r["embedding_similarity"])
    sims_sorted = [r["embedding_similarity"] for r in ranked]
    n = len(sims_sorted)
    logger.info(
        f"similarity stats: min={sims_sorted[0]:.3f} p10={sims_sorted[n // 10]:.3f} "
        f"median={sims_sorted[n // 2]:.3f} p90={sims_sorted[9 * n // 10]:.3f} max={sims_sorted[-1]:.3f}"
    )

    bottom = ranked[: args.bottom_n]
    print(f"\n--- {len(bottom)} lowest-similarity units (for manual review, not auto-rejected) ---")
    for r in bottom:
        print(f"\n[{r['doc_id'][:10]} / {r['unit_id']}] sim={r['embedding_similarity']:.3f}")
        print("EN:", r["en"][:300])
        print("HI:", r["hi"][:300])

    if args.flagged_output_file:
        args.flagged_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.flagged_output_file, "w", encoding="utf-8") as f:
            json.dump(bottom, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote {len(bottom)} flagged records to {args.flagged_output_file}")


if __name__ == "__main__":
    main()

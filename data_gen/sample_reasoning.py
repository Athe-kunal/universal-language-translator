"""Samples reasoning traces from OpenThoughts3-1.2M and natural_reasoning.

Loads `open-thoughts/OpenThoughts3-1.2M` (stratified sample by domain/source/
difficulty) and `facebook/natural_reasoning` (uniform random sample, since it
has no categorical metadata to stratify on), maps rows to the shared
TranslationDataset schema, and writes them to a combined JSONL. Resumable:
rows already present in the output file (by stable id) are skipped on rerun.

Usage:
    uv run python data_gen/sample_reasoning.py
    uv run python data_gen/sample_reasoning.py --num_openthoughts3 200 --num_natural_reasoning 200 --output_file /tmp/sample.jsonl
"""

import argparse
import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from loguru import logger

from data_gen.datamodels import TranslationDataset

OPENTHOUGHTS3_DATASET = "open-thoughts/OpenThoughts3-1.2M"
NATURAL_REASONING_DATASET = "facebook/natural_reasoning"
DEFAULT_OUTPUT_FILE = Path("sampled_reasoning.jsonl")
DEFAULT_SEED = 42
DEFAULT_NUM_OPENTHOUGHTS3 = 50_000
DEFAULT_NUM_NATURAL_REASONING = 50_000


def row_id(question: str) -> str:
    """Stable id for dedup/resume, matching translate_ds.py's problem_id convention."""
    return hashlib.md5(question.encode()).hexdigest()


def strip_think_tags(text: str) -> str:
    """Removes literal <think>/</think> tag markers, keeping all reasoning
    content between them intact — unlike translate_ds.py's strip_think()
    (which drops the whole reasoning block), here only the tag tokens are
    noise; the reasoning trace itself is the whole point of this dataset.
    """
    return text.replace("<think>", "").replace("</think>", "").strip()


def load_done(output_file: Path) -> set[str]:
    """Loads ids already written to `output_file`, so a rerun can resume.

    Args:
        output_file: Path to the (possibly not-yet-existing) output JSONL.

    Returns:
        Set of ids already present.
    """
    if not output_file.exists():
        return set()
    done = set()
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def allocate_quotas(strata_sizes: dict[Any, int], total: int) -> dict[Any, int]:
    """Proportionally allocates `total` samples across strata, largest-remainder
    method, so small strata aren't starved to zero by rounding.

    Args:
        strata_sizes: Number of rows available per stratum key.
        total: Number of samples to allocate across all strata.

    Returns:
        Map of stratum key -> quota. Sums to min(total, sum(strata_sizes.values())).
    """
    population = sum(strata_sizes.values())
    total = min(total, population)
    if population == 0:
        return {}
    raw = {key: size * total / population for key, size in strata_sizes.items()}
    quotas = {key: min(int(value), strata_sizes[key]) for key, value in raw.items()}
    shortfall = total - sum(quotas.values())
    remainders = sorted(raw, key=lambda key: raw[key] - quotas[key], reverse=True)
    for key in remainders:
        if shortfall <= 0:
            break
        if quotas[key] < strata_sizes[key]:
            quotas[key] += 1
            shortfall -= 1
    return quotas


def build_openthoughts3_strata(ds: Dataset) -> dict[tuple[Any, Any, Any], list[int]]:
    """Groups OpenThoughts3 row indices by (domain, source, difficulty).

    Args:
        ds: The loaded OpenThoughts3 dataset.

    Returns:
        Map of (domain, source, difficulty) -> list of row indices.
    """
    meta = ds.select_columns(["domain", "source", "difficulty"]).to_pandas()
    strata: dict[tuple[Any, Any, Any], list[int]] = {}
    for key, group in meta.groupby(["domain", "source", "difficulty"], sort=False):
        strata[key] = group.index.tolist()
    return strata


def sample_indices_by_strata(
    strata: dict[Any, list[int]], quotas: dict[Any, int], seed: int
) -> list[int]:
    """Samples row indices from each stratum without replacement, per quota.

    Args:
        strata: Map of stratum key -> row indices belonging to it.
        quotas: Map of stratum key -> number of indices to draw from it.
        seed: Seed for reproducible sampling.

    Returns:
        Sampled row indices, unordered across strata.
    """
    rng = random.Random(seed)
    selected: list[int] = []
    for key, indices in strata.items():
        quota = quotas.get(key, 0)
        selected.extend(rng.sample(indices, quota))
    return selected


def map_openthoughts3_row(row: dict[str, Any]) -> TranslationDataset | None:
    """Maps an OpenThoughts3 row to TranslationDataset.

    question = first 'human' turn, cot_answer = first 'gpt' turn. Any turns
    beyond that first pair are preserved in metadata (not dropped, not
    concatenated).

    Args:
        row: One row from the OpenThoughts3-1.2M dataset.

    Returns:
        A TranslationDataset row, or None if the conversation has no human
        or no gpt turn.
    """
    conversations = row["conversations"]
    human_idx = next((i for i, turn in enumerate(conversations) if turn["from"] == "human"), None)
    gpt_idx = next((i for i, turn in enumerate(conversations) if turn["from"] == "gpt"), None)
    if human_idx is None or gpt_idx is None:
        return None
    question = conversations[human_idx]["value"]
    used = {human_idx, gpt_idx}
    return TranslationDataset(
        id=row_id(question),
        question=question,
        reference_answer=None,
        cot_answer=strip_think_tags(conversations[gpt_idx]["value"]),
        metadata={
            "difficulty": row["difficulty"],
            "domain": row["domain"],
            "source": row["source"],
            "num_turns": len(conversations),
            "extra_turns": [turn for i, turn in enumerate(conversations) if i not in used],
        },
        source="openthoughts3",
    )


def map_natural_reasoning_row(row: dict[str, Any]) -> TranslationDataset | None:
    """Maps a natural_reasoning row to TranslationDataset.

    cot_answer = responses[0].response. Alternative responses are recorded
    in metadata, not concatenated or dropped.

    Args:
        row: One row from the natural_reasoning dataset.

    Returns:
        A TranslationDataset row, or None if the row has no responses.
    """
    responses = row["responses"]
    if not responses:
        return None
    question = row["question"]
    return TranslationDataset(
        id=row_id(question),
        question=question,
        reference_answer=row["reference_answer"],
        cot_answer=responses[0]["response"],
        metadata={
            "response_model": responses[0]["response_model"],
            "num_alternative_responses": len(responses) - 1,
        },
        source="natural-reasoning",
    )


def sample_openthoughts3_streaming(num_samples: int, seed: int, done: set[str]) -> list[TranslationDataset]:
    """Uniform-random single-pass reservoir sample of OpenThoughts3-1.2M via
    streaming, avoiding the ~71GB non-streaming full-dataset download/cache
    `sample_openthoughts3` requires. Not stratified — a plain unbiased
    uniform sample (Algorithm R reservoir sampling) over the full stream.

    Args:
        num_samples: Target number of sampled rows.
        seed: Seed for reproducible sampling.
        done: Ids already written to the output file, to skip on resume.

    Returns:
        Mapped TranslationDataset rows, excluding already-done ids and rows
        missing a human/gpt turn.
    """
    logger.info(f"Streaming {OPENTHOUGHTS3_DATASET} (reservoir sampling, no full-dataset download)")
    ds = load_dataset(OPENTHOUGHTS3_DATASET, split="train", streaming=True)
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, row in enumerate(ds):
        if len(reservoir) < num_samples:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < num_samples:
                reservoir[j] = row
    logger.info(f"Reservoir-sampled {len(reservoir)} rows from the stream")

    rows = []
    skipped_no_turns = 0
    skipped_done = 0
    for row in reservoir:
        mapped = map_openthoughts3_row(row)
        if mapped is None:
            skipped_no_turns += 1
        elif mapped.id in done:
            skipped_done += 1
        else:
            rows.append(mapped)
    logger.info(
        f"openthoughts3: {len(rows)} rows ready "
        f"({skipped_no_turns} missing human/gpt turn, {skipped_done} already done)"
    )
    return rows


_OT3_SHARD_COUNT = 120
_OT3_ROWS_PER_SHARD = 10_000  # 1.2M rows / 120 shards, approx


def sample_openthoughts3_shards(num_samples: int, seed: int, done: set[str]) -> list[TranslationDataset]:
    """Uniform-random sample of OpenThoughts3-1.2M by downloading a handful
    of randomly-chosen whole parquet shards, instead of streaming every row.

    OpenThoughts3-1.2M is stored as 120 roughly-equal parquet shards (~10K
    rows / ~590MB each). `sample_openthoughts3_streaming`'s reservoir
    sampling must read every one of the 1.2M rows to guarantee each had an
    equal chance of selection — measured at ~78 minutes, entirely network-
    bound. Downloading just enough whole shards to comfortably exceed
    `num_samples`, then sampling within that pool, gets a genuinely random
    subset in seconds (measured: 3 shards / 30K rows in ~14s) — at the real
    cost of not sampling from the *other* shards at all, which is only an
    unbiased sample of the full 1.2M if the shards are themselves reasonably
    shuffled (not verified here). Since this pipeline was told stratification
    isn't needed, that tradeoff is the right one.

    Args:
        num_samples: Target number of sampled rows.
        seed: Seed for reproducible shard + row selection.
        done: Ids already written to the output file, to skip on resume.

    Returns:
        Mapped TranslationDataset rows, excluding already-done ids and rows
        missing a human/gpt turn.
    """
    rng = random.Random(seed)
    needed_shards = min(_OT3_SHARD_COUNT, max(1, num_samples // _OT3_ROWS_PER_SHARD + 2))
    shard_indices = rng.sample(range(_OT3_SHARD_COUNT), needed_shards)
    data_files = [f"data/train-{i:05d}-of-{_OT3_SHARD_COUNT:05d}.parquet" for i in shard_indices]
    logger.info(f"Loading {needed_shards}/{_OT3_SHARD_COUNT} random shards of {OPENTHOUGHTS3_DATASET}")

    ds = load_dataset(OPENTHOUGHTS3_DATASET, data_files=data_files, split="train", verification_mode="no_checks")
    logger.info(f"Loaded {len(ds)} rows from selected shards")

    indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))

    rows = []
    skipped_no_turns = 0
    skipped_done = 0
    for i in indices:
        mapped = map_openthoughts3_row(ds[i])
        if mapped is None:
            skipped_no_turns += 1
        elif mapped.id in done:
            skipped_done += 1
        else:
            rows.append(mapped)
    logger.info(
        f"openthoughts3: {len(rows)} rows ready "
        f"({skipped_no_turns} missing human/gpt turn, {skipped_done} already done)"
    )
    return rows


def sample_openthoughts3(num_samples: int, seed: int, done: set[str]) -> list[TranslationDataset]:
    """Loads and stratified-samples OpenThoughts3-1.2M, mapped to TranslationDataset.

    Args:
        num_samples: Target number of sampled rows.
        seed: Seed for reproducible sampling.
        done: Ids already written to the output file, to skip on resume.

    Returns:
        Mapped TranslationDataset rows, excluding already-done ids and rows
        missing a human/gpt turn.
    """
    logger.info(f"Loading {OPENTHOUGHTS3_DATASET}")
    ds = load_dataset(OPENTHOUGHTS3_DATASET, split="train")
    logger.info(f"Loaded {len(ds)} rows; stratifying by (domain, source, difficulty)")

    strata = build_openthoughts3_strata(ds)
    quotas = allocate_quotas({key: len(indices) for key, indices in strata.items()}, num_samples)
    indices = sample_indices_by_strata(strata, quotas, seed)
    logger.info(f"Sampled {len(indices)} rows across {len(strata)} strata")

    rows = []
    skipped_no_turns = 0
    skipped_done = 0
    for i in indices:
        mapped = map_openthoughts3_row(ds[i])
        if mapped is None:
            skipped_no_turns += 1
        elif mapped.id in done:
            skipped_done += 1
        else:
            rows.append(mapped)
    logger.info(
        f"openthoughts3: {len(rows)} rows ready "
        f"({skipped_no_turns} missing human/gpt turn, {skipped_done} already done)"
    )
    return rows


def sample_natural_reasoning(num_samples: int, seed: int, done: set[str]) -> list[TranslationDataset]:
    """Loads and uniformly samples natural_reasoning, mapped to TranslationDataset.

    natural_reasoning has no categorical metadata to stratify on, so
    "balanced" here means unbiased/uniform sampling, not stratified.

    Args:
        num_samples: Target number of sampled rows.
        seed: Seed for reproducible sampling.
        done: Ids already written to the output file, to skip on resume.

    Returns:
        Mapped TranslationDataset rows, excluding already-done ids and rows
        with no responses.
    """
    logger.info(f"Loading {NATURAL_REASONING_DATASET}")
    ds = load_dataset(NATURAL_REASONING_DATASET, split="train")
    logger.info(f"Loaded {len(ds)} rows; sampling uniformly at random")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))

    rows = []
    skipped_no_responses = 0
    skipped_done = 0
    for i in indices:
        mapped = map_natural_reasoning_row(ds[i])
        if mapped is None:
            skipped_no_responses += 1
        elif mapped.id in done:
            skipped_done += 1
        else:
            rows.append(mapped)
    logger.info(
        f"natural_reasoning: {len(rows)} rows ready "
        f"({skipped_no_responses} with no responses, {skipped_done} already done)"
    )
    return rows


def write_rows(rows: list[TranslationDataset], output_file: Path) -> None:
    """Appends TranslationDataset rows to `output_file` as JSONL.

    Args:
        rows: Rows to write.
        output_file: Destination JSONL path (created, with parents, if needed).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_openthoughts3",
        type=int,
        default=DEFAULT_NUM_OPENTHOUGHTS3,
        help="Number of OpenThoughts3 rows to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--num_natural_reasoning",
        type=int,
        default=DEFAULT_NUM_NATURAL_REASONING,
        help="Number of natural_reasoning rows to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for reproducible sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to append the output JSONL to (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    done = load_done(args.output_file)
    logger.info(f"Loaded {len(done)} already-sampled ids from {args.output_file}")

    ot3_rows = sample_openthoughts3(args.num_openthoughts3, args.seed, done)
    write_rows(ot3_rows, args.output_file)
    done.update(row.id for row in ot3_rows)

    nr_rows = sample_natural_reasoning(args.num_natural_reasoning, args.seed, done)
    write_rows(nr_rows, args.output_file)

    logger.info(f"Wrote {len(ot3_rows) + len(nr_rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()

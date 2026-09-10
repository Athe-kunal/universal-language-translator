"""
Run a trained checkpoint on N reasoning-step examples from
Athekunal/english-hindi-reasoning-dataset's own held-out validation split
(reasoning_hi_val.jsonl, see data_gen/download_datasets.py --dataset reasoning_hi) and print the
English source, Hindi reference, and model prediction side by side.

Usage:
    uv run python -m validation.validate_reasoning_hi --checkpoint .models/qwen3-a2d-bd3lm-reasoning-hi/checkpoint-500 --sampler bd3lm
"""

import argparse
import json
from pathlib import Path
import random

from train_translation import _records_from_chunked_jsonl
from translate import (
    BD3LMSamplerConfig,
    MDLMSamplerConfig,
    ScriptArguments,
    estimate_max_new_tokens,
    load_pipeline,
    translate_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=".models/qwen3-a2d-bd3lm-reasoning-hi/checkpoint-final",
    )
    parser.add_argument("--jsonl_path", default="reasoning_hi_val.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reasoning_hi_validation_sample.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="BD3LM only (see dllm.core.samplers.utils.apply_repetition_penalty). "
        "1.0 is a no-op; >1.0 discourages re-picking already-committed tokens.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Examples are sorted by source length and batched (like group_by_length "
        "at train time) so a batch's shared max_new_tokens canvas stays tight for "
        "every member instead of one huge outlier forcing padding on the rest.",
    )
    parser.add_argument(
        "--max_tokens_cap",
        type=int,
        default=96,
        help="Hard ceiling passed to estimate_max_new_tokens(). The old default (96) "
        "silently truncated most reasoning-hi steps mid-sentence, since Hindi CoT "
        "steps can run past 1000 tokens - see reasoning_hi_waitfix_validation_* "
        "runs where 84/100 predictions came back at <70% of the reference's token "
        "length. Match this to training's max_length (2048) to stop truncating.",
    )
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument(
        "--sampler",
        default="bd3lm",
        choices=["mdlm", "bd3lm"],
        help="Must match how the checkpoint was trained (train_translation.py "
        "--trainer).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Only used when --sampler bd3lm; should match the block_size "
        "the checkpoint was trained with.",
    )
    args = parser.parse_args()

    print(f"Loading eval examples from {args.jsonl_path} ...")
    records = _records_from_chunked_jsonl(
        args.jsonl_path, [{"name": "reasoning", "chunks_field": "steps"}]
    )
    random.Random(args.seed).shuffle(records)
    n = min(args.n, len(records))
    examples = records[:n]
    print(f"Eval set has {len(records)} examples — sampling {n}")

    print(f"Loading model from {args.checkpoint} (sampler={args.sampler}) ...")
    model_args = ScriptArguments(model_name_or_path=args.checkpoint)
    _, tokenizer, sampler = load_pipeline(model_args, sampler_type=args.sampler)

    sources = [ex["messages"][0]["content"] for ex in examples]
    references = [ex["messages"][1]["content"] for ex in examples]

    # Sort by source token length so each batch's shared canvas size fits its
    # members tightly, then batch-translate, then unsort back to original order.
    order = sorted(range(len(sources)), key=lambda i: len(tokenizer(sources[i])["input_ids"]))
    predictions = [None] * len(sources)
    n_batches = (len(order) + args.batch_size - 1) // args.batch_size
    print(f"Translating {len(sources)} examples in {n_batches} length-sorted batches of up to {args.batch_size} ...")
    for b in range(n_batches):
        idxs = order[b * args.batch_size : (b + 1) * args.batch_size]
        batch_sources = [sources[i] for i in idxs]
        max_new_tokens = estimate_max_new_tokens(batch_sources, tokenizer, max_tokens=args.max_tokens_cap)
        if args.sampler == "bd3lm":
            config = BD3LMSamplerConfig(
                max_new_tokens=max_new_tokens,
                steps=max_new_tokens,
                temperature=args.temperature,
                remasking=args.remasking,
                block_size=args.block_size,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            config = MDLMSamplerConfig(
                max_new_tokens=max_new_tokens,
                steps=max_new_tokens,
                temperature=args.temperature,
                remasking=args.remasking,
            )
        batch_preds = translate_batch(batch_sources, tokenizer, sampler, config)
        for i, pred in zip(idxs, batch_preds):
            predictions[i] = pred
        print(f"  batch {b + 1}/{n_batches} done (max_new_tokens={max_new_tokens})")

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for i, (src, ref, pred) in enumerate(zip(sources, references, predictions), 1):
            entry = {"idx": i, "en": src, "hi_reference": ref, "hi_prediction": pred}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"\n[{i}/{n}]")
            print(f"  EN:        {src}")
            print(f"  HI (ref):  {ref}")
            print(f"  HI (pred): {pred}")

    print(f"\nSaved {n} examples to {out}")


if __name__ == "__main__":
    main()

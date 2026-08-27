"""
Run a trained checkpoint on N reasoning-step examples from
Athekunal/english-hindi-reasoning-dataset's own held-out validation split
(reasoning_hi_val.jsonl, see data_gen/download_reasoning_hi.py) and print the
English source, Hindi reference, and model prediction side by side.

Usage:
    uv run python validate_reasoning_hi.py --checkpoint .models/qwen3-a2d-bd3lm-reasoning-hi/checkpoint-500 --sampler bd3lm
"""

import argparse
import json
from pathlib import Path

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
    import random
    random.Random(args.seed).shuffle(records)
    n = min(args.n, len(records))
    examples = records[:n]
    print(f"Eval set has {len(records)} examples — sampling {n}")

    print(f"Loading model from {args.checkpoint} (sampler={args.sampler}) ...")
    model_args = ScriptArguments(model_name_or_path=args.checkpoint)
    _, tokenizer, sampler = load_pipeline(model_args, sampler_type=args.sampler)

    sources = [ex["messages"][0]["content"] for ex in examples]
    references = [ex["messages"][1]["content"] for ex in examples]

    print(f"Translating {len(sources)} examples (one at a time, adaptive length) ...")
    predictions = []
    for src in sources:
        max_new_tokens = estimate_max_new_tokens([src], tokenizer)
        if args.sampler == "bd3lm":
            config = BD3LMSamplerConfig(
                max_new_tokens=max_new_tokens,
                steps=max_new_tokens,
                temperature=args.temperature,
                remasking=args.remasking,
                block_size=args.block_size,
            )
        else:
            config = MDLMSamplerConfig(
                max_new_tokens=max_new_tokens,
                steps=max_new_tokens,
                temperature=args.temperature,
                remasking=args.remasking,
            )
        predictions.append(translate_batch([src], tokenizer, sampler, config)[0])

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

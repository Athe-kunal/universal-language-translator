"""
Run a trained checkpoint on N examples from the BPCC hin_Deva held-out
validation split (same split produced during training) and print the
English source, Hindi reference, and model prediction side by side.

Usage:
    uv run python validate_bpcc.py
    uv run python validate_bpcc.py --checkpoint .models/modernbert-chat-bpcc-translation/checkpoint-3609 --n 20
"""

import argparse
import json
from pathlib import Path

from train_translation import load_flat_translation_dataset
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
        default=".models/modernbert-chat-bpcc-translation/checkpoint-3609",
    )
    parser.add_argument("--jsonl_path", default="bpcc_hin_deva.jsonl")
    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="bpcc_validation_sample.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument(
        "--sampler",
        default="mdlm",
        choices=["mdlm", "bd3lm"],
        help="Must match how the checkpoint was trained (train_translation.py "
        "--trainer) - a bd3lm checkpoint sampled with the mdlm sampler (or "
        "vice versa) produces malformed output even though it loads fine.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Only used when --sampler bd3lm; should match the block_size "
        "the checkpoint was trained with.",
    )
    args = parser.parse_args()

    print(f"Rebuilding held-out split from {args.jsonl_path} "
          f"(eval_split={args.eval_split}, seed={args.seed}) ...")
    dataset = load_flat_translation_dataset(
        args.jsonl_path, "src", "tgt", args.eval_split, args.seed
    )
    eval_set = dataset["test"]
    n = min(args.n, len(eval_set))
    examples = eval_set.select(range(n))
    print(f"Validation set has {len(eval_set)} examples — sampling first {n}")

    print(f"Loading model from {args.checkpoint} (sampler={args.sampler}) ...")
    model_args = ScriptArguments(model_name_or_path=args.checkpoint)
    _, tokenizer, sampler = load_pipeline(model_args, sampler_type=args.sampler)

    sources = [ex["messages"][0]["content"] for ex in examples]
    references = [ex["messages"][1]["content"] for ex in examples]

    # Size the canvas per-example: this checkpoint fills unused trailing
    # positions with repetitive garbage rather than clean padding, so a
    # canvas much longer than the real translation hurts output quality.
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

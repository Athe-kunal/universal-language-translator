"""
Run a trained miles/FSDP checkpoint (converted to HF format via
.miles/tools/convert_fsdp_to_hf.py) on N examples from the BPCC RL eval set
and print the English source, Hindi reference, and model prediction side by
side, along with the same embedding-similarity reward used during training.

Unlike validate_bpcc.py (which is for the MDLM/dllm diffusion pipeline), this
is for the plain autoregressive Qwen3 checkpoint trained by
rl/run_qwen3_0_6b_bpcc_fsdp.sh - standard transformers .generate(), not a
diffusion sampler.

Usage:
    uv run python -m rl.validate_checkpoint
    uv run python -m rl.validate_checkpoint --checkpoint .rl-checkpoints/iter_0000200_hf --n 20
"""

import argparse
import asyncio
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_gen.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=".rl-checkpoints/iter_0000200_hf")
    parser.add_argument("--eval_jsonl", default="bpcc_rl_eval.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="bpcc_rl_validation_sample.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding_device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= args.n:
                break
    print(f"Loaded {len(examples)} examples from {args.eval_jsonl}")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    results = []
    for i, ex in enumerate(examples, 1):
        prompt, reference = ex["prompt"], ex["label"]
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        similarity = asyncio.run(
            embedding_similarity(prediction, reference, DEFAULT_EMBEDDING_MODEL, args.embedding_device)
        )
        results.append(
            {"idx": i, "en": prompt, "hi_reference": reference, "hi_prediction": prediction, "similarity": similarity}
        )
        print(f"\n[{i}/{len(examples)}] similarity={similarity:.3f}")
        print(f"  EN:        {prompt.split('English: ', 1)[-1]}")
        print(f"  HI (ref):  {reference}")
        print(f"  HI (pred): {prediction}")

    avg = sum(r["similarity"] for r in results) / len(results)
    print(f"\nAverage similarity over {len(results)} examples: {avg:.4f}")

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} examples to {out}")


if __name__ == "__main__":
    main()

"""Independent spot-check of a GRPO checkpoint uploaded to W&B mid-training.

Downloads the weight-only checkpoint artifact uploaded by
rl.wandb_checkpoint_callback.WandbWeightOnlyCheckpointCallback (see
rl/train_reasoning_hi_grpo.py), generates on a fixed sample of the
reasoning_hi validation set with BD3LMSampler, and scores each prediction
against its reference via the running reward server (make
rl-reward-server-up). Appends one line per checkpoint to --out so the
similarity trend across steps can be read back later.

Run with CUDA_VISIBLE_DEVICES set to the reward-server GPU so generation
doesn't contend with training (dllm's get_model always maps to the process's
first visible device):

    CUDA_VISIBLE_DEVICES=1 uv run python -m rl.spot_check_checkpoint --step 200 \\
        --wandb_run ad-finance/universal-language-translator/jw0uvr5f
"""

import argparse
import asyncio
import json
import shutil
import tempfile

from rl.reasoning_hi_grpo_data import get_reasoning_hi_dataset
from rl.reward import _embedding_similarity
from rl.reward_components import language_switch_penalty, repetition_penalty
from translate import BD3LMSamplerConfig, ScriptArguments, estimate_max_new_tokens, load_pipeline, translate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--wandb_run", required=True, help="entity/project/run_id")
    parser.add_argument("--artifact_name", default="grpo-checkpoint")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                         help="0.0 = greedy. GRPO training rollouts use TRL's default of 1.0.")
    parser.add_argument("--out", default="rl_spot_check_results.jsonl")
    return parser.parse_args()


def find_artifact_version(wandb_run: str, artifact_name: str, step: int):
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run)
    for artifact in run.logged_artifacts():
        if artifact.name.startswith(artifact_name) and artifact.metadata.get("step") == step:
            return artifact
    return None


async def score(prediction: str, reference: str) -> float:
    prediction = (prediction or "").strip()
    if not prediction or not reference:
        return 0.0
    similarity = max(0.0, await _embedding_similarity(prediction, reference))
    reward = similarity * (1.0 - repetition_penalty(prediction)) * (1.0 - language_switch_penalty(prediction))
    return max(0.0, min(1.0, reward))


def main() -> None:
    args = parse_args()

    artifact = find_artifact_version(args.wandb_run, args.artifact_name, args.step)
    if artifact is None:
        print(f"No checkpoint artifact found for step {args.step} on {args.wandb_run}")
        return

    tmp_dir = tempfile.mkdtemp(prefix=f"spot-check-step{args.step}-")
    try:
        print(f"Downloading {artifact.name} (step {args.step}) to {tmp_dir} ...")
        artifact.download(root=tmp_dir)

        eval_set = get_reasoning_hi_dataset("validation").shuffle(seed=args.seed).select(range(args.n))

        model_args = ScriptArguments(model_name_or_path=tmp_dir)
        _, tokenizer, sampler = load_pipeline(model_args, sampler_type="bd3lm")

        # Generate and score one example at a time, printing and flushing to
        # disk immediately after each - a mid-run reward-server hiccup (or
        # ctrl-C) then loses at most one example's work instead of the whole
        # batch. Runs inside a single asyncio.run() since rl.reward's httpx
        # client is a module-level singleton bound to whichever event loop
        # first used it; a fresh asyncio.run() per call breaks on the second.
        partial_path = f"{args.out}.step{args.step}.partial.jsonl"
        results = []

        async def run_all():
            with open(partial_path, "w", encoding="utf-8") as partial_f:
                for i, row in enumerate(eval_set, 1):
                    en = row["prompt"][0]["content"]
                    max_new_tokens = min(
                        args.max_new_tokens,
                        estimate_max_new_tokens([en], tokenizer, max_tokens=args.max_new_tokens),
                    )
                    config = BD3LMSamplerConfig(
                        steps=max_new_tokens,
                        max_new_tokens=max_new_tokens,
                        block_size=args.block_size,
                        temperature=args.temperature,
                        remasking="low_confidence",
                    )
                    prediction = translate_batch([en], tokenizer, sampler, config)[0]
                    reward = await score(prediction, row["hi"])
                    result = {"en": en, "hi_ref": row["hi"], "hi_pred": prediction, "reward": reward}
                    results.append(result)
                    partial_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    partial_f.flush()
                    print(f"  [{i}/{len(eval_set)}] reward={reward:.3f}  EN: {en[:60]}")

        asyncio.run(run_all())

        avg_reward = sum(r["reward"] for r in results) / len(results)
        print(f"\nStep {args.step}: avg spot-check reward over {len(results)} examples = {avg_reward:.4f}")

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": args.step, "avg_reward": avg_reward, "n": len(results), "examples": results}, ensure_ascii=False) + "\n")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

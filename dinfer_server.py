"""Standalone HTTP server for the LLaDA-MoE + dInfer translation checkpoint.

Loads .models/llada-moe-reasoning-hi-100k-waitfix/checkpoint-2000-fused once
and keeps it resident on one GPU, so app.py (and anything else) can get
translations over HTTP without paying the ~1min load + warmup cost per
process. This is the standalone counterpart to app.py's previous in-process
@st.cache_resource loading - moved out so the model survives Streamlit
restarts and isn't tied to GPU 0.

Usage:
    .venv-dinfer/bin/python dinfer_server.py --gpu 3 --port 8092

Endpoints:
    POST /translate   {"texts": [...], "gen_length": int, "repetition_penalty": float}
                       -> {"translations": [...]}
    GET  /health       -> {"status": "ok"}
"""

import argparse
import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

from dinfer import BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
from dinfer.decoding.generate_uniform import IterSmoothDiffusionLLM
from dinfer.model import LLaDAMoeModelLM

MODEL_PATH = ".models/llada-moe-reasoning-hi-100k-waitfix/checkpoint-2000-fused"
MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 64
THRESHOLD = 0.95  # best of the {0.8, 0.9, 0.95} sweep on checkpoint-2000
CONT_WEIGHT = 0.3
DEFAULT_REPETITION_PENALTY = 1.3
VOCAB_SIZE = 156896

# torch.compile(mode="reduce-overhead") captures one CUDA graph per distinct
# (batch_size, total_canvas_length) shape it sees - cheap to reuse, expensive
# (many seconds) to capture fresh. A live server can't know request shapes in
# advance like benchmark_dataset.py's offline eval set does, so instead of
# letting shapes vary freely (which was silently recompiling on nearly every
# request - the actual cause of the slowness) we round every request onto a
# small pre-warmed grid and eat the rounding waste in exchange for a hit graph
# every time after startup.
GEN_LENGTH_BUCKETS = [128, 256, 512, 1024]
BATCH_SIZE_BUCKETS = [1, 2, 4, 8]

PROMPT_TEMPLATE = (
    "<role>SYSTEM</role>detailed thinking off<|role_end|>"
    "<role>HUMAN</role>{text}<|role_end|>"
    "<role>ASSISTANT</role>"
)


def apply_repetition_penalty(logits: torch.Tensor, x: torch.Tensor, penalty: float, exclude_ids: set[int]) -> torch.Tensor:
    """Standard repetition penalty (Keskar et al. 2019 / HF's
    RepetitionPenaltyLogitsProcessor), adapted for a diffusion decoder where
    "already generated" means "already committed anywhere in the current
    sequence x" rather than "earlier in a strictly left-to-right history":
    for every token id seen in x (excluding mask/eos), divide its logit by
    `penalty` if positive or multiply by `penalty` if negative."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    exclude = torch.as_tensor(list(exclude_ids), device=x.device)
    for b in range(x.size(0)):
        seen = torch.unique(x[b])
        seen = seen[~torch.isin(seen, exclude)]
        if seen.numel() == 0:
            continue
        vals = logits[b, :, seen]
        logits[b, :, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


class RepetitionPenaltyThresholdDecoder(ThresholdParallelDecoder):
    def __init__(self, *args, repetition_penalty: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.repetition_penalty = repetition_penalty

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        logits = apply_repetition_penalty(logits, x.data, self.repetition_penalty, exclude_ids={self.mask_id, self.eos_id})
        super().decode(logits, block_start, block_end, x, iter_threshold)


class TranslateRequest(BaseModel):
    texts: list[str]
    gen_length: int = 256
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


class TranslateResponse(BaseModel):
    translations: list[str]


def load_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16)
        model = model.to(device)

        warm_x = torch.arange(50 + max(GEN_LENGTH_BUCKETS), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            model(warm_x, use_cache=False)
            model(warm_x, use_cache=True)

        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False, dynamic=True)

    return tokenizer, model


def build_dllm(model, repetition_penalty: float):
    decoder = RepetitionPenaltyThresholdDecoder(
        temperature=0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID, repetition_penalty=repetition_penalty
    )
    cache_factory = KVCacheFactory("dual", is_bd_model=False)
    return IterSmoothDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True),
        cache_factory=cache_factory,
        early_stop=True,
        cont_weight=CONT_WEIGHT,
    )


def cut_eos(seq: torch.Tensor, eos_id: int) -> torch.Tensor:
    eos_indices = (seq[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        return seq[:, : eos_indices[0].item()]
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "45710")

    device = torch.device(0)  # gpu `args.gpu` is now the only visible device -> local index 0
    torch.cuda.set_device(device)

    print(f"[dinfer_server] loading {MODEL_PATH} on GPU {args.gpu} ...")
    tokenizer, model = load_model(device)
    # Cache built dllm wrappers by repetition_penalty so repeated requests at the
    # same penalty (the common case) skip rebuilding the decoder/cache_factory.
    dllm_cache: dict[float, IterSmoothDiffusionLLM] = {}
    default_dllm = build_dllm(model, DEFAULT_REPETITION_PENALTY)
    dllm_cache[DEFAULT_REPETITION_PENALTY] = default_dllm

    print(f"[dinfer_server] warming up CUDA graphs for {len(BATCH_SIZE_BUCKETS) * len(GEN_LENGTH_BUCKETS)} "
          f"(batch_size, gen_length) buckets - this is slow (several minutes) but one-time ...")
    with torch.no_grad():
        for batch_size in BATCH_SIZE_BUCKETS:
            for gen_length in GEN_LENGTH_BUCKETS:
                dummy = torch.randint(0, VOCAB_SIZE, (batch_size, 64), dtype=torch.long, device=device)
                default_dllm.generate(dummy, gen_length=gen_length, block_length=BLOCK_LENGTH)
    print("[dinfer_server] warmup done, ready to serve")

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_PATH, "gpu": args.gpu}

    @app.post("/translate", response_model=TranslateResponse)
    @torch.no_grad()
    def translate(req: TranslateRequest):
        dllm = dllm_cache.get(req.repetition_penalty)
        if dllm is None:
            dllm = build_dllm(model, req.repetition_penalty)
            dllm_cache[req.repetition_penalty] = dllm

        prompts = [PROMPT_TEMPLATE.format(text=t) for t in req.texts]
        input_id_lists = [tokenizer(p)["input_ids"] for p in prompts]
        max_len = max(len(ids) for ids in input_id_lists)

        # Round batch_size and gen_length up onto the pre-warmed grid so this
        # call reuses a captured CUDA graph instead of triggering a fresh
        # (multi-second) torch.compile recapture.
        n_real = len(input_id_lists)
        batch_size = next((b for b in BATCH_SIZE_BUCKETS if b >= n_real), max(BATCH_SIZE_BUCKETS))
        gen_length = next((g for g in GEN_LENGTH_BUCKETS if g >= req.gen_length), max(GEN_LENGTH_BUCKETS))

        batch = torch.full((batch_size, max_len), MASK_ID, dtype=torch.long, device=device)
        for row, ids in enumerate(input_id_lists):
            batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        for row in range(n_real, batch_size):  # pad extra rows by repeating row 0 (discarded below)
            batch[row, : len(input_id_lists[0])] = batch[0, : len(input_id_lists[0])]

        out = dllm.generate(batch, gen_length=gen_length, block_length=BLOCK_LENGTH)

        translations = []
        for row, ids in enumerate(input_id_lists):  # only the n_real real rows
            answer = cut_eos(out[row, len(ids) :].unsqueeze(0), EOS_ID)[0]
            translations.append(tokenizer.decode(answer, skip_special_tokens=True).strip())
        return TranslateResponse(translations=translations)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

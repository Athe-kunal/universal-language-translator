"""Long-lived dInfer HTTP server for a translation checkpoint: fused LLaDA-MoE (--model_type llada_moe) or a dllm A2D Qwen3
MDLM (--model_type qwen3_a2d, via dinfer_integration/modeling_qwen3_a2d.py).

Loads the model once (torch.compile + CUDA-graph warmup), then serves:
    POST /translate_stream  {"texts": [...]}  ->  NDJSON, one {"idx", "hi", "seconds"} per step as its batch finishes
    POST /translate  {"texts": ["..."]}  ->  {"translations": ["..."], "seconds": 1.2}
    GET  /health

Texts of a request are grouped by the canvas they need (2.5x source tokens, as in translate.estimate_max_new_tokens;
an oversized canvas makes MDLM checkpoints duplicate text) and each group is decoded in parallel as one batch.
Run with the dInfer venv:
    CUDA_VISIBLE_DEVICES=0 .venv-dinfer/bin/python dinfer_server.py --model /mnt/.../checkpoint-2981-fused
    CUDA_VISIBLE_DEVICES=3 .venv-dinfer/bin/python dinfer_server.py --model_type qwen3_a2d --port 8093 \\
        --model .models/qwen3-a2d-mdlm-reasoning-hi-100k-waitfix/checkpoint-2975
"""

import argparse
import contextlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

from chunk_splitter import join_pieces, split_text
from dinfer import (
    BlockIteratorFactory,
    BlockWiseDiffusionLLM,
    IterSmoothWithVicinityCacheDiffusionLLM,
    KVCacheFactory,
    ThresholdParallelDecoder,
)
from dinfer.model import LLaDAMoeModelLM

PROMPT = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>{}<|role_end|><role>ASSISTANT</role>"
GEN_BUCKETS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)  # canvas sizes; a text takes the smallest one that fits 2.5x

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--model_type", choices=["llada_moe", "qwen3_a2d"], default="llada_moe")
parser.add_argument("--port", type=int, default=8092)
parser.add_argument("--threshold", type=float, default=0.95)
parser.add_argument("--cont_weight", type=float, default=None, help="iteration smoothing; default 0.3 (llada_moe) / off (qwen3_a2d)")
parser.add_argument("--max_src_tokens", type=int, default=1024, help="English tokens per decoded piece; longer steps are split")
args = parser.parse_args()
QWEN = args.model_type == "qwen3_a2d"
if args.cont_weight is None:
    args.cont_weight = 0.0 if QWEN else 0.3

torch.cuda.set_device(0)
device = torch.device(0)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(45611 + args.port % 1000), TOKENIZERS_PARALLELISM="false")

if QWEN:
    from dinfer.decoding.utils import TokenArray
    from dinfer_integration.modeling_qwen3_a2d import EOS_ID, MASK_ID, Qwen3A2DModelLM

    BLOCK = 32
    # The chat prompt contains <|im_end|>, which is also the EOS id: dInfer's batch-1 get_generated_tokens() would drop it
    # from the prompt and shift the output, so return the raw array and slice at the prompt length ourselves.
    TokenArray.get_generated_tokens = lambda self: self.data
    # The model always starts its answer with an empty think block; prefilling it saves canvas and a decode step.
    THINK = tokenizer("<think>\n\n</think>\n\n")["input_ids"]

    def prompt_ids(text: str) -> list[int]:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        return (ids["input_ids"] if isinstance(ids, dict) else ids) + THINK

    ctx = contextlib.nullcontext()
    block_start_align = False  # align=True would start the first block inside the prompt, whose <|im_end|> reads as EOS
else:
    MASK_ID, EOS_ID = 156895, 156892
    BLOCK = 64
    prompt_ids = lambda text: tokenizer(PROMPT.format(text))["input_ids"]
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")
    ctx = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True)))
    block_start_align = True

with torch.no_grad(), ctx:
    if QWEN:
        model = Qwen3A2DModelLM.from_pretrained(args.model, dtype=torch.bfloat16, device=device)
    else:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=config).eval()
        model.load_weights(args.model, torch_dtype=torch.bfloat16)
        model = model.to(device)
        x = torch.arange(50 + 128, dtype=torch.long, device=device).unsqueeze(0)
        model(x, use_cache=False)
        model(x, use_cache=True)
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False, dynamic=True)

    decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=MASK_ID, eos_id=EOS_ID)
    cache_factory = KVCacheFactory("dual", is_bd_model=False)
    iterator = BlockIteratorFactory(start_block_align=block_start_align)
    if args.cont_weight > 0:
        dllm = IterSmoothWithVicinityCacheDiffusionLLM(
            model, decoder, iterator, cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight,
            prefix_look=16, after_look=16, warmup_steps=4,
        )
    else:
        dllm = BlockWiseDiffusionLLM(model, decoder, iterator, cache_factory=cache_factory, early_stop=True)

    MAX_BATCH = 32  # steps decoded together in one generate() call
    TOKEN_BUDGET = 16384  # batch size * canvas tokens per generate() call

    def translate_batch(texts: list[str]) -> list[str]:
        """Decodes all texts in one batch. Prompts are right-padded with mask tokens like the dInfer benchmark, and the
        shared canvas is sized for the longest text (2.5x its source tokens)."""
        ids = [prompt_ids(t) for t in texts]
        longest_src = max(len(tokenizer(t)["input_ids"]) for t in texts)
        width = max(len(i) for i in ids)
        gen = next((g for g in GEN_BUCKETS if g >= 2.5 * longest_src), GEN_BUCKETS[-1])
        if not QWEN:
            gen = max(BLOCK, ((width + gen) // 32) * 32 - width)  # same 32-token total-length bucketing as the benchmark
        batch = torch.full((len(ids), width), MASK_ID, dtype=torch.long, device=device)
        for j, row in enumerate(ids):
            batch[j, : len(row)] = torch.tensor(row, device=device)
        out = dllm.generate(batch, gen_length=gen, block_length=BLOCK)
        return [tokenizer.decode(out[j, len(row):], skip_special_tokens=True).strip() for j, row in enumerate(ids)]

    def translate_stream(texts: list[str]):
        """Splits any step over --max_src_tokens English tokens into balanced pieces at paragraph/sentence boundaries,
        groups all pieces by the canvas bucket they need, and decodes each group in parallel as one batch (smallest
        canvas first). Yields (index, translation) once every piece of a step is done."""
        count = lambda t: len(tokenizer(t)["input_ids"])
        split = [split_text(t, count, args.max_src_tokens) for t in texts]
        pieces = [(i, k, p) for i, (ps, _) in enumerate(split) for k, p in enumerate(ps)]
        need = [next((g for g in GEN_BUCKETS if g >= 2.5 * count(p)), GEN_BUCKETS[-1]) for _, _, p in pieces]
        done: dict[int, dict[int, str]] = {}
        for bucket in sorted(set(need)):
            idxs = [j for j in range(len(pieces)) if need[j] == bucket]
            size = max(1, min(MAX_BATCH, TOKEN_BUDGET // bucket))  # bound batch * canvas so long canvases fit in memory
            for start in range(0, len(idxs), size):
                group = idxs[start : start + size]
                for j, hi in zip(group, translate_batch([pieces[j][2] for j in group])):
                    i, k, _ = pieces[j]
                    done.setdefault(i, {})[k] = hi
                    if len(done[i]) == len(split[i][0]):
                        yield i, join_pieces([done[i][k] for k in range(len(done[i]))], split[i][1])

    print("[dinfer_server] warming up", flush=True)
    for g in GEN_BUCKETS[:4]:
        for n in (1, 8):
            translate_batch(["Let me think about this. " * (g // 10)] * n)
    print(f"[dinfer_server] ready on :{args.port}", flush=True)

    lock = threading.Lock()  # one decode at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": args.model}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            if self.path == "/translate_stream":
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()  # HTTP/1.0: body is delimited by connection close
                t0 = time.time()
                with lock:
                    for i, hi in translate_stream(texts):
                        self.wfile.write((json.dumps({"idx": i, "hi": hi, "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n").encode())
                        self.wfile.flush()
                return
            if self.path != "/translate":
                return self._send(404, {})
            texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
            t0 = time.time()
            with lock:
                out = [hi for _, hi in sorted(translate_stream(texts))]
            self._send(200, {"translations": out, "seconds": round(time.time() - t0, 2)})

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()

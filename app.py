"""End-to-end EN -> HI reasoning-translation demo.

Picks a random document from reasoning_hi_train.jsonl (already chunked into
{"en", "hi"} steps by data_gen/download_datasets.py --dataset reasoning_hi),
and translates its steps through one of two backends, selectable in the
sidebar:

- **LLaDA-MoE + dInfer**: calls the standalone dinfer_server.py process
  (FusedMoE kernels + dual KV-caching + threshold decoding, threshold=0.95,
  cont_weight=0.3 - the best config found in the checkpoint-2000 quality
  sweep) over HTTP. No local GPU/model dependency for this path.
- **Qwen3-BD3LM (dllm)**: loads the proven, working `dllm`/BD3LMSampler
  pipeline (same one used for validation/validate_reasoning_hi.py, which
  scored sim mean=0.880 on the 100k held-out set) in-process on its own GPU.
  This is the standard pipeline, not the still-buggy dInfer integration
  from dinfer_integration/ (that one truncates mid-sentence - see
  bd3lm_iteration.py's docstring) - use this backend, not that one, until
  that bug is fixed.

Usage:
    dinfer_server.py must already be running for the LLaDA-MoE backend
    (make dinfer-server-up). Then:
        streamlit run app.py
"""

import json
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # for the in-process Qwen3-BD3LM backend

import requests
import streamlit as st

import dllm
from translate import (
    BD3LMSamplerConfig,
    ScriptArguments,
    estimate_max_new_tokens,
    load_pipeline,
    translate_batch,
)

DATASET_PATH = "reasoning_hi_train.jsonl"

# --- LLaDA-MoE + dInfer backend (HTTP) ---
SERVER_URL = os.environ.get("DINFER_SERVER_URL", "http://localhost:8092")
DINFER_BLOCK_LENGTH = 64
DEFAULT_REPETITION_PENALTY_DINFER = 1.3
REQUEST_TIMEOUT_S = 300

# --- Qwen3-BD3LM (dllm) backend (in-process) ---
BD3LM_MODEL_PATH = ".models/qwen3-a2d-bd3lm-reasoning-hi-100k-waitfix/checkpoint-2975"
BD3LM_BLOCK_SIZE = 32

MAX_NEW_TOKENS_CAP = 1024
TOKEN_BUDGET = 4096

BACKENDS = ["LLaDA-MoE + dInfer", "Qwen3-BD3LM (dllm)"]


@st.cache_resource(show_spinner=f"Loading {BD3LM_MODEL_PATH} onto GPU {os.environ['CUDA_VISIBLE_DEVICES']}…")
def _load_bd3lm():
    args = ScriptArguments(model_name_or_path=BD3LM_MODEL_PATH)
    _, tokenizer, sampler = load_pipeline(args, sampler_type="bd3lm")
    return tokenizer, sampler


def _estimate_max_new_tokens_dinfer(texts: list[str], factor: float = 2.5, min_tokens: int = 16, max_tokens: int = MAX_NEW_TOKENS_CAP) -> int:
    """Whitespace-token proxy for source length (LLaDA-MoE backend has no
    local tokenizer - the server owns the real one; this is just for
    client-side batch sizing)."""
    longest_src = max(len(t.split()) for t in texts)
    raw = max(min_tokens, min(max_tokens, round(longest_src * factor)))
    return -(-raw // DINFER_BLOCK_LENGTH) * DINFER_BLOCK_LENGTH


def _chunk_steps(texts: list[str], backend: str, tokenizer=None) -> list[tuple[list[str], int]]:
    """Greedily groups step texts into sub-batches, each paired with the
    shared max_new_tokens its steps need, such that
    len(sub_batch) * max_new_tokens stays under TOKEN_BUDGET."""
    chunks = []
    current: list[str] = []
    current_max = 0
    for text in texts:
        if backend == "Qwen3-BD3LM (dllm)":
            step_max = estimate_max_new_tokens([text], tokenizer, max_tokens=MAX_NEW_TOKENS_CAP)
        else:
            step_max = _estimate_max_new_tokens_dinfer([text])
        candidate_max = max(current_max, step_max)
        if current and (len(current) + 1) * candidate_max > TOKEN_BUDGET:
            chunks.append((current, current_max))
            current, candidate_max = [], step_max
        current.append(text)
        current_max = candidate_max
    if current:
        chunks.append((current, current_max))
    return chunks


def _translate_chunk_dinfer(chunk_texts: list[str], gen_length: int, repetition_penalty: float) -> list[str]:
    resp = requests.post(
        f"{SERVER_URL}/translate",
        json={"texts": chunk_texts, "gen_length": gen_length, "repetition_penalty": repetition_penalty},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["translations"]


def _translate_chunk_bd3lm(chunk_texts: list[str], max_new_tokens: int, tokenizer, sampler) -> list[str]:
    config = BD3LMSamplerConfig(max_new_tokens=max_new_tokens, block_size=BD3LM_BLOCK_SIZE, temperature=0.3)
    return translate_batch(chunk_texts, tokenizer, sampler, config)


def _random_doc(path: str) -> dict:
    """Reads one random line from a JSONL file without loading it all into
    memory: seeks to a random byte offset, discards the (likely partial)
    line it lands in, and reads the next full line."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(random.randint(0, size - 1))
        f.readline()
        line = f.readline()
        if not line:
            f.seek(0)
            line = f.readline()
    return json.loads(line)


st.set_page_config(page_title="CoT Translator", layout="centered")
st.title("Chain-of-Thought Translator (EN → HI)")

backend = st.sidebar.radio("Backend", BACKENDS)

if backend == "LLaDA-MoE + dInfer":
    st.caption(f"Served by dinfer_server.py at `{SERVER_URL}` · LLaDA-MoE + dInfer")
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5).json()
        st.caption(f"Server up: model=`{health['model']}` on GPU {health['gpu']}")
    except requests.exceptions.RequestException:
        st.error(
            f"Can't reach the dInfer server at {SERVER_URL}. Start it first:\n\n"
            "`make dinfer-server-up`"
        )
        st.stop()
    repetition_penalty = st.sidebar.slider(
        "Repetition penalty",
        min_value=1.0,
        max_value=2.0,
        value=DEFAULT_REPETITION_PENALTY_DINFER,
        step=0.05,
        help="Discourages the model from repeating tokens it already committed earlier in the sequence "
        "(the 'मॉड्यूलर मॉड्यूलर मॉड्यूलर...' failure mode). 1.0 disables it.",
    )
else:
    st.caption(f"Model: `{BD3LM_MODEL_PATH}` · dllm BD3LMSampler · GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

if st.button("Pick random example", type="primary"):
    st.session_state["doc"] = _random_doc(DATASET_PATH)

doc = st.session_state.get("doc")

if doc:
    steps = doc["steps"]
    st.caption(f"Document `{doc['id'][:12]}` — {len(steps)} step(s)")

    tokenizer = None
    sampler = None
    if backend == "Qwen3-BD3LM (dllm)":
        tokenizer, sampler = _load_bd3lm()

    chunks = _chunk_steps([s["en"] for s in steps], backend, tokenizer)

    placeholders = []
    for i, step in enumerate(steps, 1):
        col1, col2 = st.columns(2)
        col1.caption(f"Step {i} — EN")
        col1.text(step["en"])
        col2.caption(f"Step {i} — HI (model)")
        ph = col2.empty()
        ph.text("<pending>")
        placeholders.append(ph)
        with st.expander("Reference HI (dataset)"):
            st.text(step["hi"])

    with st.spinner(f"Decoding {len(steps)} step(s) in {len(chunks)} batch(es)…"):
        idx = 0
        for chunk_texts, chunk_max_new_tokens in chunks:
            chunk_placeholders = placeholders[idx : idx + len(chunk_texts)]
            try:
                if backend == "LLaDA-MoE + dInfer":
                    translations = _translate_chunk_dinfer(chunk_texts, chunk_max_new_tokens, repetition_penalty)
                else:
                    translations = _translate_chunk_bd3lm(chunk_texts, chunk_max_new_tokens, tokenizer, sampler)
            except requests.exceptions.RequestException as e:
                translations = [f"<server error: {e}>"] * len(chunk_texts)
            for ph, translation in zip(chunk_placeholders, translations):
                ph.text(translation or "<empty>")
            idx += len(chunk_texts)

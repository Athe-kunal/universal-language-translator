"""End-to-end EN -> HI reasoning-translation demo.

Picks a random document from reasoning_hi_train.jsonl (already chunked into
{"en", "hi"} steps by data_gen/download_reasoning_hi.py), and
diffusion-decodes its steps through the BD3LM sampler (each step starts
fully masked), batching as many steps together per call as fit under
TOKEN_BUDGET (some docs have 100+ steps, so one batch per doc can OOM
regardless of per-step canvas size - see _chunk_steps). Runs on GPU 0.

Usage:
    streamlit run app.py
"""

import json
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import streamlit as st

from translate import (
    BD3LMSamplerConfig,
    ScriptArguments,
    estimate_max_new_tokens,
    load_pipeline,
    translate_batch,
)

MODEL_PATH = ".models/qwen3-a2d-bd3lm-reasoning-hi/checkpoint-2000"
DATASET_PATH = "reasoning_hi_train.jsonl"
MAX_NEW_TOKENS_CAP = 512
# Bounds batch_size * max_new_tokens per sampler.sample() call. Docs in this
# dataset range from a handful of steps to 400+; batching all of a doc's
# steps in one call scales with both, so this caps the per-call diffusion
# canvas (activation memory) rather than the step count.
TOKEN_BUDGET = 4096


@st.cache_resource(show_spinner=f"Loading {MODEL_PATH} onto GPU 0…")
def _load():
    args = ScriptArguments(model_name_or_path=MODEL_PATH)
    _, tokenizer, sampler = load_pipeline(args, sampler_type="bd3lm")
    return tokenizer, sampler


def _chunk_steps(texts: list[str], tokenizer) -> list[tuple[list[str], int]]:
    """Greedily groups step texts into sub-batches, each paired with the
    shared max_new_tokens its steps need, such that
    len(sub_batch) * max_new_tokens stays under TOKEN_BUDGET."""
    chunks = []
    current: list[str] = []
    current_max = 0
    for text in texts:
        step_max = estimate_max_new_tokens([text], tokenizer, max_tokens=MAX_NEW_TOKENS_CAP)
        candidate_max = max(current_max, step_max)
        if current and (len(current) + 1) * candidate_max > TOKEN_BUDGET:
            chunks.append((current, current_max))
            current, candidate_max = [], step_max
        current.append(text)
        current_max = candidate_max
    if current:
        chunks.append((current, current_max))
    return chunks


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
st.caption(f"Model: `{MODEL_PATH}` · BD3LM block diffusion · GPU 0")

if st.button("Pick random example", type="primary"):
    st.session_state["doc"] = _random_doc(DATASET_PATH)

doc = st.session_state.get("doc")

if doc:
    steps = doc["steps"]
    st.caption(f"Document `{doc['id'][:12]}` — {len(steps)} step(s)")

    tokenizer, sampler = _load()
    chunks = _chunk_steps([s["en"] for s in steps], tokenizer)

    translations = []
    with st.spinner(
        f"Decoding {len(steps)} step(s) from masks in {len(chunks)} batch(es)…"
    ):
        for chunk_texts, chunk_max_new_tokens in chunks:
            config = BD3LMSamplerConfig(max_new_tokens=chunk_max_new_tokens)
            translations.extend(
                translate_batch(chunk_texts, tokenizer, sampler, config)
            )

    for i, (step, hi_model) in enumerate(zip(steps, translations), 1):
        col1, col2 = st.columns(2)
        col1.caption(f"Step {i} — EN")
        col1.write(step["en"])
        col2.caption(f"Step {i} — HI (model)")
        col2.write(hi_model or "_<empty>_")
        with st.expander("Reference HI (dataset)"):
            st.write(step["hi"])

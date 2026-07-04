"""
Minimal Streamlit app for English → Hindi translation.

Usage:
    streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import torch
import streamlit as st

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from data_gen.translate_ds import split_paragraphs as split_en

from translate import ScriptArguments, SamplerConfig, load_pipeline, translate_one, translate_batch


@st.cache_resource(show_spinner="Loading model…")
def _load(model_path: str):
    args = ScriptArguments(model_name_or_path=model_path)
    _, tokenizer, sampler = load_pipeline(args)
    return tokenizer, sampler


@st.cache_data(show_spinner="Fetching example…")
def _fetch_example():
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    ex = next(iter(ds))
    return ex["problem"], ex["solution"]


@st.cache_data(show_spinner="Loading translation cache…")
def _load_cache(path: str, mtime: float):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EN → HI Translator", layout="centered")
st.title("English → Hindi Translator")

tab_translate, tab_cache = st.tabs(["Translate", "Translation Cache"])

with tab_cache:
    cache_path = st.text_input("Cache file path", value="translation_cache.jsonl")
    p = Path(cache_path)
    if not p.exists():
        st.warning(f"`{cache_path}` not found.")
    else:
        records = _load_cache(str(p), p.stat().st_mtime)
        st.caption(f"{len(records)} example(s) in cache")

        query = st.text_input("Search (matches problem/solution text or id)", value="")
        if query:
            q = query.lower()
            records = [
                r for r in records
                if q in r.get("id", "").lower()
                or q in r.get("problem", "").lower()
                or q in r.get("solution", "").lower()
            ]
            st.caption(f"{len(records)} match(es)")

        n = st.number_input("Max rows to show", min_value=1, max_value=len(records) or 1,
                             value=min(20, len(records) or 1))

        for r in records[: int(n)]:
            with st.expander(f"{r.get('id', '')[:12]} — {r.get('problem', '')[:80]}"):
                col1, col2 = st.columns(2)
                col1.subheader("Problem — EN")
                col1.write(r.get("problem", ""))
                col2.subheader("Problem — HI")
                col2.write(r.get("problem_hi", ""))

                col1, col2 = st.columns(2)
                col1.subheader("Solution — EN")
                col1.write(r.get("solution", ""))
                col2.subheader("Solution — HI")
                col2.write(r.get("solution_hi", ""))

with tab_translate:
    model_path = st.sidebar.text_input(
        "Checkpoint path",
        value=".models/modernbert-translation/checkpoint-final",
    )
    steps = st.sidebar.slider("Diffusion steps", 16, 256, 128, step=16)
    max_new_tokens = st.sidebar.slider("Max new tokens", 32, 512, 128, step=32)
    if torch.cuda.is_available():
        st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("No GPU — running on CPU")

    # Example loader
    if st.button("Load example from OpenR1-Math-220k"):
        prob, sol = _fetch_example()
        st.session_state["question_en"] = prob
        st.session_state["solution_en"] = sol

    question_en = st.text_area(
        "Question (English)", height=120,
        placeholder="Enter the English question…",
        value=st.session_state.get("question_en", ""),
    )
    solution_en = st.text_area(
        "Solution (English)", height=220,
        placeholder="Enter the English solution…",
        value=st.session_state.get("solution_en", ""),
    )

    if st.button("Translate", type="primary", disabled=not (question_en or solution_en)):
        try:
            tokenizer, sampler = _load(model_path)
        except Exception as e:
            st.error(f"Failed to load model from `{model_path}`:\n\n{e}")
            st.stop()

        config = SamplerConfig(steps=steps, max_new_tokens=max_new_tokens)

        q = question_en.strip()
        chunks = split_en(solution_en.strip()) if solution_en.strip() else []
        texts = ([q] if q else []) + chunks

        with st.spinner(f"Translating {len(texts)} text(s) in one batch…"):
            results = translate_batch(texts, tokenizer, sampler, config)

        idx = 0
        if q:
            col1, col2 = st.columns(2)
            col1.subheader("Question — EN")
            col1.write(q)
            col2.subheader("Question — HI")
            col2.write(results[idx] or "_<empty>_")
            idx += 1

        if chunks:
            st.subheader(f"Solution — {len(chunks)} chunk(s)")
            for i, (en, hi) in enumerate(zip(chunks, results[idx:]), 1):
                col1, col2 = st.columns(2)
                col1.caption(f"Chunk {i} — EN")
                col1.write(en)
                col2.caption(f"Chunk {i} — HI")
                col2.write(hi)

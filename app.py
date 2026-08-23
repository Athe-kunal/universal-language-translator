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

from translate import (
    ScriptArguments,
    SamplerConfig,
    estimate_max_new_tokens,
    load_pipeline,
    translate_one,
    translate_batch,
)


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

tab_free, tab_translate, tab_cache, tab_reasoning = st.tabs(
    ["Translate (free text)", "Translate (Q&A)", "Translation Cache", "Reasoning Translations"]
)

with tab_free:
    model_path_free = st.sidebar.text_input(
        "Checkpoint path (free text)",
        value=".models/modernbert-chat-bpcc-translation/checkpoint-3609",
        key="model_path_free",
    )
    st.caption(f"Model: `{model_path_free}`")

    english_text = st.text_area(
        "English",
        height=150,
        placeholder="Type or paste English text to translate…",
        key="free_text_input",
    )

    if st.button("Translate to Hindi", type="primary", disabled=not english_text.strip()):
        try:
            tokenizer, sampler = _load(model_path_free)
        except Exception as e:
            st.error(f"Failed to load model from `{model_path_free}`:\n\n{e}")
            st.stop()

        max_new_tokens = estimate_max_new_tokens([english_text], tokenizer)
        config = SamplerConfig(max_new_tokens=max_new_tokens, steps=max_new_tokens)

        with st.spinner("Translating…"):
            hindi_text = translate_one(english_text.strip(), tokenizer, sampler, config)

        st.text_area("Hindi", value=hindi_text or "<empty>", height=150, disabled=True)

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

with tab_reasoning:
    st.caption(
        "Output of `data_gen/translate_reasoning.py` — chunked OpenThoughts3 / "
        "natural_reasoning documents translated via `data_gen/chunking.py` + a "
        "vllm-served model, with n-gram-repetition and placeholder-conservation "
        "retry built in."
    )
    with st.expander("Live progress (per-chunk, updates while a run is in progress)", expanded=False):
        units_path = st.text_input(
            "Per-unit JSONL path", value="translated_reasoning_units.jsonl", key="units_path"
        )
        if st.button("Refresh", key="units_refresh"):
            st.rerun()
        up = Path(units_path)
        if not up.exists():
            st.info(f"`{units_path}` not found yet.")
        else:
            unit_records = _load_cache(str(up), up.stat().st_mtime)
            st.caption(f"{len(unit_records)} chunk(s) translated so far")
            retried_live = sum(1 for u in unit_records if u.get("retries", 0) > 0)
            exhausted_live = sum(1 for u in unit_records if u.get("exhausted"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks so far", len(unit_records))
            c2.metric("Retried", retried_live)
            c3.metric("Exhausted", exhausted_live)
            st.caption("Most recent 20 chunks:")
            for u in unit_records[-20:][::-1]:
                flag = " 🔴 exhausted" if u.get("exhausted") else (" 🟡 retried" if u.get("retries", 0) > 0 else "")
                st.markdown(f"**[{u.get('doc_id', '')[:10]} / {u.get('unit_id', '')}]**{flag}")
                ucol1, ucol2 = st.columns(2)
                ucol1.caption("EN")
                ucol1.write(u.get("en", "")[:300])
                ucol2.caption("HI")
                ucol2.write((u.get("hi") or "_<fell back to EN>_")[:300])

    reasoning_path = st.text_input(
        "Translated-reasoning JSONL path", value="translated_reasoning.jsonl", key="reasoning_path"
    )
    rp = Path(reasoning_path)
    if not rp.exists():
        st.warning(f"`{reasoning_path}` not found.")
    else:
        docs = _load_cache(str(rp), rp.stat().st_mtime)
        st.caption(f"{len(docs)} document(s) loaded")

        source_filter = st.selectbox(
            "Source", ["all", "openthoughts", "naturalreasoning"], key="reasoning_source_filter"
        )
        docs_view = docs if source_filter == "all" else [d for d in docs if d.get("source") == source_filter]

        all_units = [u for d in docs_view for u in d.get("units", [])]
        total_units = len(all_units)
        retried_units = sum(1 for u in all_units if u.get("retries", 0) > 0)
        exhausted_units = sum(1 for u in all_units if u.get("exhausted"))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", len(docs_view))
        col2.metric("Units", total_units)
        col3.metric("Retried", f"{retried_units} ({100 * retried_units / total_units:.1f}%)" if total_units else "0")
        col4.metric(
            "Exhausted (fell back to EN)",
            f"{exhausted_units} ({100 * exhausted_units / total_units:.1f}%)" if total_units else "0",
        )

        query = st.text_input("Search (matches id/question/en_cot_answer)", value="", key="reasoning_query")
        if query:
            q = query.lower()
            docs_view = [
                d
                for d in docs_view
                if q in d.get("id", "").lower()
                or q in d.get("question", "").lower()
                or q in d.get("en_cot_answer", "").lower()
            ]
            st.caption(f"{len(docs_view)} match(es)")

        only_with_issues = st.checkbox("Only show documents with retried/exhausted units", value=False)
        if only_with_issues:
            docs_view = [d for d in docs_view if any(u.get("retries", 0) > 0 for u in d.get("units", []))]

        n = st.number_input(
            "Max documents to show",
            min_value=1,
            max_value=len(docs_view) or 1,
            value=min(20, len(docs_view) or 1),
            key="reasoning_max_rows",
        )

        for d in docs_view[: int(n)]:
            n_retried = sum(1 for u in d.get("units", []) if u.get("retries", 0) > 0)
            n_exhausted = sum(1 for u in d.get("units", []) if u.get("exhausted"))
            badge = f" ⚠️ retried={n_retried} exhausted={n_exhausted}" if n_retried else ""
            with st.expander(f"[{d.get('source', '')}] {d.get('id', '')[:12]} — {d.get('question', '')[:80]}{badge}"):
                col1, col2 = st.columns(2)
                col1.subheader("Full document — EN")
                col1.write(d.get("en_cot_answer", ""))
                col2.subheader("Full document — HI (reconstructed)")
                col2.write(d.get("hi_cot_answer", ""))

                st.subheader(f"Units ({len(d.get('units', []))})")
                for u in d.get("units", []):
                    flag = ""
                    if u.get("exhausted"):
                        flag = " 🔴 exhausted — fell back to EN"
                    elif u.get("retries", 0) > 0:
                        flag = f" 🟡 retried {u['retries']}x"
                    st.markdown(f"**[{u.get('unit_id', '')}]** kind=`{u.get('kind')}`{flag}")
                    ucol1, ucol2 = st.columns(2)
                    ucol1.caption("EN")
                    ucol1.write(u.get("en", ""))
                    ucol2.caption("HI")
                    ucol2.write(u.get("hi") if u.get("hi") is not None else "_<fell back to EN — see erroneous attempt below>_")
                    if u.get("erroneous_translation"):
                        st.caption("Erroneous attempt shown to the model on retry:")
                        st.text(u["erroneous_translation"])
                    st.divider()

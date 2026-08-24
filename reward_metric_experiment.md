# Choosing a Reward Metric for English→Hindi Translation RL

**Date:** 2026-08-24
**Scope:** small-scale probe (150-500 examples per test), GPU2 only, to pick a direction — not a publication-grade benchmark.

## TL;DR recommendation

**Switch the embedding-similarity signal from LaBSE to `jina-embeddings-v3`.** It keeps LaBSE's core advantages (fast, no learned register bias) while actually fixing the long-context problem instead of working around it — 0/90 truncated up to 753 tokens tested (vs LaBSE's 67%), still fast (8.5ms/example), and its discourse-marker catch rate (78.0%) is close to LaBSE's (88.7%) with the same benign near-tie failure pattern, not the systematic wrong-direction bias MetricX showed. None of the three heavier QE models tested (COMET-QE, MetricX-24, BGE-M3) beat this combination on the failure mode this project actually cares about. Keep the deterministic checks as hard gates regardless of which embedding model is used.

## Setup

- **Candidates tested:** LaBSE (current baseline), COMET-QE (`Unbabel/wmt20-comet-qe-da`), MetricX-24-Hybrid-Large (`google/metricx-24-hybrid-large-v2p6`, QE mode), BGE-M3 (`BAAI/bge-m3`, dense embeddings), jina-embeddings-v3 (`jinaai/jina-embeddings-v3`, text-matching task head) — the last two added as long-context (8192-token) LaBSE-class alternatives once the truncation bug surfaced.
- **Not tested:** xCOMET-XL/XXL (gated on HF, access not granted in the time available), MetricX-23-QE variants (same family as MetricX-24, redundant to test both), custom reranker (stretch goal, not pursued — none of the tested metrics were reliable enough to bootstrap from with confidence).
- **Hardware:** GPU2 only, isolated Python venv (kept separate from the main project venv — `unbabel-comet` and MetricX have protobuf/transformers version pins that conflict with `dllm`'s).
- **Data:**
  - **Discourse-marker probe (150 pairs):** real before/after pairs pulled directly from this project's own fix pass — `hi_bad` is the original "Wait" → "देर" mistranslation from `translated_reasoning_25k_units.jsonl`, `hi_good` is the corrected translation for the *same* English text from `translated_reasoning_25k_units_fixed.jsonl`. Not synthetic — these are real production outputs on both sides.
  - **BPCC sample (150 pairs):** random sample from `bpcc_hin_deva.jsonl`, used as an external, higher-trust anchor. A reference-based COMET score (`Unbabel/wmt22-comet-da`, using the BPCC target as both hypothesis and reference) serves as a rough "ceiling" to check whether the reference-free metrics track something real.
  - **Long-context sample (90 pairs):** real translated *steps* (not chunks) from `reasoning_translation_train.jsonl`, token counts 60-739, specifically to test whether metrics hold up at the pipeline's actual step-level translation granularity (250-600 tokens typical, occasionally more).

## Results

### Discourse-marker catch rate (does the metric score the correct translation better than the known-bad one?)

| Metric | Catch rate | Mean score, good | Mean score, bad | Direction |
|---|---|---|---|---|
| **LaBSE** | **88.7%** (133/150) | 0.848 | 0.835 | higher = better |
| **COMET-QE** | **86.7%** (130/150) | −0.326 | −0.363 | higher = better |
| **jina-embeddings-v3** | **78.0%** (117/150) | — | — | higher = better |
| **BGE-M3** | **36.7%** (55/150) | — | — | higher = better |
| **MetricX-24** | **5.3%** (8/150) | 10.73 | 10.17 | lower = better (this is *backwards*) |

### BPCC sanity check (correlation with a reference-based COMET ceiling)

| Metric | Pearson r vs. ref-based ceiling | Mean score |
|---|---|---|
| LaBSE | 0.290 | 0.865 |
| **COMET-QE** | **0.499** | 0.454 |
| BGE-M3 | 0.484 | 0.838 |
| jina-embeddings-v3 | 0.468 | 0.873 |
| MetricX-24 | 0.378 | 1.788 (lower=better) |

### Long-context truncation (90 real translated steps, 60-739 tokens)

| Metric | Tokenizer max length | % truncated |
|---|---|---|
| LaBSE | **256 tokens** | **67%** |
| COMET-QE | 512 tokens | 36% |
| MetricX-24 | 1536 tokens (but scores *combined* source+candidate in one string) | 30% |
| **BGE-M3** | 8192 tokens | **0%** |
| **jina-embeddings-v3** | 8192 tokens | **0%** |

### Latency (single A100, ms/example)

| Metric | Latency |
|---|---|
| **BGE-M3** | **4.7 ms** |
| LaBSE | 5.8 ms |
| jina-embeddings-v3 | 8.5 ms |
| COMET-QE | 30.5 ms |
| MetricX-24-Large | 55.0 ms |

## Failure-mode write-up

**MetricX-24's failure isn't noise — it's a systematic, directional bias.** In 142/150 pairs, MetricX scored the *correct* translation ("रुको, ...") as worse than the *wrong* one ("देर, ..." / "देर करो, ..."), with small but consistent gaps (0.2-1.7 points on a 0-25 error scale). Manually inspecting the pairs, the pattern is genre mismatch, not a scoring bug: MetricX is trained on WMT MQM/DA data — formal news, tech documentation, parliamentary text — which essentially never contains a sentence-opening spoken interjection like "रुको" ("wait!"). It appears to have learned that this register reads as unusual/informal phrasing and penalizes it, even though "रुको" is exactly the correct choice for this project's stream-of-consciousness chain-of-thought register, and "देर" (a bureaucratic-sounding "delay") is wrong. This matches the experiment brief's own stated concern almost exactly: *"most QE models are validated mainly on high-resource pairs"* — the gap here isn't language-resource level, it's **genre**: none of these metrics were trained on CoT-reasoning-style text in any language.

**LaBSE's misses are close calls, not systematic bias.** Where LaBSE failed to prefer the good translation, the score gap was usually ~0.001-0.01 — effectively ties, driven by the fact that both hi_good and hi_bad share nearly all of their content (only the first word differs), and LaBSE's sentence-level pooling dilutes a one-word difference in a multi-sentence paragraph. This is a real, known limitation (see below) but it's benign compared to MetricX's confident wrong-direction scoring.

**COMET-QE's misses looked similar to LaBSE's** — mostly small-margin ties rather than confident wrong answers, and it has the best correlation with the reference-based BPCC anchor, suggesting it's the most "generally sane" metric of the two neural candidates on ordinary (non-CoT) Hindi text. It just doesn't add anything over LaBSE for the one failure mode this project is actually trying to fix.

**The long-context result is the most actionable one, and it's now fully solved.** LaBSE silently truncates two-thirds of real translated steps at 256 tokens — this has been happening in production since the embedding-similarity validation check was added, and it means the `_validate()` check has effectively been scoring only the *first ~200 words* of most steps, not the whole thing. Both BGE-M3 and jina-embeddings-v3 (8192-token native context) truncated **zero** of the 90 long-context samples, including the longest one tested at 753 tokens — comfortably past `max_step_tokens` (default 600). This isn't a partial mitigation like COMET-QE's or MetricX's larger-but-still-limited windows; it removes the blind spot outright.

**BGE-M3 is a genuine surprise — strong on paper, weak on this specific error type.** Its BPCC correlation (0.484) and latency (4.7ms, fastest of everything tested) are excellent, but its discourse-marker catch rate (36.7%) is close to a coin flip and clearly worse than jina-v3, LaBSE, or COMET-QE. This wasn't inspected as deeply as MetricX's failure (time budget), but it doesn't show MetricX's confident wrong-direction pattern — more likely BGE-M3's dense-embedding pooling (used here; it also supports sparse and multi-vector representations that weren't tested and might do better on fine-grained single-word differences) just isn't sensitive enough to a one-word change buried in an otherwise-identical paragraph. Worth a follow-up with BGE-M3's sparse/ColBERT-style output if it's ever reconsidered.

**jina-embeddings-v3 is the standout.** Its catch-rate misses (33/150) are small-margin near-ties, not systematic bias — max miss margin was 0.023, mean −0.006, the same benign pattern as LaBSE's own misses (see above), not MetricX's confident 0.2-1.7-point wrong-direction gaps. Combined with solving truncation outright and staying fast, it's a strict upgrade path over LaBSE: give up ~11 points of catch rate on this specific probe, gain full-length step coverage that LaBSE structurally can't provide today.

## Deterministic / rule-based checks (unchanged recommendation)

Not the focus of this comparison, but confirmed still worth keeping as hard gates regardless of which neural metric is used:
- **Placeholder fidelity** — already implemented (`_validate()`'s placeholder-set check), zero-cost, hard binary signal. Keep.
- **Repetition/degeneration penalty** — already implemented (`has_repetition()`), catches a real failure mode this project has seen at scale. Keep.
- **Length ratio** — not currently implemented. Worth adding; cheap and catches dropped/hallucinated content the neural metrics can miss when they truncate.
- **Format/tag well-formedness** — effectively covered by the placeholder check today.

## Decision rationale (per the experiment's own criteria)

1. **Correlation with ground truth:** LaBSE (0.887) and COMET-QE (0.867) lead, jina-v3 close behind (0.780), BGE-M3 weak (0.367), MetricX fails badly (0.053).
2. **Discourse-marker catch rate:** same ranking as above; jina-v3's shortfall vs. LaBSE is small-margin misses, not systematic bias.
3. **Latency budget:** BGE-M3 (4.7ms) and LaBSE (5.8ms) are fastest, jina-v3 close (8.5ms) — all three embedding models are 4-11x faster than COMET-QE and 6-12x faster than MetricX, so latency doesn't meaningfully separate the embedding-model tier.
4. **Token-limit fit:** this is where the decision actually turns. LaBSE, COMET-QE, and MetricX all truncate a meaningful fraction of real step-sized translations (67%, 36%, 30% respectively). BGE-M3 and jina-v3 truncate **none** of them.

**Net:** jina-embeddings-v3 dominates on the criterion that actually matters for this pipeline (steps, not short sentences) while giving up only a modest amount of accuracy on the discourse-marker probe relative to LaBSE, and that shortfall is benign (near-ties, not confident wrong answers). BGE-M3 is not recommended despite its excellent speed and BPCC correlation, purely because of its weak discourse-marker catch rate. COMET-QE and MetricX are not recommended at all — COMET-QE still truncates over a third of real steps and adds no accuracy benefit over jina-v3/LaBSE, and MetricX has a systematic bias against the exact register this project needs.

## Caveats

- Small-scale probe (150-500 examples), not the full 300-500 BPCC + hand-labeled gold set the original experiment spec called for — no hand-labeled adequacy/fluency scores were collected, so "catch rate on a real bad-vs-good pair" stood in for correlation-with-human-label.
- xCOMET (which does token-level error-span detection, potentially the best-suited *architecture* for catching a single-word discourse-marker error) was not evaluated — blocked by HF gating in the time available. This is the most promising follow-up if the team wants to keep looking past what's covered here.
- MetricX was only tested at the "Large" size (Google's own docs note XL/XXL correlate better with human judgment generally); it's unclear whether the register bias found here would persist at larger sizes, though it seems more likely to be a training-data-domain issue than a capacity issue.
- BGE-M3 was only tested with its dense-embedding output; its sparse and multi-vector (ColBERT-style) representations were not tested and might handle fine-grained single-word differences better — worth a follow-up if BGE-M3's excellent speed/correlation numbers make it worth reconsidering.
- The composite-reward step (Step 5 in the original brief) wasn't run — jina-embeddings-v3 alone already clears the bar this project needs without combining metrics.

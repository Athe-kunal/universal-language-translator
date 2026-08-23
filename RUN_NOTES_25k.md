# 25K + 25K translation run — decisions & assumptions

Written while you were away, covering the run launched at 2026-08-22 21:00
(PID 2772762, background). Read this before trusting the output.

## UPDATE (major bottleneck found + fixed for future runs, current run NOT restarted)

The run got stuck at 0 translated documents for a very long time. Root cause,
confirmed by direct timing:

1. `chunk_document()` runs synchronously inside each of the 50,000 tasks,
   blocking asyncio's single-threaded event loop — with 50K tasks queued,
   this serializes into a long backlog before translation HTTP traffic can
   flow at all. **Fixed**: `chunk_document()` now runs via `asyncio.to_thread`
   in `process_document()` (`data_gen/translate_reasoning.py`).
2. The much bigger cost: `wtpsplit`'s SaT model (used by `chunking.py`'s
   oversized-unit split fallback) defaults to **CPU**, and CPU inference for
   it is roughly **20-40x slower than GPU** — confirmed directly: a batch of
   5 split-heavy documents didn't finish in 3+ minutes on CPU; the same batch
   took well under 1s/doc once the model was moved to GPU. **Fixed**: added
   `chunking.set_sat_device()` + a `--sat_device` CLI flag to
   `translate_reasoning.py`, wired to call it at startup.

**Both fixes are committed and will apply to any future/new run** (pass
`--sat_device cuda:2` next time, sharing GPU2 with the embedding model — both
are small models with plenty of headroom there).

**This specific already-running job was NOT restarted with the fixes.**
Reasoning: `chunk_document()` is called exactly once per document, before
translation starts — it's a one-time front-loaded cost, not something that
recurs during the actual translation phase. By the time this was diagnosed,
the run was ~89% through chunking (44,451/50,000). Restarting would have cost
~78 minutes to re-run the network-bound OT3 sampling alone (no caching exists
for the raw sample, only for final translated output) — likely slower overall
than just letting the last ~11% of chunking finish on CPU. So: this run
finishes chunking the slow way, then translation should proceed normally
(GPU3-bound, not CPU-bound) with no further wtpsplit involvement.

If you want the GPU-accelerated chunking benefit for a *future* run (e.g. a
much larger release-scale run later), use `--sat_device cuda:2`.

## What's running

```
uv run python data_gen/translate_reasoning.py \
  --num_openthoughts3 25000 --num_natural_reasoning 25000 --seed 42 \
  --concurrency 256 --embedding_device cuda:2 \
  --output_file translated_reasoning_25k.jsonl \
  --units_output_file translated_reasoning_25k_units.jsonl
```

- **Output (document-level, for the HF release):** `translated_reasoning_25k.jsonl` — one row per document, with `en_cot_answer`/`hi_cot_answer` (full reconstructed text) plus per-unit detail.
- **Output (chunk-level, for training):** `translated_reasoning_25k_units.jsonl` — one row per translation unit, written incrementally as each chunk finishes (safe to read while the run is still in progress).
- **Resumable:** if this process dies or is killed, rerunning the exact same command will skip every document already in `translated_reasoning_25k.jsonl` (matched by id) and only translate what's missing. Same `--seed 42` is required for the *sample* to be reproducible; the resume-by-id logic works regardless of seed.
- **Log:** `/tmp/claude-30001/.../scratchpad/translate_25k_25k.log` (session-local scratch dir, will not survive past this session — the two jsonl output files above are the durable artifacts).

## Decisions made without you in the loop

### 1. OpenThoughts3 sampling: streaming, NOT stratified

You explicitly said "no need of stratification" and "we don't need the full
openthoughts dataset, just 25K" as your last two instructions before leaving,
superseding the earlier AskUserQuestion answer (which had picked stratified +
full 71GB download). I implemented a new function,
`sample_openthoughts3_streaming()` in `data_gen/sample_reasoning.py`, using
single-pass reservoir sampling (Algorithm R) over the HF streaming iterator:

- Unbiased uniform random sample of 25,000 rows from the full 1.2M.
- Does **not** stratify by domain/source/difficulty — plain uniform draw.
- Does **not** cache the full ~71GB dataset to local disk (streaming mode).
- Still reads through all 1.2M rows over the network once (unavoidable —
  reservoir sampling needs to see every row to decide inclusion), so this
  isn't free, but it avoids the local storage/cache cost.
- I validated the reservoir algorithm's correctness against a small synthetic
  stream (uniform coverage check, correct output length in normal and
  short-stream edge cases) before launching — see the conversation transcript
  for the check — but did **not** get to independently verify the *actual*
  1.2M-row OT3 sample's domain/difficulty distribution after the fact, since
  the run was launched and left going. If you want to confirm the sample
  isn't accidentally skewed (e.g. if the dataset's row order correlates with
  some property), that's worth a quick post-hoc check against
  `translated_reasoning_25k.jsonl`'s `metadata.domain`/`metadata.difficulty`
  fields once it's done.

**If this wasn't what you wanted** (e.g. you actually did want stratification
for the final release, and "no need of stratification" meant something
narrower): the original `sample_openthoughts3()` stratified function is still
in `sample_reasoning.py`, untouched — swap the import back in
`translate_reasoning.py`'s `main()` and rerun with a fresh `--output_file` to
get a stratified sample instead. The unstratified run doesn't need to be
discarded either way — it's a valid, just differently-sampled, dataset.

### 2. NaturalReasoning sampling: unchanged

Uses the existing `sample_natural_reasoning()` — always was uniform (never
stratified, since natural_reasoning has no categorical metadata to stratify
on), full ~4GB download. This matches "same 25K for natural reasoning" as you
confirmed. No open question here.

### 3. Concurrency: 256

You said "increase the concurrency to even higher number, I think it can
take it" without a specific value. I picked 256 to match the historical
`CONCURRENCY = 256` constant already used elsewhere in this repo
(`data_gen/translate_ds.py`) as a reasonable, previously-validated ceiling
for this kind of workload — not empirically re-tuned for this specific vllm
instance. If the vllm server (`Qwen/Qwen3-4B-Instruct-2507`, single GPU,
`vllm-gpu3`, port 8077) is visibly struggling (check `make vllm-logs
NAME=vllm-gpu3` for queueing/errors, or watch GPU3 utilization via
`nvidia-smi`), consider restarting the run with a lower value — nothing about
256 was load-tested end-to-end before this launch, only inferred from prior
project convention.

### 4. Embedding model device: GPU 2

You pointed me at GPU 2 earlier for the LaBSE similarity check
(`--embedding_device cuda:2`). At launch time GPU 2 had ~77GB free (only
~4.8GB used). I didn't re-check GPU 2's occupancy immediately before this
launch — if something else has since claimed it, the run would hit a CUDA
OOM and the current document would fail (logged as `document failed: ...`,
document skipped, loop continues — **not** a full crash, per the existing
per-document try/except in `main()`). Worth a `nvidia-smi` check if you see
a cluster of `document failed` lines in the log.

### 5. Timeline / stopping point

I did not set an explicit stop point — this will run until all 50,000
documents (25K OT3 + 25K NR) are processed, or until it's manually killed.
Given the earlier 100+100 validation run took ~24 minutes for ~21K chunks at
concurrency=48, and this run has ~5x the concurrency but ~250x the document
count, expect this to run for many hours (the earlier "20+ hour" estimate
from before the concurrency bump was rough, and 256 concurrency should cut
that down meaningfully, but I have no hard number — the vllm server's actual
compute ceiling, not the concurrency setting, determines the real throughput
once you're past the point where more in-flight requests just queue instead
of parallelizing).

## What's *not* new since the last time we talked

Everything else about the pipeline is exactly what we validated together
before you left:
- `<think>`/`</think>` tag stripping (content preserved, only the literal tag tokens removed) — fixed and confirmed correct on real data.
- N-gram repetition check (catches 1-5 token repeated phrases, not just single tokens).
- LaBSE embedding-similarity check wired into the retry loop (`similarity_floor=0.5` default), sharing the existing higher-temperature retry mechanism.
- Orphan-unit absorption + code-island reclassification in `chunking.py` (fixes the `token_count=1` standalone-fragment issue from earlier).
- All 29 tests in `tests/test_chunking.py` passing as of this run.

## Suggested next steps when you're back

1. Check the run is still alive: `ps aux | grep translate_reasoning`.
2. Check progress: `wc -l translated_reasoning_25k.jsonl translated_reasoning_25k_units.jsonl`.
3. If you want a live look while it's still running, `streamlit run app.py` → "Reasoning Translations" tab → "Live progress" section reads the units file incrementally.
4. Decide whether the unstratified OT3 sample is acceptable for the HF release, or whether to redo with stratification (see point 1 above).

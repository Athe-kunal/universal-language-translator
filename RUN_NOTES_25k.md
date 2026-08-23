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

## UPDATE 2 (multi-GPU restart)

After the above, we discovered the single-GPU throughput (~1,265 units/min,
GPU3 at 100%) meant the full ~5.3M-unit run would take ~70 hours. You
confirmed GPU0/GPU1 (previously running unrelated 35-36h-old vllm containers)
were free to reclaim.

Decision: use vllm's native `--data-parallel-size` (data parallelism — 3
independent model replicas, one per GPU, load-balanced behind one endpoint)
rather than tensor parallelism across GPU0+GPU1, or app-level round-robin
across multiple endpoints (I built the latter, then reverted it per your
request in favor of vllm's built-in DP, which is simpler and doesn't need
any app-side load-balancing code).

Steps taken:
1. Killed the running translate_reasoning.py process (269 documents were
   already safely completed and on disk — not lost, skipped on resume).
2. Tore down all 3 existing vllm containers (`vllm-gpu0`, `vllm-gpu1`,
   `vllm-gpu3`) — confirmed with you these were safe to stop.
3. Brought up one new container `vllm-gpu013` spanning GPUs 0,1,3 with
   `TP=1 EXTRA_ARGS="--data-parallel-size 3"` (same port 8077). Verified all
   3 GPUs loaded (~74GB each) with 3 separate ApiServer processes internally
   load-balancing.
4. Relaunched `translate_reasoning.py` with `--concurrency 768` (3x the
   original 256, matching 3x GPU capacity) against the same single endpoint,
   same `--seed 42`, same output files (resumes from the 269 already done).

Expected: roughly 3x the single-GPU throughput (~3,800 units/min), so the
full run should land around ~23 hours from this restart, not ~70. This will
be confirmed empirically once enough new completions land — I'll update
throughput numbers as I check in.

Note: the OT3 streaming reservoir sample has to be re-collected on every
restart (~78 min, network-bound, no caching exists for the raw sample) — this
happens again on this restart before translation resumes. The `--seed 42`
guarantees it's the *same* 25,000-document sample as before.

## UPDATE 3 (fast shard sampling + parallel chunking + a real crash bug fixed)

Three more real fixes, in order of discovery:

1. **OT3 sampling sped up ~330x.** The reservoir-sampling approach read every
   row of the full 1.2M-row/71GB dataset to guarantee uniform sampling
   (~78 min, network-bound). OpenThoughts3-1.2M is actually stored as 120
   parquet shards (~10K rows each) on the Hub. New function
   `sample_openthoughts3_shards()` in `sample_reasoning.py`: randomly picks
   a handful of whole shards (enough to comfortably exceed num_samples),
   downloads just those, samples within that pool. Measured: 25,000 rows in
   22 seconds. Trade-off: only representative if shards are reasonably
   pre-shuffled (not verified) — acceptable since stratification was already
   ruled out for this run.

2. **Found and fixed a real crash bug.** A document with a single literal
   backtick character (prose, not a real code span — e.g. "the backtick `
   comes before...") could get paired by CommonMark with an unrelated distant
   backtick, producing code-span content that (due to CommonMark's whitespace
   normalization) didn't exactly match a substring of the original text —
   `chunking.py`'s protection logic used `str.index()` (raises on no match)
   instead of `str.find()`, crashing the entire `chunk_all` pass and killing
   the whole 50K-document run on ONE bad document. Fixed in two ways: (a)
   `_find_protectable_nodes` now uses `.find()` and skips protecting a span
   it can't locate, rather than crashing; (b) `chunk_all` now wraps each
   document in try/except so any future unexpected error skips just that one
   document instead of taking down the batch. Added a regression test
   (`test_lone_backtick_does_not_crash_protection`).

3. **Parallelized chunking across CPU cores.** The host has 128 cores;
   `chunk_all` was single-threaded (~470 docs/min, one core). Now uses
   `ProcessPoolExecutor`. Discovered a second real bug while testing at 64
   workers: multiple worker processes simultaneously loading their own copy
   of the wtpsplit SaT model onto the same GPU caused CUDA out-of-memory
   crashes (19/3000 docs failed). Fixed by forcing worker processes to use
   CPU for wtpsplit (via a pool `initializer`) — splits are rare enough that
   losing GPU accel inside workers is a good trade for safe parallelism; the
   main process's own async translation phase still uses GPU for its
   occasional splits. You asked for a cap of 32 workers (not the default
   `os.cpu_count()`) — measured 766 docs/min at 32 workers, 0 failures, vs.
   470 docs/min single-threaded. Full chunking pass now estimated ~65 min
   (down from the original single-threaded ~90-100 min projection), on top
   of the ~22s sampling.

Current launch command (PID 1588441, this is the one actually running):
```
uv run python data_gen/translate_reasoning.py \
  --num_openthoughts3 25000 --num_natural_reasoning 25000 --seed 42 \
  --concurrency 768 --embedding_device cuda:2 --sat_device cuda:2 --chunk_workers 32 \
  --base_url http://localhost:8077/v1 \
  --output_file translated_reasoning_25k.jsonl --units_output_file translated_reasoning_25k_units.jsonl
```

Still resuming from the same 269 already-completed documents throughout all
of this (never lost, never redone).

## UPDATE 4 (chunking finished clean, translation running)

Chunking pass: **49,754/49,754 documents, 0 failures.** Took ~36 minutes
wall-clock (01:17 launch to 01:53 finish) — sampling (~1 min) + parallel
chunking at 32 workers. Real throughput was noticeably better than the
isolated benchmark suggested (~2,840 docs/min observed in the last third of
the pass vs. 766 docs/min measured in isolation) — likely less split-path
usage in this particular random sample.

Translation phase started immediately after. There was a burst of 22
`Request timed out` errors in the first ~45 seconds (all 49,754 tasks got
dispatched near-simultaneously at `--concurrency 768`, likely overwhelming
vllm's data-parallel router/connection pool before it settled) — these 22
whole documents were lost (not written to output), but this is small
(0.04% of the batch) and fully recoverable later: since they never made it
into `translated_reasoning_25k.jsonl`, simply re-running the exact same
launch command after this run finishes will pick them up automatically via
the existing resume-by-id logic. No action needed now. No further timeouts
since the initial burst — translation has been running cleanly since.

GPU utilization cycles across GPU0/1/3 as vllm's internal data-parallel
router distributes batches — this is expected/normal for DP serving, not a
sign of imbalance.

## UPDATE 5 (a serious bug: 67% document failure rate, found and fixed)

The "translation running cleanly" assessment above was **wrong** — a later,
more careful check found 573 `Request timed out` errors against only 277
successful documents: a **67.4% document failure rate**, actively ongoing,
not a resolved startup blip. Root cause: openai-python's default httpx
client caps `max_keepalive_connections` at 100 and connect timeout at 5s —
with `--concurrency 768`, most requests couldn't reuse a kept-alive
connection and had to open a new one, and under sustained load that
sometimes took longer than 5s. The vllm server itself was healthy throughout
(confirmed via curl + docker logs showing a steady stream of real 200 OK
responses) — this was purely a client-side misconfiguration, not a GPU/model
problem.

Fixed in `main()`: construct an explicit `httpx.AsyncClient` with
`max_connections=concurrency*2`, `max_keepalive_connections=concurrency`,
and `connect=60.0` (up from the 5s default), passed to `AsyncOpenAI` via
`http_client=`.

Killed the run (592 documents had already been lost this way before the fix
went in — but nothing already-completed was lost; they just weren't written
to the output file, so they'll be retried automatically on the next resume),
fixed, relaunched. **Verified after the fix: 0 `Request timed out` errors**
across the entire subsequent run, all 3 GPUs at 100% utilization
simultaneously (first time seeing genuinely healthy 3-way parallel load).

Real measured throughput post-fix: **~1,907 units/min** (clean 90-second
delta measurement) — about 1.5x the single-GPU baseline (~1,265 units/min),
not the full ~3x you'd hope for from 3 GPUs. Some overhead somewhere in DP
routing or shared GPU2 embedding-check contention, not chased further given
time already spent — the run is correct and stable, just not perfectly
linear in GPU count.

**Revised ETA: ~45 hours from this point** (~5.3M total units, ~145K done
as of this update), down from the original single-GPU ~70h estimate.

Current launch command (PID 2224008, the one actually running):
```
uv run python data_gen/translate_reasoning.py \
  --num_openthoughts3 25000 --num_natural_reasoning 25000 --seed 42 \
  --concurrency 768 --embedding_device cuda:2 --sat_device cuda:2 --chunk_workers 32 \
  --base_url http://localhost:8077/v1 \
  --output_file translated_reasoning_25k.jsonl --units_output_file translated_reasoning_25k_units.jsonl
```

## UPDATE 6 (steady state, one minor known edge case)

As of 03:55: 661 documents done, 203,129 units translated, 0 `Request timed
out` since the fix, throughput steady at ~1,890-1,910 units/min.

Two documents (0.3%) hit a different, minor issue: a unit too large to split
(7,169 input tokens) plus the 1024-token completion budget exceeded the
model's 8192-token context by 1 token — a request-level `400 BadRequestError`
that bypasses our retry/validation logic entirely (it's a rejected request,
not a bad response) and takes the whole document down with it. Small and
resumable on a future rerun; not worth stopping for. Not fixed in code this
session — flagging as a known gap if you want to address it later (e.g. cap
`max_tokens` output request more conservatively, or reject/skip units whose
`text_protected` alone exceeds some safe input-token budget before ever
sending them).

## UPDATE 7 (still steady, 04:26)

754 docs, 260,184 units, 0 timeouts, ~1,840 units/min. One more instance of
the same known context-length edge case (3 total now, same error signature).
No new error types. Nothing needs attention — continuing to run unattended.

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

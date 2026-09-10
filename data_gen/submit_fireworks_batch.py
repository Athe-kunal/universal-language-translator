"""Two-phase driver for a Fireworks Batch Inference API job over the
request files `translate_fireworks.py` prepares, via `firectl` (Fireworks'
own CLI) rather than hand-rolled HTTP calls - the raw Dataset/
BatchInferenceJob REST schema (upload endpoint, output row format) isn't
fully documented anywhere, and firectl is Fireworks' supported client for
exactly this.

Split into two phases because `firectl dataset create` / `firectl
batch-inference-job create` (the two calls that actually spend money) are
BLOCKED when run by an AI agent ("mutating command ... cannot run inside an
AI agent") - a deliberate safety guard, not a bug to route around:

  prepare: safe to run unattended. Waits for translate_fireworks.py's
    output to stabilize, concatenates batch_requests_*.jsonl into one
    upload-ready file, estimates cost, and prints the exact `firectl`
    commands to run - but does not run them.
  poll: safe to run unattended, but only AFTER a human has run the two
    printed commands and the job exists. Polls `firectl batch-inference-job
    get` until COMPLETED/FAILED/EXPIRED, then downloads the output dataset.

Neither phase re-assembles a final {"id", "steps": [{"en","hi"}]} dataset:
the output dataset's exact row schema isn't documented (Fireworks' own docs
only say "a results file in JSONL format"), so rather than guess it blindly
this stops at the raw download for inspection: a separate step (once the
real schema is confirmed against real output) joins it with units.jsonl via
custom_id.

Usage:
    uv run python -m data_gen.submit_fireworks_batch prepare --batch_dir fireworks_batch_30k
    # ... run the two printed firectl commands yourself ...
    uv run python -m data_gen.submit_fireworks_batch poll --batch_dir fireworks_batch_30k \\
        --job_id reasoning-hi-30k --output_dataset_id reasoning-hi-30k-output
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from data_gen import config

DEFAULT_POLL_INTERVAL = 120
TERMINAL_STATES = {"COMPLETED", "FAILED", "EXPIRED"}


def _normalize_state(state: str) -> str:
    """Strips the proto enum's "JOB_STATE_" prefix if present - firectl's
    `-o json` returns the raw enum name (e.g. "JOB_STATE_COMPLETED"), not
    the short form the batch-inference docs use ("COMPLETED"); this lets
    TERMINAL_STATES/callers compare against the short form regardless.
    """
    return state.removeprefix("JOB_STATE_")
# 8B params -> 4B-16B pricing tier: $0.20/1M tokens serverless, halved by
# the Batch API's documented 50% discount. Uniform for input/output (no
# separate output rate published at this tier) - see
# https://docs.fireworks.ai/serverless/pricing.
_PRICE_PER_1M_TOKENS_BATCH = 0.10


def _setup_logging(batch_dir: Path, name: str) -> logging.Logger:
    batch_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(batch_dir / f"{name}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    return logging.getLogger(name)


def run_firectl(args: list[str], log: logging.Logger) -> str:
    """Runs a read-only firectl command, returning its stdout. Never call
    this with a mutating subcommand (create/delete/update) - those are
    blocked for an agent and must be run by a human; see module docstring.

    Args:
        args: firectl subcommand and flags (without the "firectl" prefix).
        log: Logger for the command being run.

    Returns:
        Captured stdout, stripped.

    Raises:
        subprocess.CalledProcessError: If firectl exits non-zero.
    """
    log.info(f"$ firectl {' '.join(args)}")
    result = subprocess.run(["firectl", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def estimate_cost(manifest_file: Path) -> float:
    """Rough cost estimate from a translate_fireworks.py manifest.json,
    assuming Hindi output is roughly as many tokens as the English input
    (a coarse but directionally-useful assumption - actual Hindi/Devanagari
    tokenization can run higher or lower depending on the serving
    tokenizer).

    Args:
        manifest_file: Path to manifest.json.

    Returns:
        Estimated USD cost (input + assumed-equal output, at the Batch
        API's discounted per-token rate).
    """
    manifest = json.loads(manifest_file.read_text())
    input_tokens = manifest["input_tokens"]["total_input_tokens"]
    assumed_output_tokens = input_tokens  # coarse 1:1 assumption
    total_tokens = input_tokens + assumed_output_tokens
    return total_tokens / 1_000_000 * _PRICE_PER_1M_TOKENS_BATCH


def concatenate_batch_requests(batch_dir: Path, out_file: Path) -> int:
    """Concatenates every batch_requests_*.jsonl in `batch_dir` into one
    file - a single Fireworks Dataset upload (its documented size limit is
    80GiB, comfortably above what this pipeline produces) is simpler to
    track than one job per file.

    Args:
        batch_dir: Directory with translate_fireworks.py's output.
        out_file: Path to write the concatenated JSONL to.

    Returns:
        Number of rows written.
    """
    request_files = sorted(batch_dir.glob("batch_requests_*.jsonl"))
    if not request_files:
        raise FileNotFoundError(f"No batch_requests_*.jsonl found in {batch_dir}")
    n = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for f in request_files:
            with open(f, encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    n += 1
    return n


def wait_for_prep(units_file: Path, log: logging.Logger, poll_interval: int = 30) -> None:
    """Blocks until `translate_fireworks.py` has finished writing
    `units_file` (i.e. the run producing it has exited) - detected via the
    file existing and being stable (unchanged size across two checks), so
    this can be launched before the prep run finishes and pick up right
    after.

    Args:
        units_file: Path to the prep run's units.jsonl.
        log: Logger for progress messages.
        poll_interval: Seconds between stability checks.
    """
    log.info(f"Waiting for {units_file} to exist and stabilize...")
    last_size = -1
    stable_checks = 0
    while stable_checks < 2:
        if units_file.exists():
            size = units_file.stat().st_size
            if size == last_size and size > 0:
                stable_checks += 1
            else:
                stable_checks = 0
            last_size = size
        time.sleep(poll_interval)
    log.info(f"{units_file} stable at {last_size} bytes - prep run finished.")


def cmd_prepare(args: argparse.Namespace) -> None:
    log = _setup_logging(args.batch_dir, "prepare")

    units_file = args.batch_dir / "units.jsonl"
    if args.wait_for_prep:
        wait_for_prep(units_file, log)

    manifest_file = args.batch_dir / "manifest.json"
    cost = estimate_cost(manifest_file)
    log.info(f"Estimated cost ({config.FIREWORKS_MODEL}, Batch API, assuming ~1:1 output/input tokens): ${cost:.2f}")

    concatenated = args.batch_dir / "batch_requests_all.jsonl"
    n_rows = concatenate_batch_requests(args.batch_dir, concatenated)
    log.info(f"Concatenated {n_rows} rows into {concatenated}")

    job_id = args.job_id or args.batch_dir.name
    output_dataset_id = f"{job_id}-output"
    runbook = f"""
Ready. This job's estimated cost is ${cost:.2f} — run these two commands
yourself (they're blocked for an agent by Fireworks' own safety guard):

  firectl dataset create {job_id} {concatenated} --display-name "reasoning_hi input ({args.batch_dir.name})"

  firectl batch-inference-job create \\
      --input-dataset-id {job_id} \\
      --output-dataset-id {output_dataset_id} \\
      --model {config.FIREWORKS_MODEL} \\
      --job-id {job_id} \\
      --max-job-duration {args.max_job_duration}

Then, to poll to completion and download (safe to run unattended):

  uv run python -m data_gen.submit_fireworks_batch poll --batch_dir {args.batch_dir} \\
      --job_id {job_id} --output_dataset_id {output_dataset_id}
"""
    log.info(runbook)
    print(runbook)


def cmd_poll(args: argparse.Namespace) -> None:
    log = _setup_logging(args.batch_dir, "poll")
    log.info(f"Polling job {args.job_id} every {args.poll_interval}s...")

    state = "UNKNOWN"
    status_json = "{}"
    while True:
        status_json = run_firectl(["batch-inference-job", "get", args.job_id, "-o", "json"], log)
        status = json.loads(status_json)
        state = _normalize_state(status.get("state", "UNKNOWN"))
        log.info(f"job={args.job_id} state={state}")
        if state in TERMINAL_STATES:
            break
        time.sleep(args.poll_interval)

    if state != "COMPLETED":
        log.error(f"Job ended in state={state}, not downloading output. Full status:\n{status_json}")
        sys.exit(1)

    download_dir = args.batch_dir / "fireworks_output"
    run_firectl(["dataset", "download", args.output_dataset_id, "--output-dir", str(download_dir)], log)
    units_file = args.batch_dir / "units.jsonl"
    log.info(f"Downloaded output dataset to {download_dir}")
    log.info(
        "Raw Fireworks output saved locally. Next step (not run here - schema needs confirming against real "
        f"output first): join {download_dir}'s results with {units_file} on custom_id, restore ⟦N⟧ placeholders "
        "via each unit's spans, and group into {\"id\", \"steps\": [{\"en\",\"hi\"}]} records."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Concatenate requests, estimate cost, print the commands to run.")
    prep.add_argument("--batch_dir", type=Path, required=True, help="Dir from translate_fireworks.py.")
    prep.add_argument("--job_id", default=None, help="Dataset/job id to use (default: batch_dir's name).")
    prep.add_argument("--max_job_duration", default="48h")
    prep.add_argument("--wait_for_prep", action="store_true", help="Block until units.jsonl stabilizes first.")
    prep.set_defaults(func=cmd_prepare)

    poll = sub.add_parser("poll", help="Poll an existing job to completion and download its output.")
    poll.add_argument("--batch_dir", type=Path, required=True)
    poll.add_argument("--job_id", required=True)
    poll.add_argument("--output_dataset_id", required=True)
    poll.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL)
    poll.set_defaults(func=cmd_poll)

    args = parser.parse_args()
    load_dotenv()
    args.func(args)


if __name__ == "__main__":
    main()

"""Shared OpenAI-compatible client helpers, factored out so any script that
translates via chat completions - translate_ds.py, translate_fireworks.py's
online mode, or a future vendor - builds/calls the same client instead of
reimplementing it per script. Swap vendors by pointing base_url/api_key/
model at a different one (local vllm, Fireworks, OpenAI itself); the call/
concurrency/parsing logic here doesn't change.

Also holds the OpenAI Batch API helpers (file upload -> batch create ->
poll -> download), via litellm - used by translate_ds_batch.py. This is
strictly OpenAI's batch resource model (file+batch). Fireworks' Batch
Inference API is a different Dataset+BatchInferenceJob resource model (see
https://docs.fireworks.ai/guides/batch-inference) and isn't implemented
here yet - its dataset file-upload endpoint isn't confirmed from docs, so
translate_fireworks.py currently only prepares Fireworks batch request
files rather than submitting them.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import litellm
from openai import AsyncOpenAI


class AsyncChatClient:
    """Thin wrapper around an OpenAI-compatible AsyncOpenAI client: builds
    the client once, gates concurrency with a semaphore, and extracts the
    response text - the part every online-translation script repeats.
    """

    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)

    async def complete(self, messages: list[dict], **kwargs) -> str:
        """One chat completion, concurrency-gated.

        Args:
            messages: Chat messages, as OpenAI's API expects.
            **kwargs: Passed straight through to the API call (temperature,
                max_tokens, ...).

        Returns:
            The completion's text content, stripped.
        """
        async with self.sem:
            resp = await self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return (resp.choices[0].message.content or "").strip()


async def submit_batch(
    requests: list[dict],
    log: logging.Logger,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
) -> str:
    """Uploads `requests` (OpenAI batch rows) and creates an OpenAI Batch
    API job.

    Args:
        requests: OpenAI batch rows ({"custom_id", "method", "url", "body"}).
        log: Logger for progress messages.
        endpoint: The batch endpoint every row's `url` targets.
        completion_window: OpenAI's batch completion window.

    Returns:
        The created batch's id.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for req in requests:
            tmp.write(json.dumps(req) + "\n")
        tmp_path = tmp.name

    try:
        log.info(f"Uploading {len(requests)} requests ({Path(tmp_path).stat().st_size / 1024:.1f} KB)...")
        with open(tmp_path, "rb") as fh:
            file_obj = await litellm.acreate_file(file=fh, purpose="batch", custom_llm_provider="openai")
        log.info(f"File uploaded: {file_obj.id}")

        batch = await litellm.acreate_batch(
            completion_window=completion_window,
            endpoint=endpoint,
            input_file_id=file_obj.id,
            custom_llm_provider="openai",
        )
        log.info(f"Batch created: {batch.id}")
        return batch.id
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def poll_batch(batch_id: str, log: logging.Logger, interval: int = 60) -> str:
    """Polls an OpenAI Batch API job until it reaches a terminal state.

    Args:
        batch_id: The batch id from `submit_batch`.
        log: Logger for progress messages.
        interval: Seconds between status polls.

    Returns:
        The output_file_id, once the batch completes.

    Raises:
        RuntimeError: If the batch ends in a non-"completed" terminal state.
    """
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        status = await litellm.aretrieve_batch(batch_id=batch_id, custom_llm_provider="openai")
        counts = status.request_counts
        log.info(
            f"batch={batch_id} status={status.status} "
            f"completed={getattr(counts, 'completed', '?')} "
            f"failed={getattr(counts, 'failed', '?')} "
            f"total={getattr(counts, 'total', '?')}"
        )
        if status.status in terminal:
            if status.status != "completed":
                raise RuntimeError(f"Batch {batch_id} ended with status={status.status}")
            return status.output_file_id
        await asyncio.sleep(interval)


async def download_batch(output_file_id: str, log: logging.Logger) -> dict[str, str]:
    """Downloads and parses an OpenAI Batch API output file.

    Args:
        output_file_id: From `poll_batch`.
        log: Logger for progress messages.

    Returns:
        {custom_id: translated_text}, for successful rows only (errored
        rows are logged and skipped).
    """
    log.info(f"Downloading output file {output_file_id}...")
    content = await litellm.afile_content(file_id=output_file_id, custom_llm_provider="openai")
    raw = content.content if hasattr(content, "content") else content

    results: dict[str, str] = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id", "")
        if row.get("error"):
            log.warning(f"custom_id={cid} error: {row['error']}")
            continue
        text = row["response"]["body"]["choices"][0]["message"]["content"] or ""
        results[cid] = text.strip()

    log.info(f"Parsed {len(results)} results from output file")
    return results

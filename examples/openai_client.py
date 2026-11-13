#!/usr/bin/env python3
"""Convenience client: query any OpenAI-compatible endpoint with SILICA prompts.

This is a minimal reference client for running new models.  It is NOT the
sealed runner that produced the results reported in the paper, which pinned
model snapshots, tokenizers, serving engines and generation settings and
recorded their provenance.

Usage::

    silica prompts --dataset DATASET --cohort common --output prompts.jsonl
    python examples/openai_client.py --prompts prompts.jsonl --output predictions.jsonl \\
        --base-url http://HOST:PORT/v1 --model MODEL
    silica evaluate --dataset DATASET --predictions predictions.jsonl

The client sends each rendered chat unchanged with greedy decoding and writes
one record per row in the release prediction format.  The message ``content``
is the scored final answer; a separately returned reasoning trace is stored in
``reasoning_text`` and never scored.  An API key, if needed, is read from the
environment variable named by ``--api-key-env``.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def complete(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Send one chat-completions request and return its decoded JSON body."""
    body = json.dumps(
        {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def predict(args: argparse.Namespace, prompt: dict[str, Any]) -> dict[str, Any]:
    """Query the endpoint for one prompt and return a release-format prediction."""
    try:
        response = complete(args, prompt["messages"])
        choice = response["choices"][0]
        message = choice.get("message") or {}
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
        return {
            "id": prompt["id"],
            "response_text": None,
            "finish_reason": None,
            "completion_tokens": None,
            "request_error": True,
        }
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    finish = choice.get("finish_reason")
    prediction = {
        "id": prompt["id"],
        "response_text": content if isinstance(content, str) else "",
        "finish_reason": finish if finish in ("stop", "length") else None,
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
        "request_error": False,
    }
    if isinstance(reasoning, str):
        prediction["reasoning_text"] = reasoning
    return prediction


def main() -> int:
    """Query every prompt and write the predictions file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, required=True, help="output of `silica prompts`")
    parser.add_argument("--output", type=Path, required=True, help="predictions JSONL to write")
    parser.add_argument("--base-url", required=True, help="endpoint root, e.g. http://HOST:PORT/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    with args.prompts.open(encoding="utf-8") as handle:
        prompts = [json.loads(line) for line in handle]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        predictions = list(pool.map(lambda prompt: predict(args, prompt), prompts))
    with args.output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    errors = sum(item["request_error"] for item in predictions)
    print(f"wrote {len(predictions)} predictions to {args.output} ({errors} request errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

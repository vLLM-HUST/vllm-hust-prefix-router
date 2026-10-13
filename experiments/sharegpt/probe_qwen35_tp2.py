"""Issue one real generation request to a Qwen TP2 service."""

from __future__ import annotations

import argparse
import json
import time
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="prefix-router-evidence")
    args = parser.parse_args()

    payload = json.dumps(
        {
            "model": args.model,
            "prompt": "In one sentence, explain tensor parallel inference.",
            "max_tokens": 24,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = Request(
        args.url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
        status = response.status
        route = response.headers.get("x-vllm-hust-router-node")
    choices = result.get("choices", [])
    if status != 200 or not choices or not choices[0].get("text"):
        raise RuntimeError(f"incomplete generation: status={status}, {result}")
    print(
        json.dumps(
            {
                "status": status,
                "model": result.get("model"),
                "route": route,
                "text": choices[0]["text"],
                "finish_reason": choices[0].get("finish_reason"),
                "usage": result.get("usage"),
                "elapsed_s": round(time.monotonic() - start, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

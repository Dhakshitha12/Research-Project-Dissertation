"""Replay identical held-out arrival arrays against one or more OpenFaaS functions."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import heldout_arrivals


def invoke(url: str, payload: bytes, timeout: float):
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        decoded = json.loads(body) if body else {}
        return response.status, elapsed_ms, bool(decoded.get("cold_start", False))


def payload_for_url(url: str) -> bytes:
    function_name = url.rstrip("/").rsplit("/", 1)[-1]
    if "image" in function_name:
        event = {
            "width": 64,
            "height": 64,
            "out_width": 32,
            "out_height": 32,
            "pixels": [(i * 17) % 256 for i in range(64 * 64 * 3)],
        }
    elif "json" in function_name:
        event = {"records": [
            {"id": i, "value": (i % 997) / 7, "group": f"Group-{i % 17}",
             "active": i % 3 != 0}
            for i in range(1000)
        ]}
    else:
        raise ValueError(
            f"Cannot infer workload payload for {function_name!r}; "
            "use an OpenFaaS function named with 'image' or 'json'."
        )
    return json.dumps(event, separators=(",", ":")).encode()


async def replay_one(label, url, family, episode, arrivals, speedup, timeout, rows):
    loop = asyncio.get_running_loop()
    start = loop.time()
    request_index = 0
    payload = payload_for_url(url)
    for second, count in enumerate(arrivals):
        target = start + second / speedup
        delay = target - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks = [
            loop.run_in_executor(None, invoke, url, payload, timeout)
            for _ in range(int(count))
        ]
        results = await asyncio.gather(*tasks)
        for status, elapsed_ms, cold_start in results:
            rows.append({
                "function": label,
                "family": family,
                "episode": episode,
                "request": request_index,
                "status": status,
                "latency_ms": elapsed_ms,
                "cold_start": cold_start,
            })
            request_index += 1
    print(f"replayed {label}: {family} episode {episode}, {request_index} requests",
          flush=True)


async def run(args):
    functions = {}
    for item in args.function:
        if "=" not in item:
            raise ValueError("--function must be LABEL=URL")
        label, url = item.split("=", 1)
        functions[label] = url.rstrip("/")
    if not functions:
        raise ValueError("Provide at least one --function LABEL=URL.")

    arrivals = heldout_arrivals()
    selected = [
        (family, episode, stream)
        for (family, episode), stream in arrivals.items()
        if (not args.family or family in args.family)
    ]
    rows = []
    for family, episode, stream in selected:
        for label, url in functions.items():
            await replay_one(label, url, family, episode, stream, args.speedup,
                             args.timeout, rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "function", "family", "episode", "request", "status", "latency_ms",
            "cold_start",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", action="append", default=[],
                        help="Paired target in LABEL=URL form; repeat for each function.")
    parser.add_argument("--family", action="append", default=[],
                        help="Held-out family to replay; repeat to select multiple.")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="Compress arrival timing; 1.0 preserves the trace clock.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default="../results/openfaas_replay.csv")
    args = parser.parse_args()
    if args.speedup <= 0 or args.timeout <= 0:
        parser.error("--speedup and --timeout must be positive.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

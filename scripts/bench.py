#!/usr/bin/env python3
"""Latency and throughput of the native module. One configuration per run:

    uv run python scripts/bench.py --model Qwen/Qwen3-0.6B --device cpu --threads 8

The short request is the one rust-cev's scripts/bench.py sends (a ticket and 3
questions); the long one pads the state to about 2.3k tokens. "first" is the
very first request (calibration not cached yet), "new state" changes the state
every call, "same state" repeats it (prefix cached). Throughput sends new
states from `--parallel` threads.
"""

import argparse
import itertools
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from cev import Cev

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Payments, invoices, refunds", "tech": "Bugs, crashes, errors", "sales": "New purchases and upgrades"},
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "severity": {
        "type": "score",
        "instructions": "How severe is the issue?",
        "criteria": ["Cosmetic; no impact", "Degraded, workaround exists", "Blocking; no workaround"],
    },
}
TICKET = {"subject": "App crashes on login", "body": "Since the update, the iOS app crashes as soon as I tap Log in."}
HISTORY = [
    {
        "at": f"2026-09-{i % 28 + 1:02d}T10:{i:02d}:00Z",
        "from": "agent" if i % 2 else "customer",
        "text": f"Follow-up {i}: still seeing the crash after reinstalling, clearing the cache and restarting the phone. Build 4.{i}.",
    }
    for i in range(40)
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--debias", default="calibrate,full")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--requests", type=int, default=32)
    args = ap.parse_args()

    t = time.perf_counter()
    cev = Cev.load(args.model, device=args.device, dtype=args.dtype, threads=args.threads, concurrency=args.concurrency)
    print(
        f"{args.model} on {args.device}: loaded in {time.perf_counter() - t:.1f} s, "
        f"{cev.model['threads']} threads, concurrency {cev.model['concurrency']}"
    )
    nonce = itertools.count()

    def decide(state, debias):
        t = time.perf_counter()
        r = cev.decide(state, QUESTIONS, debias=debias, no_store=True)
        return (time.perf_counter() - t) * 1e3, r["usage"]["input_tokens"]

    for debias in args.debias.split(","):
        for name, base in (("short", TICKET), ("long", {**TICKET, "history": HISTORY})):
            fresh = lambda: {**base, "nonce": f"run-{next(nonce)}"}  # noqa: E731
            first, tokens = decide(fresh(), debias)
            uncached = statistics.median(decide(fresh(), debias)[0] for _ in range(args.runs))
            state = fresh()
            decide(state, debias)
            cached = statistics.median(decide(state, debias)[0] for _ in range(args.runs))
            print(f"  {debias:9} {name:5} {tokens:5} tok: first {first:5.0f} ms, new state {uncached:5.0f} ms, same state {cached:5.0f} ms")
        t = time.perf_counter()
        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(lambda _: decide({**TICKET, "nonce": f"run-{next(nonce)}"}, debias), range(args.requests)))
        per_sec = args.requests / (time.perf_counter() - t)
        print(f"  {debias:9} throughput: {per_sec:.1f} requests/s from {args.parallel} threads")
    cev.close()


if __name__ == "__main__":
    main()

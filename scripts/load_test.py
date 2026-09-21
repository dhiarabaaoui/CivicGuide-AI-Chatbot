"""Small dependency-free load test for the no-cost lexical retrieval endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--maximum-error-rate", type=float, default=0.01)
    parser.add_argument("--maximum-p95-ms", type=float, default=2500.0)
    return parser.parse_args()


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def one_request(base_url: str) -> tuple[float, int]:
    body = json.dumps({
        "message": "What documents are required to register a vehicle?",
        "domain": "dmv",
        "use_dense": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/retrieve",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    return (time.perf_counter() - started) * 1000, status


def main() -> int:
    args = parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be positive")

    results: list[tuple[float, int]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, args.base_url) for _ in range(args.requests)]
        for future in as_completed(futures):
            results.append(future.result())

    durations = [duration for duration, _ in results]
    errors = sum(status < 200 or status >= 300 for _, status in results)
    elapsed = time.perf_counter() - started
    report = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "errors": errors,
        "error_rate": errors / len(results),
        "requests_per_second": len(results) / elapsed,
        "latency_ms": {
            "mean": statistics.fmean(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations),
        },
    }
    print(json.dumps(report, indent=2))
    passed = (
        report["error_rate"] <= args.maximum_error_rate
        and report["latency_ms"]["p95"] <= args.maximum_p95_ms
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

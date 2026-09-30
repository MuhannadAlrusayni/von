"""Serial latency measurement mirroring JevBench's Speed protocol.

JevBench Speed (composite_v12.py): score(s) = clamp(100 - 20*log10(s/0.1)),
Speed = mean(score(p50), score(p95)) over the serial standard+judge run.
Self-hosted endpoints get raw*2 + 0.15 s before scoring (an assumption in
their method, not a measurement). Jev-class needs adjusted p50 <= 1.30 s,
i.e. raw p50 <= 0.575 s.

This client sends one request at a time over HTTP to a running `von serve`,
timing the full round trip, exactly one question per request as the official
`typesafe` adapter does. The public dataset has no judge tier, so the
"standard+judge" run is approximated by the 72 standard items; hard-tier
p50/p95 are reported separately as JevBench does.

Usage:
    uv run python benchmarks/measure_latency.py --url http://127.0.0.1:8000 \
        --public ~/scratch/jevbench/datasets/public --out results/latency_local.json \
        --hardware "Intel i7-1260P, openvino:cpu" --endpoint-kind cpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Dict, List

import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hard_fast import build_request  # noqa: E402

TIERS = {"standard": "original.jsonl", "hard": "hard.jsonl", "easy": "easy.jsonl"}


def load(public: str, tier: str) -> List[dict]:
    with open(os.path.join(public, TIERS[tier]), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def payload(row: dict) -> dict:
    state, instructions, criteria = build_request(row, "plain")
    qtype = row["question"].get("type", "choice")
    crit = dict(criteria) if qtype != "score" else [criteria[k] for k in sorted(criteria, key=int)]
    return {"state": state, "questions": {"q": {"type": qtype, "instructions": instructions, "criteria": crit}}}


def post(url: str, body: dict, timeout: float = 300.0) -> float:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} from server: {e.read().decode(errors='replace')[:2000]}")
    return time.perf_counter() - t0


def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def speed_point(s: float) -> float:
    return max(0.0, min(100.0, 100 - 20 * math.log10(s / 0.1)))


def adjusted(s: float, kind: str) -> float:
    return s if kind == "api" else s * 2.0 + 0.15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--public", default=os.path.expanduser("~/scratch/jevbench/datasets/public"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--endpoint-kind", default="cpu", choices=["cpu", "gpu", "api"])
    ap.add_argument("--device", default="")
    ap.add_argument("--tiers", default="standard,hard")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the standard tier")
    args = ap.parse_args()

    tiers = args.tiers.split(",")
    rows = {t: load(args.public, t) for t in tiers}

    # Warm-up: the first calls pay graph compilation / cache population.
    for row in rows[tiers[0]][: args.warmup]:
        post(args.url, payload(row))

    per_tier: Dict[str, List[float]] = {}
    per_item: List[dict] = []
    for t in tiers:
        lat: List[float] = []
        reps = args.repeat if t == "standard" else 1
        for _ in range(reps):
            for row in rows[t]:
                s = post(args.url, payload(row))
                lat.append(s)
                per_item.append({"id": row["id"], "tier": t, "seconds": s})
        per_tier[t] = lat
        print(f"{t:9s} n={len(lat):3d} p50={pct(lat,.5):.3f}s p95={pct(lat,.95):.3f}s mean={statistics.mean(lat):.3f}s")

    std = per_tier.get("standard", [])
    p50, p95 = pct(std, 0.5), pct(std, 0.95)
    a50, a95 = adjusted(p50, args.endpoint_kind), adjusted(p95, args.endpoint_kind)
    speed = (speed_point(a50) + speed_point(a95)) / 2
    jev_class = a50 <= 1.30

    result = {
        "system": "von-395m",
        "checkpoint": "von-1.2",
        "endpoint_kind": args.endpoint_kind,
        "hardware": args.hardware,
        "device": args.device,
        "measured_where": "serial HTTP client on the same host, model loaded and warmed before timing",
        "run": f"serial {len(std)}-decision standard run (public has no judge tier)",
        "python": platform.python_version(),
        "speed": {
            "p50_s_raw": p50, "p95_s_raw": p95,
            "p50_s_adjusted": a50, "p95_s_adjusted": a95,
            "adjustment": "none" if args.endpoint_kind == "api" else "x2 + 0.15 s (JevBench self-hosted assumption)",
            "score": speed,
            "hard_tier_p50_s": pct(per_tier.get("hard", []), 0.5) if "hard" in per_tier else None,
            "hard_tier_p95_s": pct(per_tier.get("hard", []), 0.95) if "hard" in per_tier else None,
        },
        "jev_class_latency_line": {"adjusted_p50_max_s": 1.30, "raw_p50_max_s": 0.575, "passes": jev_class},
        "per_item": per_item,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSpeed score {speed:.1f}  adjusted p50 {a50:.3f}s  Jev-class latency line: {'PASS' if jev_class else 'FAIL'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

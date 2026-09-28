"""Chain-of-options gate: Von-1.2 plain vs Von-1.2 + chains, paired McNemar.

Runs the same checkpoint twice on a set of public items (default: the pooled
numeric-sensitive slice from stat_gate.py, n=114), once with VON_CHAINS_DIR
unset and once pointing at the chain library, and reports the paired gate.
Per-item traces (route, bindings, computed values, description, fallback)
are written so a wrong chain answer can be blamed on routing, binding,
execution, or the final read.

    uv run python benchmarks/probe_chains.py --out benchmarks/data/chains_gate.json
    uv run python benchmarks/probe_chains.py --family temporal_numeric --out benchmarks/data/chains_temporal.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_hard_fast import build_request  # noqa: E402
from stat_gate import is_numeric_item, mcnemar_gate  # noqa: E402

PUBLIC = os.environ.get("JEVBENCH_PUBLIC", os.path.expanduser("~/scratch/jevbench/datasets/public"))
FILES = {"hard": "hard.jsonl", "standard": "original.jsonl", "easy": "easy.jsonl"}


def load_items(tiers: List[str]) -> List[dict]:
    rows = []
    for t in tiers:
        with open(os.path.join(PUBLIC, FILES[t]), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_tier"] = t
                    rows.append(r)
    return rows


def _hit(row: dict, pick) -> bool:
    exp = row["expected"]
    qtype = row["question"].get("type", "choice")
    if qtype == "noul":
        return str(pick).lower() in ("yes", "true") if str(exp).lower() in ("yes", "true") else str(pick).lower() in ("no", "false")
    if qtype == "score":
        try:
            return int(round(float(pick))) == int(exp)
        except (TypeError, ValueError):
            return False
    return pick == exp


def score_one(backend, row: dict):
    state, instructions, criteria = build_request(row, "plain")
    qtype = row["question"].get("type", "choice")
    payload = dict(criteria) if qtype != "score" else [criteria[k] for k in sorted(criteria, key=int)]
    backend.last_chain_trace = None
    t0 = time.perf_counter()
    res = backend.evaluate(state=state, questions={"q": {"type": qtype, "instructions": instructions, "criteria": payload}})
    dt = time.perf_counter() - t0
    a = res.answers["q"]
    if qtype == "noul":
        pick = "yes" if a.noul >= 0.5 else "no"
    elif qtype == "score":
        pick = a.score
    else:
        pick = a.choice
    tr = backend.last_chain_trace
    if tr is not None and not isinstance(tr, dict):
        tr = dataclasses.asdict(tr)
        tr["values"] = {k: (str(v) if not isinstance(v, (int, float, str, type(None))) else v) for k, v in tr["values"].items()}
    return {"id": row["id"], "family": row.get("family"), "tier": row["_tier"], "type": qtype,
            "pick": pick, "expected": row["expected"], "hit": _hit(row, pick), "seconds": dt,
            "surface": (row.get("provenance") or {}).get("surface_answer"), "trace": tr}


def run(rows: List[dict], chains_dir: str | None, device: str) -> List[dict]:
    if chains_dir:
        os.environ["VON_CHAINS_DIR"] = chains_dir
    else:
        os.environ["VON_CHAINS_DIR"] = "off"
    from von.backends.option_marker_backend import OptionMarkerBackend
    b = OptionMarkerBackend(checkpoint_dir=os.path.join(ROOT, "checkpoints/von-1.2"), device=device)
    b._get_model()
    out = []
    for i, r in enumerate(rows):
        res = score_one(b, r)
        out.append(res)
        tag = f"chain={res['trace'].get('chain')}" if res["trace"] else "plain"
        print(f"[{i+1:3d}/{len(rows)}] {r['id'][-28:]:28s} {'HIT ' if res['hit'] else 'miss'} {tag}", flush=True)
    return out


def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=os.path.join(ROOT, "src/von/chains/library"))
    ap.add_argument("--family", default="")
    ap.add_argument("--tiers", default="hard,standard")
    ap.add_argument("--device", default="openvino:cpu")
    ap.add_argument("--mode", default="route", help="chain runner mode: route | bindall (VON_CHAINS_MODE)")
    ap.add_argument("--baseline", default="", help="reuse a previous plain-run json instead of rerunning")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load_items(a.tiers.split(","))
    if a.family:
        rows = [r for r in rows if r.get("family") == a.family]
    else:
        rows = [r for r in rows if is_numeric_item(r)]
    print(f"{len(rows)} items")

    if a.baseline:
        base = {r["id"]: r for r in json.load(open(a.baseline))["baseline"]}
        base = [base[r["id"]] for r in rows]
    else:
        print("== plain ==")
        base = run(rows, None, a.device)
    print("== chains ==")
    os.environ["VON_CHAINS_MODE"] = a.mode
    cand = run(rows, a.chains, a.device)

    gate = mcnemar_gate([r["hit"] for r in base], [r["hit"] for r in cand], label="chains vs plain")
    routed = sum(1 for r in cand if r["trace"] and r["trace"].get("chain"))
    fell = {}
    for r in cand:
        if r["trace"] and r["trace"].get("fallback"):
            k = r["trace"]["fallback"].split(":")[0]
            fell[k] = fell.get(k, 0) + 1
    summary = {"n": len(rows), "plain_acc": sum(r["hit"] for r in base) / len(rows),
               "chains_acc": sum(r["hit"] for r in cand) / len(rows), "routed_to_chain": routed,
               "fallbacks": fell, "gate": gate,
               "plain_mean_s": sum(r["seconds"] for r in base) / len(rows),
               "chains_mean_s": sum(r["seconds"] for r in cand) / len(rows)}
    print(json.dumps(summary, indent=1, default=str))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "baseline": base, "candidate": cand}, open(a.out, "w"), indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Input tokens per decision as Von reports them (usage.input_tokens), per tier.

JevBench's Cost axis is a size-class hosted tariff times the system's own
input-token count. This measures that count on the public items through the
real backend, and prices it at the encoder-class tariff JevBench applied to
Laya ($0.01 / M input tokens, $0 output).

    uv run python benchmarks/measure_tokens.py --out results/speed/tokens_public.json
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
from eval_hard_fast import build_request  # noqa: E402

PUBLIC = os.environ.get("JEVBENCH_PUBLIC", os.path.expanduser("~/scratch/jevbench/datasets/public"))
FILES = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}
TARIFF_USD_PER_M = 0.01


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="openvino:cpu")
    a = ap.parse_args()
    os.environ["VON_CHAINS_DIR"] = "off"
    from von.backends.option_marker_backend import OptionMarkerBackend
    b = OptionMarkerBackend(checkpoint_dir=os.path.join(ROOT, "checkpoints/von-1.2"), device=a.device); b._get_model()
    per_tier, items = {}, []
    for tier, fn in FILES.items():
        counts = []
        for line in open(os.path.join(PUBLIC, fn), encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            state, ins, crit = build_request(r, "plain")
            qtype = r["question"].get("type", "choice")
            q = {"type": qtype, "instructions": ins}
            if qtype == "score":
                q["criteria"] = [crit[k] for k in sorted(crit, key=int)]
            elif crit:
                q["criteria"] = crit
            res = b.evaluate(state=state, questions={"q": q})
            n = res.usage.input_tokens
            counts.append(n); items.append({"id": r["id"], "tier": tier, "type": qtype, "input_tokens": n})
        per_tier[tier] = {"n": len(counts), "mean": st.mean(counts), "median": st.median(counts), "p95": sorted(counts)[int(0.95 * (len(counts) - 1))],
                          "usd_per_1000": st.mean(counts) / 1e6 * TARIFF_USD_PER_M * 1000}
        print(f"{tier:9s} n={len(counts):3d} mean={st.mean(counts):6.1f} median={st.median(counts):6.1f} p95={per_tier[tier]['p95']:5d}  ${per_tier[tier]['usd_per_1000']:.5f}/1k")
    # JevBench's v1.1-tier figure is easy+standard; hard reported separately
    es = [i["input_tokens"] for i in items if i["tier"] in ("easy", "standard")]
    summary = {"tariff_usd_per_M_input": TARIFF_USD_PER_M, "basis": "encoder-class hosted tariff as applied to laya in v1.4",
               "easy+standard_mean_tokens": st.mean(es), "usd_per_1000_v11_tiers": st.mean(es) / 1e6 * TARIFF_USD_PER_M * 1000,
               "usd_per_1000_hard": per_tier["hard"]["usd_per_1000"], "all_mean_tokens": st.mean([i["input_tokens"] for i in items]),
               "laya_reference": {"tokens": 205, "usd_per_1000": 0.00288}, "von_board_reference": {"usd_per_1000": 0.00551, "basis": "reconstructed v1.3, unmeasured"}}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "per_tier": per_tier, "items": items}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

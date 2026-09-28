# Von-1.2 Speed remeasurement (2026-09-27)

JevBench v1.4 lists `von-395m` with Speed 70.5 carried over from a v1.3 run
with `endpoint_kind: unknown`, `hardware: null`, no raw p50/p95. That row
currently puts Von outside Jev-class ("Speed below the 2x latency line")
even though the number was never measured on any recorded hardware.

Protocol: jevbench `docs/METHOD-v1.4.md`. Serial HTTP client on the same
host as `von serve`, model loaded and warmed before timing, one decision
per request, standard-tier public items (72; public has no judge tier) plus
the 111 hard-tier items reported separately. Self-hosted adjustment
`raw x2 + 0.15 s`. Jev-class line: adjusted p50 <= 1.30 s, i.e. raw p50
<= 0.575 s. Client: `benchmarks/measure_latency.py`; launcher:
`benchmarks/launch_speed_measure.py` (self-terminating EC2, results to S3).

## Results

| endpoint_kind | hardware | device | p50 raw | p95 raw | p50 adj | p95 adj | hard p50 raw | hard p95 raw | Speed | Jev-class line |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu | AWS c7i.xlarge, Intel Xeon Platinum 8488C (Sapphire Rapids), 4 vCPU | openvino:cpu | **0.096 s** | 0.110 s | 0.341 s | 0.369 s | 0.339 s | 3.84 s | **89.0** | PASS |
| gpu | AWS g5.xlarge, NVIDIA A10G 24 GB | cuda | **0.023 s** | 0.024 s | 0.195 s | 0.197 s | 0.039 s | 0.387 s | **94.2** | PASS |
| (board, stale) | unknown | unknown | — | — | — | — | — | — | 70.5 | "below line" |

Reference rows from the same board: Laya p50 raw 0.787 s / adjusted 1.72 s
(Speed 71.1, outside Jev-class); hopper (Qwen3.5-4B, A6000) p50 raw 0.129 s
(Speed 86.8).

Raw per-item latencies and metadata: `results/speed/latency_cpu_20260927_2139.json`,
`results/speed/latency_gpu_20260927_2144.json`. Cloud spend: c7i.xlarge
~25 min + g5.xlarge ~10 min, under $1.

Both endpoints clear the line by a wide margin, so goal step 2b (depth
pruning) is skipped. Cost is unchanged by this measurement: the JevBench
Cost axis is a tariff times measured input tokens, not runtime.

## Submission payload (benchmarkheaven / jevbench issue)

Request: re-measure `von-395m` Speed, or accept the CPU row below as the
self-hosted measurement (it is the conservative one of the two).

```json
{
  "key": "von-395m",
  "display": "Von 1.2 (395M)",
  "repo": "https://github.com/wfzyx/von",
  "underlying": "ModernBERT-large encoder + option-marker decision head, 395M, OpenVINO-accelerated",
  "licence": "Apache-2.0",
  "open": true,
  "endpoint_kind": "cpu",
  "endpoint_condition": "AWS c7i.xlarge (4 vCPU Intel Xeon Platinum 8488C), von serve --device openvino:cpu, local loopback HTTP; serial",
  "speed": {
    "p50_s_raw": 0.0955,
    "p95_s_raw": 0.1095,
    "p50_s_adjusted": 0.3410,
    "p95_s_adjusted": 0.3691,
    "adjustment": "x2 + 0.15 s (assumption, not measured)",
    "run": "serial 72-decision standard run, public items",
    "hardware": "AWS c7i.xlarge, Intel Xeon Platinum 8488C, 4 vCPU, OpenVINO CPU",
    "measured_where": "same host as the server, model loaded and warmed before timing",
    "hard_tier_p50_s": 0.3390,
    "hard_tier_p95_s": 3.8399
  },
  "speed_gpu_alternate": {
    "p50_s_raw": 0.0232,
    "p95_s_raw": 0.0237,
    "p50_s_adjusted": 0.1955,
    "p95_s_adjusted": 0.1973,
    "hardware": "AWS g5.xlarge, NVIDIA A10G 24 GB, cuda",
    "hard_tier_p50_s": 0.0388,
    "hard_tier_p95_s": 0.3868
  },
  "reproduce": {
    "install": "pip install von-sdk==1.2.3  (Von 1.2 row)",
    "serve": "von serve --host 127.0.0.1 --port 8123 --device openvino:cpu",
    "measure": "python benchmarks/measure_latency.py --url http://127.0.0.1:8123 --out latency.json",
    "raw": "results/speed/latency_cpu_20260927_2139.json"
  }
}
```

## Composite projection

Holding Intelligence 34.5, Calibration 75.7 and Cost 77.8 fixed and replacing
Speed 70.5 with the measured 89.0 (CPU), `jevbench.composite_v12.jevbench_score`
(equal-weight geometric mean, before the v1.4 sealed-gap penalty) moves from
61.5 to 65.2 (x1.06). The published composite 27.5 carries the same
multiplicative penalty, so the projected published score is ~29.1 (GPU row:
66.1, ~29.5). Laya sits at 30.3. More important than the number: the row
moves from "outside Jev-class" to inside the Jev-class capability ranking,
which Laya (adjusted 1.72 s) does not.

## Cost: real token count (2026-09-28)

`usage.input_tokens` now sums the encoder's real `input_ids` length over every
forward pass the request ran (was `len(chars)//4`, never tokenizer-backed).
Measured on the 231 public items via `benchmarks/measure_tokens.py`, priced at
the encoder-class tariff JevBench applied to Laya ($0.01/M input):

| tier | n | mean tok | median | p95 | $/1k |
|---|---|---|---|---|---|
| easy | 48 | 56 | 58 | 75 | 0.00056 |
| standard | 72 | 68 | 64 | 88 | 0.00068 |
| hard | 111 | 1103 | 507 | 3018 | 0.01103 |

Blended (314 v1.1 + 220 hard, JevBench's fixture weights): **$0.00492/1k →
Cost 79.3** (board row: 0.00551 → 77.8, reconstructed from v1.3, unmeasured).
Laya: 205 tok/decision, $0.00288/1k, Cost 86.2.

`--max-state-tokens` probe on the 111 public hard items (paired McNemar vs
uncapped Von-1.2): cap 512 → Δ 0.0 pp (5/5 discordant), cap 1024 → Δ 0.0 pp
(3/3 discordant). Projected Cost at cap 512: $0.00202/1k → **90.8**; at 1024:
86.2. Not submitted; a cap is a serving choice to be declared with the run.


## Von 1.3: chains on, same weights (2026-09-28)

Same protocol, `von serve` with the bundled chain library active (bindall).

| endpoint | raw p50 | raw p95 | adjusted p50 | Speed | hard p50 | hard p95 | Jev-class |
|---|---|---|---|---|---|---|---|
| c7i.xlarge, openvino:cpu | 0.104 s | 0.125 s | 0.358 s | **88.4** | 4.25 s | 75.3 s | PASS |
| g5.xlarge A10G, cuda | 0.023 s | 0.023 s | 0.195 s | **94.2** | 0.45 s | 6.6 s | PASS |

Raw: `results/speed/latency_cpu_chains.json`, `results/speed/latency_gpu_chains.json`.
Chains add up to 16 encoder sub-decisions on numeric hard items; the overall
p50 is unmoved (standard/easy dominate), the hard tail is where it lands.

## Submission: Von 1.3 (new row, same weights)

```json
{
  "key": "von-1.3",
  "display": "Von 1.3 (395M, chain-of-options)",
  "repo": "https://github.com/wfzyx/von",
  "underlying": "ModernBERT-large encoder + option-marker head (von-1.2 weights, unchanged) + deterministic chain-of-options controller; zero generated tokens",
  "licence": "Apache-2.0",
  "open": true,
  "endpoint_kind": "cpu",
  "endpoint_condition": "AWS c7i.xlarge (4 vCPU Xeon 8488C), von serve --device openvino:cpu, loopback, serial; GPU alternate g5.xlarge A10G",
  "reproduce": {
    "install": "pip install von-sdk==1.3.0",
    "serve": "von serve --host 127.0.0.1 --port 8123 --device openvino:cpu",
    "measure": "python benchmarks/measure_latency.py --url http://127.0.0.1:8123 --out latency.json",
    "raw": "results/speed/latency_cpu_chains.json"
  },
  "usage": "usage.input_tokens is the tokenizer count summed over every encoder pass the request ran (chains included)"
}
```

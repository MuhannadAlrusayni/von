# Von

**An open-source, non-autoregressive System One decision model.**
*Calibrated Choice / Noul / Score inference from a 395M ModernBERT encoder, one forward pass, CPU-served.*

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-wfzyx%2Fvon-blue)](https://huggingface.co/wfzyx/von)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)

<p align="center">
  <img src="assets/von-doom.gif" alt="Von choosing movement and combat actions in Doom" width="640">
</p>
<p align="center"><sub><b>Von playing Doom.</b> Every action is one forward pass scoring six option descriptions against a text rendering of the depth buffer. Same shipped weights that answer routing questions; no policy network, no RL.</sub></p>

Von answers three question types over any JSON or text *state* without generating tokens: **Choice** (pick one of K described options, with a probability over all of them), **Noul** (probability that a condition holds) and **Score** (calibrated position on an ordinal scale). It is wire-compatible with the TypeSafe `/v1/systemone` specification, ships as `von-sdk` for Python and TypeScript, and runs on CPU (OpenVINO), CUDA, ROCm and Apple MPS.

## Results

JevBench v1.4 (the public System One benchmark; scores from `results/v1.4/jevbench-v1.4-results.json` in the [jevbench repo](https://github.com/fstandhartinger/jevbench)). I/C/S/K are the Intelligence, Calibration, Speed and Cost axes; the composite is their equal-weight geometric mean with JevBench's generalization gate applied. Public-tier accuracies are chance-uncorrected fractions. Latency is JevBench's adjusted p50 (self-hosted rows: raw ×2 + 0.15 s). Cost is JevBench's tariff estimate per 1,000 decisions.

| system | params | I | C | S | K | composite | easy | standard | hard | sealed | p50 latency | $/1k | endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (TypeSafe, closed) | — | 53.1 | 76.3 | 83.3 | 52.0 | **63.3** | 1.000 | 0.990 | 0.741 | 0.367 | 0.65 s | 0.040 | api |
| hopper (Qwen3.5-4B + LoRA) | 4B | 48.0 | 79.1 | 86.8 | 58.7 | 59.4 | 1.000 | 0.969 | 0.650 | 0.341 | 0.41 s | 0.024 | gpu |
| Qwen3.5-4B frozen (semif) | 4B | 44.4 | 66.8 | 83.7 | 59.5 | 47.7 | 1.000 | 0.979 | 0.595 | 0.263 | 0.55 s | 0.022 | gpu |
| jeff (GLiFormer-large) | ~400M | 36.8 | 67.9 | 63.5 | 76.6 | 30.6 | 1.000 | 0.760 | 0.377 | 0.331 | 2.03 s | 0.006 | cpu |
| Laya (ModernBERT-large + option marker) | 421M | 36.1 | 63.7 | 71.1 | 86.2 | 30.3 | 0.944 | 0.729 | 0.341 | 0.308 | 1.72 s | 0.003 | cpu |
| **Von 1.2** (this repo) | **395M** | 34.5 | **75.7** | 70.5¹ | 77.8 | 27.5 | 0.931 | 0.688 | 0.373 | 0.279 | **0.34 s**¹ | 0.006 | cpu |
| GLiNER2-large | ~300M | 31.1 | 24.8 | 61.7 | 73.3 | 15.1 | 0.986 | 0.625 | 0.364 | 0.286 | 2.34 s | 0.008 | cpu |

¹ The board's Speed 70.5 for Von is a v1.3 carry-over with no recorded hardware. Remeasured 2026-09-27 under the JevBench protocol: raw p50 **0.096 s** on a 4-vCPU Xeon 8488C (OpenVINO), adjusted 0.34 s, Speed 89.0; **0.023 s** raw on an A10G, Speed 94.2. Both are inside the Jev-class latency line (≤ 1.30 s adjusted). Details and the submission payload: [`results/speed_remeasure.md`](results/speed_remeasure.md).

Older, longer tables (jabr v2 49-task suite, ViZDoom, training-data coverage) live in [`docs/benchmarks.md`](docs/benchmarks.md).

### Von vs Laya

Same trunk class (ModernBERT-large encoder + option-marker head), same endpoint kind. Every number is from the JevBench v1.4 results file unless marked.

| metric | Von 1.2 | Laya | source |
|---|---|---|---|
| parameters | 395M | 421M | model cards |
| Intelligence | 34.5 | 36.1 | v1.4 axes |
| Calibration | 75.7 | 63.7 | v1.4 axes |
| Speed (board) | 70.5 (unmeasured carry-over) | 71.1 | v1.4 axes |
| Speed (remeasured, CPU) | 89.0 | — | `results/speed_remeasure.md` |
| Cost | 77.8 | 86.2 | v1.4 axes |
| Composite | 27.5 | 30.3 | v1.4 |
| public easy / standard / judge / hard | 0.931 / 0.688 / 0.685 / 0.373 | 0.944 / 0.729 / 0.692 / 0.341 | v1.4 tiers |
| sealed accuracy (308 items) | 0.279 | 0.308 | v1.4 sealed |
| sealed ECE | 0.107 | 0.172 | v1.4 sealed |
| adjusted p50 latency | 0.34 s (remeasured) | 1.72 s | v1.4 / remeasure |
| Jev-class latency line (≤ 1.30 s) | inside (remeasured) | outside | v1.4 rule |
| $ per 1,000 decisions (estimate) | 0.0055 | 0.0029 | v1.4 cost |
| order-invariant option scoring | yes (independent-options masking) | not stated | this repo |

Where Laya leads: standard tier (+4.1 pp), Cost, sealed accuracy. Where Von leads: Calibration (+12.0), hard tier (+3.2 pp), sealed ECE, and, once the remeasurement is on the board, Speed and Jev-class eligibility.

## Quickstart

```bash
pip install von-sdk          # or: uv add von-sdk
bun add von-sdk              # TypeScript / Node
```

```python
import von

r = von.decide(
    state="Database replication lag on cluster us-west-2 exceeded 45 seconds.",
    choices={
        "infrastructure": "Database, hardware, network, or server failures",
        "billing": "Invoices, payments, refunds, subscription queries",
        "feature_request": "Requests for new platform capabilities",
    },
    instructions="Classify the root cause domain of this incident.",
)
r.choice, r.confidence, r.probabilities   # 'infrastructure', 0.84, {...}

p = von.judge(state="Connection pool exhausted on port 5432.", instructions="Is this blocking customers?")
s = von.rate(state="Memory at 98%, OOM killer firing.", criteria=["Nominal", "Degraded", "Critical"],
             instructions="Assess degradation level.")
```

Several questions over one state cost one forward pass:

```python
resp = von.system_one(
    state={"ticket": "INC-4091", "message": "Payment gateway timeouts on charge authorizations. Urgent."},
    questions={
        "intent": von.choice(instructions="Nature of the ticket?", criteria={"payment_failure": "...", "access_issue": "..."}),
        "is_urgent": von.noul(instructions="Needs immediate SLA intervention?"),
        "severity": von.score(instructions="Rate severity.", criteria=["Low", "Medium", "High", "Critical"]),
    },
)
resp.answers["intent"].choice, resp.answers["is_urgent"].noul, resp.answers["severity"].score
```

TypeScript mirrors the same API (`decide`, `judge`, `rate`, `systemOne`) and can also talk to any `/v1/systemone` server via `VON_BASE_URL`.

### Server

```bash
von serve --host 0.0.0.0 --port 8000                 # auto-selects cuda / mps / openvino:gpu / openvino:cpu / cpu
curl -X POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "von-1.2.0",
  "state": {"error": "Disk volume /var/log at 98% capacity."},
  "questions": {"needs_action": {"type": "noul", "instructions": "Does this require operational intervention?"}}
}'
```

### Container

A CPU and a CUDA image are in [PR #12](https://github.com/wfzyx/von/pull/12); once it lands: `docker run -p 8000:8000 ghcr.io/wfzyx/von:cpu`. Until then build from the `Dockerfile` on that branch.

## Configuration

| flag / env | default | effect |
|---|---|---|
| `--device` / `VON_DEVICE` | `auto` | `cuda`, `mps`, `openvino:gpu`, `openvino:cpu`, `cpu`. Auto prefers OpenVINO CPU over plain torch CPU. |
| `--max-state-tokens N` / `VON_MAX_STATE_TOKENS` | 8192 | States longer than N tokens are middle-truncated (60 % head, 40 % tail) so question and options always fit the 8192 window. Truncated responses carry a `truncation` field and `X-Von-Truncated` / `Warning` headers. |
| `--chains DIR` / `VON_CHAINS_DIR` | off | Enable chain-of-options (below). |
| `VON_API_KEY` | unset | Bearer token required by the server when set (client falls back to `TYPESAFE_API_KEY`). |
| `VON_MODEL_ID` | `wfzyx/von` | Hugging Face repo or local checkpoint directory. |

## Chain-of-options (experimental)

Deterministic multi-step computation for temporal/numeric items, driven entirely by Von's own Choice decisions and zero generated tokens:

1. a regex proposer lists candidate spans (dates, durations, time zones, amounts, percents, tables) — it never decides;
2. Von picks which chain applies (or `none`) as a Choice over chain descriptions;
3. Von binds each typed slot as a Choice over the proposed spans of that kind;
4. a fixed operator library (`add_duration`, `in_zone`, `elapsed_hours`, `prorate`, `cumsum`, …) executes;
5. the computed result is described in one sentence and the original question is asked about it.

Chains are TOML files (`src/von/chains/library/`: deadline+timezone, month-window/leap, proration, cumulative-vs-limit, elapsed-window). Enable with `von serve --chains src/von/chains/library`. Any routing miss, unbound slot or executor error falls back to the plain answer. Gate status: paired McNemar on the pooled 114-item numeric slice, reported in `PROGRESS.md`; off by default until it clears.

## Training and calibration

Von 1.2 is ModernBERT-large with an option-marker head, trained with a listwise softmax cross-entropy + Brier objective on ~63k operational decision items plus synthetic two-hop and numeric sets, with independent-options attention masking so option order cannot change the answer. Calibration is an input-conditioned temperature map fitted post hoc (`checkpoints/von-1.2/marker_calibration.json`). Every accuracy claim in this repo goes through `benchmarks/stat_gate.py` (paired exact McNemar + minimum detectable effect); results below the MDE are reported as UNRESOLVABLE, never as wins.

## License

Apache-2.0.

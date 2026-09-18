# 🪨 Von (`von`)

**The Open-Source System One Decision Model.**  
*Sub-15ms decisions. 14MB footprint. Zero cloud latency. Zero API keys. Zero VC tax.*  
*Named in homage to **John von Neumann** and **Ludwig von Mises**.*

---

## Why Von?

TypeSafe AI raised $40M to charge $0.042/1M tokens for what amounts to a **smart `if` statement** behind a closed-source cloud API waitlist.

**Von** is 100% open-source and runs locally on your machine. Powered by the `cactus-needle` 3.x Simple Attention Network (SAN), Von strips out autoregressive text generation bloat to deliver pure, structured, calibrated decisions in **under 15 milliseconds** on a standard CPU.

- **Non-Autoregressive:** Evaluates all questions in a single forward pass.
- **Zero Hallucinations:** Structurally guaranteed output types. No Markdown drift, no JSON formatting errors.
- **Epistemically Calibrated:** Confidence scores and probability distributions that reflect statistical reality.
- **14MB Binary:** Runs in ~28MB RAM on CPU. No GPU required.
- **Drop-in Jev Compatible:** Ships with an in-process SDK and a `von serve` HTTP server matching TypeSafe's `POST /v1/systemone` wire protocol.

---

## The Name: Von

Named in homage to two giants of decision theory and computation:
1. **John von Neumann:** Pioneer of modern computer architecture, game theory, minimax decision rules, and expected utility theory.
2. **Ludwig von Mises:** Philosopher of praxeology—the science of human action and purposeful decision-making under uncertainty.

---

## Multiple Backends

Von supports three distinct backends depending on your memory and accuracy budget:

```python
import von

# 1. Needle 3 (Default): 14MB footprint, ~28MB RAM, pure CPU reflex (<15ms)
von.set_backend("needle")

# 2. ModernBERT-151M: GLiClass candidate slot scoring with Brier calibration (~35ms)
von.set_backend("modernbert")

# 3. Qwen 0.5B (PCD): Parallel Constrained Decoding reading logits directly
von.set_backend("qwen0.5b")
```

### Benchmark Comparison (OpenJev `authored144` Suite)

| Backend / Model | Weights | Hardware / Env | Balanced Acc | Acc / Weight (%/MB) | Latency / Call | VRAM Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Von (`needle`)** | **14 MB** | **CPU (In-Process)** | **46.7%** | **3.336% / MB** 🏆 | **~38 ms** (embed) | **0 MB** |
| **Von (`modernbert`)** | **290 MB** | **CPU (In-Process)** | **45.7%** | **0.158% / MB** | **~35 ms** | **0 MB** |
| **Von (`qwen0.5b`)** | **942 MB** | **CPU (In-Process)** | **64.4%** | **0.068% / MB** | **~650 ms (CPU)** | **0 MB** |
| Qwen3-0.6B (OpenJev) | 639 MB | GPU / WebGPU | 44.0% | 0.069% / MB | ~35 ms | 1.2 GB |
| MiniCPM5-2B (OpenJev) | 1.56 GB | GPU / WebGPU | 68.6% | 0.044% / MB | ~40 ms | 3.5 GB |
| Qwen3.5-4B (OpenJev) | 3.01 GB | RTX 3090 (24GB) | 81.3% | 0.027% / MB | ~48 ms | 8.0 GB |
| Published Jev (TypeSafe) | ~8–16 GB* | Closed Cloud API | 88.3% | ~0.005–0.011%/MB | 100–300 ms | Cloud (MoE) |

*\* Jev weight estimate based on Archer Hume's reverse-engineering analysis across 10,000 API calls, indicating a causal sparse MoE backbone (~8B–14B total parameters, ~2B active parameters).*

---

## Installation

```bash
pip install von
# or with uv
uv add von
```

---

## The Three Primitives

Von implements the three core System One question types:

| Primitive | Question Type | Output Shape | When to Use |
| :--- | :--- | :--- | :--- |
| **`Noul`** | Is this true? | `noul: float` (0.0 to 1.0) | Yes/No judgments (e.g. `refund_requested`, `is_urgent`) |
| **`Choice`** | Which of these options? | `choice`, `probabilities`, `confidence` | Categorical routing (e.g. `department`, `intent`) |
| **`Score`** | Where on this scale? | `score`, `probabilities`, `legend`, `confidence` | Continuous ordered scales (e.g. `bug_severity`, `frustration`) |

---

## Quickstart

### 1. Fast Discrete Decisions (`von.decide`)

```python
import von

decision = von.decide(
    "My card was charged twice for order #1234 and I want my money back!",
    choices=["billing_refund", "technical_bug", "feature_request"],
)

print(decision.choice)         # 'billing_refund'
print(decision.confidence)     # 0.85
print(decision.probabilities)  # {'billing_refund': 0.88, 'technical_bug': 0.08, ...}
```

### 2. Yes/No Probability Judgments (`von.judge`)

```python
import von

p_urgent = von.judge(
    "Production database is locked and customer writes are failing!",
    instructions="Is this an urgent or blocking production outage?",
)

print(p_urgent)  # 0.96
if p_urgent > 0.8:
    page_on_call()
```

### 3. Continuous Scale Rating (`von.rate`)

```python
import von

rating = von.rate(
    "The export button crashes only on Safari 17.2 with error code 4",
    criteria=[
        "Cosmetic; no impact on core functionality",
        "Broken or degraded feature, but a workaround exists",
        "Blocking issue; no workaround exists",
    ],
)

print(rating.score)       # 1.15 (between level 1 and level 2)
print(rating.confidence)  # 0.72
```

---

## Speculative Fan-Out (`von.system_one`)

Ask all independent questions against your state in a single call. No latency multiplier:

```python
import von

state = {
    "opportunity": "newsletter_sponsorship",
    "company": "Managed Postgres",
    "message": "We make a managed PostgreSQL hosting product and would like to sponsor your newsletter in October.",
}

questions = {
    "is_sponsor_inquiry": von.noul(
        instructions="Does `message` ask to sponsor the newsletter?"
    ),
    "category": von.choice(
        instructions="What kind of product is described by `company` and `message`?",
        criteria={
            "dev_tool": "Developer tools, hosting, databases, APIs",
            "course": "Books, video courses, training",
            "unrelated": "Non-developer consumer products",
        },
    ),
    "quality": von.score(
        instructions="How specific is the sponsorship request?",
        criteria=[
            "Generic pitch template, no concrete ask",
            "Mentions the brand but no timeframe",
            "Concrete ask with timeframe and product named",
        ],
    ),
}

res = von.system_one(state=state, questions=questions)

answers = res.answers
print(answers["is_sponsor_inquiry"].noul)   # 0.98
print(answers["category"].choice)            # 'dev_tool'
print(answers["quality"].score)              # 1.82

# Pure code branches on calibrated judgment
if answers["is_sponsor_inquiry"].noul > 0.8 and answers["category"].choice == "dev_tool":
    send_rate_card(state)
```

---

## Running the HTTP Server (`von serve`)

Need a drop-in replacement for TypeSafe's cloud API? Start the local server:

```bash
von serve --port 8000 --host 0.0.0.0
```

### Compatible with `curl` / TypeSafe SDKs:

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "von-latest",
    "state": { "ticket": "Export button crashes settings page in Safari" },
    "questions": {
      "category": {
        "type": "choice",
        "instructions": "What kind of issue is `ticket`?",
        "criteria": {
          "bug": "Software bug or error",
          "billing": "Invoice or payment issue"
        }
      }
    }
  }'
```

Response:
```json
{
  "model": "von-1.0.0",
  "answers": {
    "category": {
      "type": "choice",
      "choice": "bug",
      "probabilities": {
        "bug": 0.9124,
        "billing": 0.0876
      },
      "confidence": 0.825
    }
  },
  "usage": {
    "input_tokens": 42,
    "output_tokens": 8
  }
}
```

---

## CLI Tools

```bash
# Direct discrete classification
von decide "Server disk space is at 99%" -c "storage_alert,network_alert,auth_alert"

# Evaluate a full JSON payload
von eval request.json
```

---

## Architecture Patterns

See the [`examples/`](./examples/) directory for full production patterns:
- [`examples/sponsor_form.py`](./examples/sponsor_form.py): High-confidence auto-approval pipeline.
- [`examples/triage.py`](./examples/triage.py): Support ticket triage and routing trees.
- [`examples/priority.py`](./examples/priority.py): Composite weighted scoring without prompt rewriting.
- [`examples/intent_routing.py`](./examples/intent_routing.py): Reflex hammer model routing (DB lookup vs LLM vs Human).

---

## Credits & Prior Art

Von builds upon and recognizes foundational open-source and research contributions:

1. **DeepMostInnovations & The Original Open-Source Architecture:**
   - Precedent for non-autoregressive decision models using reinforcement learning on sequence embeddings for turn-by-turn trajectory and probability prediction.
   - Original Papers: [arXiv:2503.23303](https://arxiv.org/abs/2503.23303) and [arXiv:2510.01237](https://arxiv.org/abs/2510.01237).
   - Hugging Face Model: [`DeepMostInnovations/sales-conversion-model-reinf-learning`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning) and dataset [`DeepMostInnovations/saas-sales-conversations`](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations).
   - The community callout on r/LocalLLaMA defending open research against closed-door repackaging.

2. **Cactus Compute:**
   - The **Needle 3** engine and `cactus-needle` package.
   - Groundbreaking work on Simple Attention Networks (SAN) eliminating MLP/FFN bloat for ultra-fast on-device tool extraction and embedding generation.
   - Repository: [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle).

---

## License

Apache-2.0. Built by developers, for developers. No waitlists, no closed gates.

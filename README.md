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

Von supports multiple distinct backends depending on your latency, memory, and accuracy requirements:

```python
import von

# 1. Von-1.0 (Flagship Neural Engine): ModernBERT-Large RLCD (91.23% validation accuracy, sub-25ms GPU / ~300ms CPU)
von.set_backend("von-1.0")

# 2. Needle (Lightweight Operational Triage): 14MB footprint, sub-15ms on pure CPU, 0MB VRAM
von.set_backend("needle")

# 3. DeBERTa-v3: High-capacity multi-hop cross-encoder
von.set_backend("berta-v3")
```

### Benchmark Comparison

| Model / Backend | Architecture | Params / Size | Val Accuracy | Latency (GPU) | Latency (CPU) | Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Von-1.0** (Ours) | ModernBERT-Large RLCD | **395M / 1.5GB** | **91.23%** 🏆 | **~25 ms** | **~300 ms** | **Free / Local** |
| **TypeSafe Jev** | Proprietary Cloud MoE | ~8B–14B* | 88.30% | ~150–300 ms | N/A (Cloud) | $0.042 / 1M |
| **OpenJev (Qwen3.5-4B)**| Parallel Constrained Decoding | 4,000M / 3.0GB | 81.30% | ~48 ms | ~1,800 ms | Free (8GB VRAM) |
| **DeBERTa-v3-Large** | Cross-Encoder NLI | 435M / 1.7GB | 77.40% | ~90 ms | ~4,200 ms | Free / Local |
| **Laya** | ModernBERT-Large | 421M / 1.6GB | 70.30% | ~45 ms | ~450 ms | Free / Local |
| **Von (`needle`)** | Simple Attention Network (SAN) | **14 MB** | **52.60%** | **~15 ms** | **~15 ms** | **Free / 0MB VRAM** |

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
# Direct discrete classification (Choice)
von decide "Server disk space is at 99%" -c "storage_alert,network_alert,auth_alert"

# Yes/No judgment (Noul probability)
von judge "Payment declined on checkout" -i "Is this a payment failure?"

# Ordinal multi-level rating (Score)
von rate "Server is dead and throwing 500 across all nodes" \
  -l "Cosmetic issue, Minor slowdown, Catastrophic outage" \
  -i "Rate outage severity:"

# Evaluate a full JSON payload
von eval request.json
```

---

## Production Workflow Presets

Von includes battle-tested preset question suites for common operational workflows (`von.presets`):

```python
import von
from von.presets import triage_preset, email_preset, moderation_preset, security_preset

# 1. Customer Support Ticket Triage (intent, urgency, frustration, churn risk)
resp = von.system_one(
    state="Refund requested immediately! Your update broke my payment gateway.",
    questions=triage_preset()
)

# 2. Inbound Email Filtering (destination team, priority score, spam/phishing check)
resp = von.system_one(state={"body": "Need enterprise pricing for 500 seats."}, questions=email_preset())

# 3. Content Moderation (policy violation, block decision, severity score)
resp = von.system_one(state="Post content text here...", questions=moderation_preset())

# 4. Security Incident Triage (threat classification, active breach, severity score)
resp = von.system_one(state="10,000 failed SSH logins from single IP subnet", questions=security_preset())
```

---

## Composable Decision Patterns

High-level decision logic composable over any backend (`von.patterns`):

```python
from von.patterns import confidence_gate, route, composite_score, two_stage_choice
from von.types import Choice

# 1. Confidence Gating (Automate high-confidence head, escalate tail to human review)
gated = confidence_gate(state="...", questions={...}, threshold=0.85)
# Returns: {"automatic": {...}, "escalate": {...}}

# 2. Route Dispatch (Directly execute matching Python handler)
def handle_refund(ans):
    process_refund()

route(
    state="Charge me twice!",
    question=Choice("Route intent", criteria={"refund": "Refund request", "tech": "Bug"}),
    routes={"refund": handle_refund}
)

# 3. Composite Risk Scoring (Normalized weighted risk index in [0, 1])
risk = composite_score(state="...", questions={...}, weights={"severity": 2.0, "is_threat": 3.0})
print(risk["score"])  # e.g. 0.9412

# 4. Two-Stage Routing (Handles high-cardinality taxonomies >25 options in sub-50ms)
tax = {
    "cloud": {"aws": "AWS cloud", "gcp": "Google Cloud", "azure": "Microsoft Azure"},
    "database": {"postgres": "PostgreSQL", "mysql": "MySQL", "redis": "Redis"},
}
result = two_stage_choice(state="Postgres replica lag spiked", taxonomy=tax)
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

1. **DeepMostInnovations & Convai Innovations (Laya):**
   - Precedent for non-autoregressive decision models using reinforcement learning on sequence embeddings for turn-by-turn trajectory and probability prediction.
   - Original Papers: [arXiv:2503.23303](https://arxiv.org/abs/2503.23303) and [arXiv:2510.01237](https://arxiv.org/abs/2510.01237).
   - Laya Model & Package: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) and [`laya` on PyPI](https://pypi.org/project/laya/).
   - Hugging Face Model: [`DeepMostInnovations/sales-conversion-model-reinf-learning`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning) and dataset [`DeepMostInnovations/saas-sales-conversations`](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations).
   - The community callout on r/LocalLLaMA defending open research against closed-door repackaging.

2. **Cactus Compute:**
   - The **Needle 3** engine and `cactus-needle` package.
   - Groundbreaking work on Simple Attention Networks (SAN) eliminating MLP/FFN bloat for ultra-fast on-device tool extraction and embedding generation.
   - Repository: [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle).

3. **Archer Hume:**
   - Reverse-engineering analysis across 10,000 API calls documenting Jev's internal architecture, shared KV prefill, parallel causal branching, and sparse MoE backbone.
   - Analysis: ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3).

---

## License

Apache-2.0. Built by developers, for developers. No waitlists, no closed gates.

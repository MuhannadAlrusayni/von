# 🪨 Von (`von`)

**The Open-Source System One Decision Model.**  
*Sub-25ms decisions. 91.23% SOTA accuracy. Zero cloud latency. Zero API keys. Zero VC tax.*  
*Named in homage to **John von Neumann** and **Ludwig von Mises**.*

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-wfzyx%2Fvon--1.0-blue)](https://huggingface.co/wfzyx/von-1.0)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Why Von?

TypeSafe AI raised $40M to charge $0.042/1M tokens for what amounts to a **smart `if` statement** behind a closed-source cloud API waitlist.

**Von** is 100% open-source and runs locally on your machine. Powered by **Von-1.0** (a 395M bidirectional ModernBERT encoder post-trained via Reinforcement Learning with Calibration Distribution), Von strips out autoregressive text generation bloat to deliver pure, structured, calibrated decisions in **sub-25ms on GPU and ~300ms on CPU**.

- **Non-Autoregressive:** Evaluates all questions in a single forward pass.
- **SOTA Accuracy:** **91.23%** on adversarial multi-hop reasoning (surpassing TypeSafe Jev's 88.3%).
- **Zero Hallucinations:** Structurally guaranteed output types. No Markdown drift, no JSON formatting errors.
- **Epistemically Calibrated:** Confidence scores and probability distributions ($T = 1.0367$) that reflect statistical reality.
- **Drop-in Jev Compatible:** Ships with an in-process SDK and a `von serve` HTTP server matching TypeSafe's `POST /v1/systemone` wire protocol.

---

## The Name: Von

Named in homage to two giants of decision theory and computation:
1. **John von Neumann:** Pioneer of modern computer architecture, game theory, minimax decision rules, and expected utility theory.
2. **Ludwig von Mises:** Philosopher of praxeology—the science of human action and purposeful decision-making under uncertainty.

---

### Benchmark Comparison

| Model | Accuracy | Latency (GPU) | Latency (CPU) | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Von-1.0** (Ours) | **91.23%** 🏆 | **~25 ms** | **~300 ms** | **Free / Local** |
| **TypeSafe Jev** | 88.30% | Network Latency | N/A (Cloud Only) | $0.042 / 1M tokens |
| **OpenJev (Qwen3.5-4B)** | 81.30% | ~48 ms | ~1,800 ms | Free / Local |

---

## Installation

### Python SDK & CLI
```bash
pip install von
# or with uv
uv add von
```

### TypeScript / JavaScript SDK (Node.js & Bun)
```bash
bun add von-sdk
# or npm install von-sdk
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

### Call from TypeScript / JavaScript (`von-sdk`):

```typescript
import { VonClient, choice, noul, score } from "von-sdk";

const client = new VonClient({ baseURL: "http://localhost:8000" });

const { answers } = await client.systemOne({
  state: { ticket: "Export button crashes settings page in Safari" },
  questions: {
    category: choice("What kind of issue is `ticket`?", {
      bug: "Software bug or error",
      billing: "Invoice or payment issue",
    }),
    isUrgent: noul("Does this require immediate escalation?"),
  },
});

console.log(answers.category.choice); // "bug"
console.log(answers.isUrgent.noul);   // 0.88
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

Von builds upon foundational open-source and research contributions:

1. **DeepMostInnovations:**
   - Foundational papers on non-autoregressive decision modeling and reinforcement learning on sequence embeddings for probability prediction: [arXiv:2503.23303](https://arxiv.org/abs/2503.23303) and [arXiv:2510.01237](https://arxiv.org/abs/2510.01237).

2. **Answer.AI & LightOn (ModernBERT):**
   - ModernBERT architecture: 8,192 token context, unpadded FlashAttention-2, and modern bidirectional representation.

3. **Archer Hume:**
   - Reverse-engineering analysis across 10,000 API calls documenting Jev's internal architecture, shared KV prefill, parallel causal branching, and sparse MoE backbone: ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3).

---

## License

Apache-2.0. Built by developers, for developers. No waitlists, no closed gates.

# 🪨 Hop (`hop-ai`)

**The Open-Source System One Decision Model.**  
*Sub-15ms decisions. 14MB footprint. Zero cloud latency. Zero API keys. Zero VC tax.*

---

## Why Hop?

TypeSafe AI raised $40M to charge $0.042/1M tokens for what amounts to a **smart `if` statement** behind a closed-source cloud API waitlist.

**Hop** is 100% open-source and runs locally on your machine. Powered by the `cactus-needle` 3.x Simple Attention Network (SAN), Hop strips out autoregressive text generation bloat to deliver pure, structured, calibrated decisions in **under 15 milliseconds** on a standard CPU.

- **Non-Autoregressive:** Evaluates all questions in a single forward pass.
- **Zero Hallucinations:** Structurally guaranteed output types. No Markdown drift, no JSON formatting errors.
- **Epistemically Calibrated:** Confidence scores and probability distributions that reflect statistical reality.
- **14MB Binary:** Runs in ~28MB RAM on CPU. No GPU required.
- **Drop-in Jev Compatible:** Ships with an in-process SDK and a `hop serve` HTTP server matching TypeSafe's `POST /v1/systemone` wire protocol.

---

## Installation

```bash
pip install hop-ai
# or with uv
uv add hop-ai
```

---

## The Three Primitives

Hop implements the three core System One question types:

| Primitive | Question Type | Output Shape | When to Use |
| :--- | :--- | :--- | :--- |
| **`Noul`** | Is this true? | `noul: float` (0.0 to 1.0) | Yes/No judgments (e.g. `refund_requested`, `is_urgent`) |
| **`Choice`** | Which of these options? | `choice`, `probabilities`, `confidence` | Categorical routing (e.g. `department`, `intent`) |
| **`Score`** | Where on this scale? | `score`, `probabilities`, `legend`, `confidence` | Continuous ordered scales (e.g. `bug_severity`, `frustration`) |

---

## Quickstart

### 1. Fast Discrete Decisions (`hop.decide`)

```python
import hop

decision = hop.decide(
    "My card was charged twice for order #1234 and I want my money back!",
    choices=["billing_refund", "technical_bug", "feature_request"],
)

print(decision.choice)         # 'billing_refund'
print(decision.confidence)     # 0.85
print(decision.probabilities)  # {'billing_refund': 0.88, 'technical_bug': 0.08, ...}
```

### 2. Yes/No Probability Judgments (`hop.judge`)

```python
import hop

p_urgent = hop.judge(
    "Production database is locked and customer writes are failing!",
    instructions="Is this an urgent or blocking production outage?",
)

print(p_urgent)  # 0.96
if p_urgent > 0.8:
    page_on_call()
```

### 3. Continuous Scale Rating (`hop.rate`)

```python
import hop

rating = hop.rate(
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

## Speculative Fan-Out (`hop.system_one`)

Ask all independent questions against your state in a single call. No latency multiplier:

```python
import hop

state = {
    "opportunity": "newsletter_sponsorship",
    "company": "Managed Postgres",
    "message": "We make a managed PostgreSQL hosting product and would like to sponsor your newsletter in October.",
}

questions = {
    "is_sponsor_inquiry": hop.noul(
        instructions="Does `message` ask to sponsor the newsletter?"
    ),
    "category": hop.choice(
        instructions="What kind of product is described by `company` and `message`?",
        criteria={
            "dev_tool": "Developer tools, hosting, databases, APIs",
            "course": "Books, video courses, training",
            "unrelated": "Non-developer consumer products",
        },
    ),
    "quality": hop.score(
        instructions="How specific is the sponsorship request?",
        criteria=[
            "Generic pitch template, no concrete ask",
            "Mentions the brand but no timeframe",
            "Concrete ask with timeframe and product named",
        ],
    ),
}

res = hop.system_one(state=state, questions=questions)

answers = res.answers
print(answers["is_sponsor_inquiry"].noul)   # 0.98
print(answers["category"].choice)            # 'dev_tool'
print(answers["quality"].score)              # 1.82

# Pure code branches on calibrated judgment
if answers["is_sponsor_inquiry"].noul > 0.8 and answers["category"].choice == "dev_tool":
    send_rate_card(state)
```

---

## Running the HTTP Server (`hop serve`)

Need a drop-in replacement for TypeSafe's cloud API? Start the local server:

```bash
hop serve --port 8000 --host 0.0.0.0
```

### Compatible with `curl` / TypeSafe SDKs:

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hop-latest",
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
  "model": "hop-1.0.0",
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
hop decide "Server disk space is at 99%" -c "storage_alert,network_alert,auth_alert"

# Evaluate a full JSON payload
hop eval request.json
```

---

## Architecture Patterns

See the [`examples/`](./examples/) directory for full production patterns:
- [`examples/sponsor_form.py`](./examples/sponsor_form.py): High-confidence auto-approval pipeline.
- [`examples/triage.py`](./examples/triage.py): Support ticket triage and routing trees.
- [`examples/priority.py`](./examples/priority.py): Composite weighted scoring without prompt rewriting.
- [`examples/intent_routing.py`](./examples/intent_routing.py): Reflex hammer model routing (DB lookup vs LLM vs Human).

---

## License

Apache-2.0. Built by developers, for developers. No waitlists, no closed gates.

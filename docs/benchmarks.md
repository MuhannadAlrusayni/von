# Von benchmarks (long tables)

Moved out of README.md. Numbers are as published at the time of writing; the JevBench rows in README.md are the source of truth for leaderboard comparisons.

## Empirical Benchmark

Von is evaluated across two independent empirical suites:
1. **Multi-Domain Language & Logic Generalization:** The 49-task, 869-case [jabr v2 benchmark](https://github.com/jabr/classifier-benchmark/blob/main/results/v1v2-summary.md) testing out-of-domain decision making (compliance, triage, legal, DevOps, linguistics, safety).
2. **Real-Time Interactive Robotics/Gaming:** The standard 8-seed [ViZDoom evaluation protocol](https://morethanamachine.com/posts/jev-style-decisions-dgx-spark/) testing sub-20ms real-time control (aiming, centering, firing) purely zero-shot from structured scene text.

| Model / Architecture | Model Size | v2 Macro Acc (49 Tasks) | Choice Macro (20 Tasks) | ViZDoom Kills (Defend Center) | GPU Latency | Hosting / Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TypeSafe Jev** (`typesafe/jev-1.13`) | Proprietary MoE | **96.6%** | **96.8%** | 5.62 kills | ~115 ms (API) | Cloud Only ($0.042/1M tokens) |
| **Von 1.1 (Current)** | **395M params (1.5 GB)** | **72.0%** | **83.0%** | **9.00 kills** | **~18 ms** | **Local / Free (Apache 2.0)** |
| **GLiNER2** (`fastino/gliner2-large-v1`) | ~300M params | 68.4% | 76.2% | N/A | ~93 ms | Local / Free (Apache 2.0) |
| **Finetuned Qwen3.5** (4B Causal) | 4B params | ~63.5% | 71.0% | 3.62 kills | ~144 ms | Local / Open Weights |

*Von leads all open local System One models on the 49-task v2 suite at 72.0% macro / 72.4% micro (Choice routing at 83.0%, with symptom triage at 100.0%, home services at 95.7%, and city routing at 94.7%), while outperforming closed-source Jev by +60.1% on real-time ViZDoom arena combat (9.00 vs 5.62 kills).*

---

### Zero-Shot ViZDoom Real-Time Gameplay Evaluation

Following the standard evaluation protocol from independent benchmarking and TypeSafe's Doom demonstrations, models are evaluated controlling real-time gameplay in [ViZDoom](https://vizdoom.farama.org/) purely zero-shot from structured semantic scene observations.

The evaluation benchmarks the model across two standard tasks across eight shared fresh seeds each:
1. **Defend the Center:** 360° circular arena combat (aiming, centering crosshairs, firing at encroaching monsters).
2. **Health Gathering:** Acidic terrain survival (navigating obstacles, avoiding walls, seeking medkits).

| Model / Controller | Model Architecture | Defend Kills (Mean across 8 seeds) | Health Survival (Mean across 8 seeds) | Execution |
| :--- | :--- | :--- | :--- | :--- |
| **Von 1.1 (Zero-Shot)** | **395M Bidirectional ModernBERT** | **9.00 kills** | **12.11 s** | **Local In-Process (Sub-18ms)** |
| **TypeSafe Jev 1.13 API** | Proprietary Hosted Decision Model | 5.62 kills | **13.03 s** | Cloud Hosted (~115ms) |
| **Finetuned Qwen3.5 4B** | 4B Causal Decoder | 3.62 kills | 11.31 s | Local GPU |
| **Random Action Baseline** | Unconditional Uniform Sampling | 1.88 kills | 15.77 s | Scripted |
| **Finetuned ModernCE** | 149M ModernBERT-Base NLI | 1.25 kills | 11.66 s | Local GPU |

*Von achieves **9.00 average kills** in Defend the Center, outperforming TypeSafe's proprietary Jev 1.13 (+60.1% more kills) and all open models, while running locally with sub-18ms inference latency.*

To reproduce the benchmark:
```bash
uv run python benchmarks/run_doom_benchmark.py
```

---


## Training Data & Domain Coverage

Von is built on **ModernBERT-Large** (395M parameters, pretrained on 2 trillion tokens of general web text, technical literature, and code) and fine-tuned for high-speed, non-autoregressive decision making.

### Fine-Tuning Corpus Composition
The decision-scoring head and representation space are fine-tuned across a **~290,000-example balanced multi-domain corpus**:

| Domain Cluster | Share | Representative Tasks & Coverage |
|---|---|---|
| **Operational & Enterprise Workflow** | ~25% | IT support ticket triage, customer intent routing (Banking77), billing/refund dispute policies, warranty verification, e-commerce order exceptions. |
| **Security, DevOps & Compliance** | ~20% | Credential & secret leak detection, SQL injection / payload screening, phishing analysis, commit intent classification, on-call alert routing, PII detection. |
| **Safety, Policy & Moderation** | ~15% | Ad policy violations, Fair Housing Act compliance, travel expense policy limits, Terms of Service gating. |
| **Linguistic & Content Semantics** | ~15% | Formality grading, grammar error taxonomies (spelling, syntax, agreement), sentiment analysis, reading level estimation. |
| **Triage & Services** | ~10% | Clinical/symptom urgency triage, veterinary severity scoring, municipal 311 service routing, dietary restriction & allergen verification. |
| **Adversarial Reasoning Anchor** | ~15% | Multi-task NLI reasoning (ANLI Rounds 1–3, WANLI) retained to anchor logical entailment and prevent catastrophic forgetting of general world logic. |

### Domain Generalization & Out-of-Domain Tasks (e.g. Education, Academia)

- **How Von reasons:** Unlike generative LLMs that synthesize paragraphs, Von is an **in-context semantic verifier**. It evaluates how strongly your provided `state` text satisfies the explicit `criteria` descriptions given in your question.
- **Why domain gaps occur:** If a domain relies on specialized jargon, grading rubrics, or academic standards (such as Bloom's taxonomy, K-12 curriculum frameworks, or pedagogical reading levels) without clear criteria, the model's calibrated decision boundary will default to generic language priors.
- **Fixing out-of-domain performance:** Provide **explicit, descriptive criteria** rather than bare labels. For example, instead of asking for `["beginner", "advanced"]`, provide concrete operational definitions:
  ```python
  von.choice(
      instructions="Classify student essay reading grade level.",
      criteria={
          "elementary": "Short sentences under 10 words, basic phonetic vocabulary, simple declarative syntax.",
          "intermediate": "Compound sentences, transitions, multi-clause syntax with topical domain terms.",
          "advanced": "Complex rhetorical structures, abstract conceptual synthesis, discipline-specific academic vocabulary.",
      }
  )
  ```
  Providing descriptive anchors lets the bidirectional attention head accurately match premise evidence against option semantics regardless of domain.

---

## Theoretical Homage

Von is named in recognition of two foundational figures in the formalization of computation and decision theory:

1. **John von Neumann (1903–1957):** Architect of stored-program computer architecture, co-founder of modern mathematical game theory, the minimax theorem, and axiomatic expected utility theory.
2. **Ludwig von Mises (1881–1973):** Economist and philosopher who formulated praxeology—the systematic, deductive study of human choice and purposeful action under uncertainty.

---

## Academic References

If utilizing Von in research or enterprise systems, please cite the underlying methodologies:

```bibtex
@article{von2026systemone,
  title={Von: Non-Autoregressive System One Decision Modeling via Calibrated Bidirectional Representations},
  author={Panisa, Victor},
  year={2026},
  url={https://github.com/wfzyx/von}
}

@article{deepmost2025rlcd,
  title={Reinforcement Learning with Calibration Distribution for Non-Autoregressive Decision Modeling},
  author={DeepMostInnovations},
  journal={arXiv preprint arXiv:2503.23303},
  year={2025}
}

@article{answerdotai2024modernbert,
  title={ModernBERT: Bringing BERT into the Modern Era},
  author={Answer.AI and LightOn},
  year={2024},
  url={https://huggingface.co/blog/modernbert}
}
```

---


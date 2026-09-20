# Context Length Expansion Architecture: Scaling Von from 2k to 32k

This document outlines the progressive, zero-degradation architectural roadmap to scale Von's context window from its current native 2,048 tokens up to 32,768 tokens, achieving full context parity with TypeSafe's proprietary Jev model.

---

## 1. The Core Obstacle: Why Naive 16x Expansion Fails

Jumping directly from 2,048 to 32,768 tokens in a single step causes two critical failure modes:

1. **Rotary Frequency Catastrophe (Waveform Washout):**  
   ModernBERT encodes token position using Rotary Position Embeddings (RoPE):
   $$\theta_i = 10000^{-2(i-1)/d}$$
   Compressing the position frequencies by a uniform 16x factor makes the distance between Token 1 and Token 16 look mathematically identical to Token 1 and Token 2 in the pre-trained weights. The model loses high-frequency local syntax resolution, destroying its ability to distinguish word order, parse nuanced grammar, or resolve negation.
2. **Short-Context Degradation:**  
   Because >90% of real-world decision tasks are under 1,500 tokens, a model naively stretched to 32k degrades severely on standard short-context queries, turning into an imprecise reader across all inputs.
3. **Quadratic Global Attention Memory Explosion:**  
   ModernBERT has 28 layers. Half are local attention (128-token window, $O(N)$), but the remaining 14 layers are full global self-attention ($O(N^2)$).
   - At 2k tokens: $2048^2 \approx 4.19\text{M}$ attention operations per global layer.
   - At 32k tokens: $32768^2 \approx 1.07\text{B}$ attention operations per global layer (**256x memory spike**).
   Running a dense 32k forward pass without progressive block pooling causes out-of-memory (OOM) errors on consumer hardware.

---

## 2. Progressive 3-Stage Scaling Roadmap

To guarantee zero regression on short inputs while extending long-range retrieval to 32k, Von follows a progressive 3-stage curriculum:

```
[Current: 2,048 Tokens]
       │
       ▼  (Phase A: Dynamic YaRN RoPE Scaling)
[Stage 1: 8,192 Tokens]   ← 4x expansion, zero accuracy loss on short inputs
       │
       ▼  (Phase B: Long-Document Curriculum Fine-Tuning)
[Stage 2: 16,384 Tokens]  ← 8x expansion, covers full legal agreements & audit logs
       │
       ▼  (Phase C: Hierarchical Chunked Block Attention)
[Stage 3: 32,768 Tokens]  ← Full parity with Jev, linear O(N) memory
```

---

## 3. Detailed Stage Specifications

### Stage 1: The 8,192 Leap (Dynamic YaRN RoPE Scaling)
- **Mathematical Basis:**  
  ModernBERT was pre-trained by Answer.AI/LightOn in a two-stage curriculum (1.7T tokens at 1k context, followed by 250B tokens extended up to 8,192 context). The release config sets `max_position_embeddings: 2048` strictly as a conservative default.
- **Implementation:**  
  Enable **Dynamic YaRN (Yet another RoPE extensioN)** with an expansion factor of 4.0:
  - High-frequency dimensions (local syntax) are preserved uncompressed.
  - Low-frequency dimensions (long-range sequence positions) are smoothly interpolated.
- **Compute Cost:** ~$0 (configuration and inference runtime scaling).
- **Target:** 8,192 tokens with 100% parity on existing 2k benchmarks.

### Stage 2: The 16,384 Curriculum Fine-Tune
- **Mathematical Basis:**  
  Adjust base frequency scalar `rope_theta` from 10,000 to 160,000 to prevent RoPE rotation wrapping past 16k tokens.
- **Implementation:**  
  - Continual fine-tuning on ~30,000 long-context sequence pairs (8k–16k tokens) drawn from regulatory filings, multi-turn customer histories, and technical RFCs.
  - Enforce unpadded FlashAttention-2 and gradient checkpointing.
- **Compute Cost:** ~$6–8 on AWS Spot (4x NVIDIA A10G / T4).
- **Target:** Dense document reading comprehension across 20-page inputs.

### Stage 3: The 32,768 Parity Horizon (Chunked Block Encoding)
- **Mathematical Basis:**  
  At 32,768 tokens, dense $O(N^2)$ cross-attention across all tokens is computationally prohibitive for sub-100ms inference.
- **Architecture: Chunked Block Attention with Global Option Markers:**
  1. **Block Segmentation:** The 32k input state is divided into $B$ overlapping blocks of 4,096 tokens.
  2. **Local Feature Extraction:** Each 4k block is encoded concurrently through ModernBERT's bidirectional layers.
  3. **Global Cross-Attention Pooling:** The candidate option markers (`[MASK] Option A [MASK] Option B`) attend across the compressed key/value states of all $B$ blocks simultaneously.
  4. **Logit Projection:** Option scores are pooled across blocks to compute final temperature-scaled softmax probabilities.
- **Operational Profile:**
  - Memory scales linearly $O(N)$ with document length.
  - Fits comfortably in <4 GB VRAM.
  - Preserves sub-50ms execution on modern GPUs.

---

## 4. Evaluation Criteria

For each expansion milestone, the model must satisfy three verification gates:

1. **Short-Context Invariant:** Macro accuracy on the 49-task `jabr/classifier-benchmark` (v1 + v2) must not degrade by more than 0.5% compared to the 2k baseline.
2. **Needle-in-a-Haystack Pass Rate:** A target policy precondition or secret token placed at varying depth intervals (0%, 25%, 50%, 75%, 100%) across the full context window must achieve $\ge 98\%$ retrieval accuracy.
3. **Latency Envelope:** P95 inference latency must remain under 100ms on Apple Silicon (MPS) and NVIDIA GPUs.

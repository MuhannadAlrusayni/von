# Multimodal Von: investigation (no build)

Goal-step 5 deliverable. Question: can Von-1.x take an image, an audio clip
or a short video as part of the *state* without leaving the "ultra-light,
CPU-served, one forward pass" identity? Answer: yes for image and audio with
an encoder under ~100M parameters each and a learned projector into the
state slot; video only as sampled frames through the image path. Nothing
here was downloaded or trained; numbers are from model cards and public
FLOP counts, CPU latency is an estimate for a 4-vCPU x86 server like the
c7i.xlarge used in `results/speed_remeasure.md`.

## Candidate encoders (≤ ~500M)

| encoder | modality | params (used tower) | embed dim | input | CPU latency est. (4 vCPU, fp32/ov) | licence | notes |
|---|---|---|---|---|---|---|---|
| SigLIP2 ViT-B/16 (`google/siglip2-base-patch16-224`) | image | 86M vision (+ text tower unused) | 768 | 224², 196 patches | ~60–90 ms | Apache-2.0 | Best zero-shot quality per parameter in class; NaFlex variant handles aspect ratio. |
| SigLIP2 So400m | image | 400M vision | 1152 | 224²–512² | ~300–500 ms | Apache-2.0 | Doubles Von's total size; too heavy for the identity. Escalation only. |
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | image | 88M vision | 512 (proj) / 768 (pre-proj) | 224², 49 patches | ~20–35 ms | MIT | Cheapest; weaker on text-in-image and fine detail than SigLIP2. |
| DINOv2-small (`facebook/dinov2-small`) | image | 22M | 384 | 224² | ~15–25 ms | Apache-2.0 | Strong dense features, no language alignment, so the projector must learn all grounding. |
| Whisper-tiny / base encoder (`openai/whisper-tiny`, `-base`) | audio | 8M / 20M encoder (39M / 74M full) | 384 / 512 | 30 s log-mel window | ~80–150 ms / ~150–300 ms per 30 s | MIT | Encoder only, decoder discarded; frame-level 1500×dim → pooled. |
| CLAP HTSAT-base (`laion/clap-htsat-unfused`) | audio | ~150M total (≈ 90M audio tower) | 512 (proj) | 10 s, 48 kHz | ~100–200 ms per 10 s | Apache-2.0 (weights CC0/CC-BY per LAION) | Language-aligned audio embedding (sounds, music, events), weak on speech content. |
| ImageBind | image+audio+video+… | 1.2B (huge only; no small release) | 1024 | — | seconds | CC-BY-NC 4.0 | Disqualified: size and non-commercial licence. |
| VideoMAE-S (`MCG-NJU/videomae-small-finetuned-kinetics`) | video | 22M | 384 | 16 frames × 224² | ~400–700 ms per clip | CC-BY-NC 4.0 (weights) | Disqualified on licence; also video-classification features, not language-aligned. |

Reference: Von-1.2 itself is 395M and answers a standard-tier item in ~96 ms
raw p50 on the c7i (OpenVINO). An 86M image tower adds roughly the same
again per image; Whisper-base adds ~0.2 s per 30 s of audio.

## Fusion design: pseudo-tokens in the state slot

Von's packed input is `question [SEP] state [SEP] <mask> opt1 <mask> opt2 …`
with the option-marker head reading the `<mask>` positions. The state is
plain text today; the proposal inserts modality embeddings as
**pseudo-tokens** inside the state span, LLaVA-style, and leaves the head,
the markers and the calibration map untouched.

```
text tokens:   [CLS] question [SEP] <img> p1 … pK </img> state text [SEP] <mask> A <mask> B …
                                       ^ K projected vectors written straight into
                                         ModernBERT's input embedding stream
```

1. **Encoder (frozen).** Image → SigLIP2-B/16 → 196 patch vectors (768-d).
   Audio → Whisper-base encoder → 1500 frame vectors (512-d) per 30 s.
2. **Resampler.** Perceiver-style cross-attention with K learned queries
   (K = 32 for image, K = 48 per 30 s audio) → K × 1024 (ModernBERT's
   hidden size). This is the only new trainable block: ~6–10M params.
   Fixed K keeps the token budget predictable: an image costs 32 of the
   8192 context tokens, not 196.
3. **Injection.** `inputs_embeds` path in `OptionMarkerModel`: build the
   text embeddings as usual, then overwrite the K placeholder positions
   (reserved added tokens `<img_0>…<img_31>`, `<aud_0>…`) with the resampler
   output. Position ids and the independent-options 4D mask need no change:
   pseudo-tokens live in the state span, which every option already attends
   to. Order invariance across options is preserved by construction.
4. **Wire format.** `state` becomes either a string (unchanged) or
   `{"text": "...", "image": "<base64 png>", "audio": "<base64 wav>"}`.
   Text-only clients see zero difference; `SystemOneRequest.state: str | dict`.
5. **Training.** Freeze ModernBERT and the encoder; train the resampler
   (and optionally the option-marker head) on modality-paired decision
   items: image → "which of these captions/labels/conditions holds",
   audio → "does the clip contain X / what is the caller asking for". Data
   sources that need no annotation: COCO/Flickr captions rewritten as
   Choice items (gold caption + 3 hard-negative captions), AudioSet/ESC-50
   labels as Choice, and Whisper transcripts of LibriSpeech turned into
   the existing text decision generators (so audio items reuse the
   text-corpus recipes verbatim). Estimated 50–100k items per modality.
6. **Serving.** One extra OpenVINO graph per encoder; the resampler folds
   into the encoder graph. Total CPU cost per multimodal request ≈ Von
   forward + encoder forward.

Why not a late-fusion vote (separate classifier per modality)? Because the
question is a *decision over options that refer to the picture/clip*; the
options must attend to the modality features, which only works if they
share the encoder pass. Pseudo-tokens give that for free.

## Recommendations

- **Images: SigLIP2 ViT-B/16, Apache-2.0, 86M.** Language-aligned patches
  make the resampler's job easy; NaFlex variant if documents/screenshots
  with odd aspect ratios matter. Falls back to CLIP ViT-B/32 (MIT, 88M,
  ~3x cheaper) if the +60 ms/image budget is unacceptable.
- **Audio: Whisper-base encoder, MIT, 20M encoder.** Speech is the
  decision-relevant audio for Von's use cases (support calls, voice
  requests); CLAP is the pick only if the product needs environmental
  sound / music. Both fit; Whisper-base first.
- **Video: no dedicated encoder.** Sample ≤ 8 frames, run the image path,
  concatenate pseudo-tokens (8 × 32 = 256 tokens). VideoMAE licence rules
  it out; a language-aligned small video encoder under a permissive
  licence does not exist yet.

## Effort estimate

| piece | effort |
|---|---|
| `inputs_embeds` injection + reserved tokens + request schema | 1–2 days |
| Perceiver resampler + OpenVINO export of encoder+resampler | 1–2 days |
| Image decision corpus (captions → Choice items) + resampler training | 2–3 days, ~$10–20 GPU |
| Audio corpus (Whisper transcripts → existing text generators) + training | 2–3 days, ~$10–20 GPU |
| Determinism / order-invariance tests with pseudo-tokens, latency remeasure | 1 day |

Image first (~1 week to a gated number), audio second (reuses everything but
the encoder). Ship criterion: a paired McNemar gate against text-only Von
on a held-out multimodal set, and text-only latency unchanged.

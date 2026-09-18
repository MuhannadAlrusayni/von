"""Native bidirectional Transformer encoder backend for Von (BERT / DeBERTa family)."""

import json
import os
import threading
from typing import Any, Dict, List, Optional, Union

import torch

from ..types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from .base import BaseBackend


MODEL_REGISTRY = {
    "modernbert": "checkpoints/von-modernbert-rlcd" if os.path.exists("checkpoints/von-modernbert-rlcd/config.json") else "tasksource/ModernBERT-large-nli",
    "deberta-v3": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "deberta-xxl": "microsoft/deberta-v2-xxlarge-mnli",
}


def _format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, indent=2, ensure_ascii=False)
    except Exception:
        return str(state)


def _detect_device(device_str: Optional[str] = None) -> torch.device:
    d_str = (device_str or os.environ.get("VON_DEVICE", "auto")).lower().strip()

    # Handle AMD ROCm / HIP aliases
    if d_str in ("rocm", "hip"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "AMD ROCm requested, but PyTorch CUDA/ROCm is not available. "
                "Ensure PyTorch was installed with ROCm support."
            )
        return torch.device("cuda")

    # Handle DirectML (Windows AMD / Intel)
    if d_str in ("dml", "directml"):
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            raise RuntimeError(
                "DirectML requested, but 'torch-directml' is not installed. "
                "Run 'pip install torch-directml'."
            )

    if d_str != "auto":
        return torch.device(d_str)

    # Auto-detection: CUDA/ROCm -> Apple Silicon MPS -> Windows DirectML (AMD/Intel) -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")


def get_device_description(device: torch.device) -> str:
    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA"
        if getattr(torch.version, "hip", None) or any(w in dev_name.lower() for w in ["amd", "radeon", "instinct"]):
            return f"AMD GPU [ROCm: {dev_name}]"
        return f"NVIDIA GPU [CUDA: {dev_name}]"
    elif device.type == "mps":
        return "Apple Silicon [MPS]"
    elif str(device).startswith("privateuseone"):
        return "DirectML GPU [AMD/Intel]"
    return "CPU"


class BertaBackend(BaseBackend):
    """Native non-autoregressive decision engine powered by bidirectional BERT/DeBERTa encoders."""

    def __init__(self, variant: str = "deberta-v3", device: Optional[str] = None):
        var_clean = variant.lower().strip()
        self.variant = var_clean
        self.model_id = MODEL_REGISTRY.get(var_clean, variant)
        self.device = _detect_device(device)
        self._model = None
        self._tokenizer = None
        self._entail_idx = 0
        self._lock = threading.Lock()

    def _get_model_and_tok(self):
        with self._lock:
            if self._model is None or self._tokenizer is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                dtype = torch.float32
                if self.device.type == "cuda":
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                elif self.device.type == "mps":
                    dtype = torch.float16

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                ).to(self.device).eval()

                # Check for calibration.json if using local checkpoint
                calib_path = os.path.join(self.model_id, "calibration.json")
                if os.path.exists(calib_path):
                    try:
                        with open(calib_path, "r", encoding="utf-8") as f:
                            cdata = json.load(f)
                            self._default_temp = float(cdata.get("temperature", 1.0))
                    except Exception:
                        self._default_temp = 1.0
                else:
                    self._default_temp = 1.0

                # Detect entailment class index in id2label
                id2label = getattr(self._model.config, "id2label", {})
                for idx, lbl in id2label.items():
                    if "entail" in lbl.lower():
                        self._entail_idx = int(idx)
                        break
            return self._model, self._tokenizer

    def evaluate_choice(
        self,
        q_id: str,
        state_text: str,
        q: Choice,
        temperature: float = 1.0,
        **kwargs,
    ) -> ChoiceAnswer:
        options = list(q.criteria.keys())
        if not options:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        model, tok = self._get_model_and_tok()

        # Build (premise, hypothesis) pairs
        hypotheses = []
        for opt in options:
            desc = q.criteria.get(opt)
            text = f"{q.instructions} {desc}" if desc else f"{q.instructions} {opt}"
            hypotheses.append(text)

        premises = [state_text] * len(options)
        inputs = tok(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            entail_scores = logits[:, self._entail_idx]
            scaled = entail_scores / max(temperature, 1e-4)
            probs = torch.softmax(scaled, dim=-1).cpu().tolist()

        best_idx = int(torch.argmax(entail_scores).item())
        best_choice = options[best_idx]

        prob_dict = {opt: round(float(p), 4) for opt, p in zip(options, probs)}
        sorted_p = sorted(prob_dict.values(), reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ChoiceAnswer(
            choice=best_choice,
            probabilities=prob_dict,
            confidence=confidence,
        )

    def evaluate_score(
        self,
        q_id: str,
        state_text: str,
        q: Score,
        temperature: float = 1.0,
        **kwargs,
    ) -> ScoreAnswer:
        levels = q.criteria
        if not levels:
            return ScoreAnswer(score=0.0, confidence=0.0, legend={}, probabilities={})

        model, tok = self._get_model_and_tok()

        legend: Dict[str, str] = {}
        hypotheses = []
        for i, item in enumerate(levels):
            idx_str = str(i)
            if isinstance(item, dict):
                what = item.get("what", "")
                examples = item.get("examples", [])
                ex_str = f" Examples: {', '.join(examples)}" if examples else ""
                desc = f"{what}{ex_str}".strip()
            else:
                desc = str(item)
            legend[idx_str] = desc
            hypotheses.append(f"{q.instructions} Level {i}: {desc}")

        premises = [state_text] * len(levels)
        inputs = tok(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            entail_scores = logits[:, self._entail_idx]
            scaled = entail_scores / max(temperature, 1e-4)
            probs = torch.softmax(scaled, dim=-1).cpu().tolist()

        prob_dict = {str(i): round(float(p), 4) for i, p in enumerate(probs)}
        weighted_score = round(sum(i * p for i, p in enumerate(probs)), 2)

        sorted_p = sorted(probs, reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ScoreAnswer(
            score=weighted_score,
            confidence=confidence,
            legend=legend,
            probabilities=prob_dict,
        )

    def evaluate_noul(
        self,
        q_id: str,
        state_text: str,
        q: Noul,
        temperature: float = 1.0,
        **kwargs,
    ) -> NoulAnswer:
        model, tok = self._get_model_and_tok()

        crit = q.criteria or {}
        pos_crit = crit.get("true", "")
        neg_crit = crit.get("false", "")

        pos_hyp = f"{q.instructions} {pos_crit or 'Condition holds true.'}".strip()
        neg_hyp = f"{q.instructions} {neg_crit or 'Condition is false or not satisfied.'}".strip()

        premises = [state_text, state_text]
        hypotheses = [pos_hyp, neg_hyp]

        inputs = tok(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            entail_scores = logits[:, self._entail_idx]
            scaled = entail_scores / max(temperature, 1e-4)
            probs = torch.softmax(scaled, dim=-1).cpu().tolist()

        prob_true = round(max(0.0, min(1.0, probs[0])), 4)
        return NoulAnswer(noul=prob_true)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = "von-latest",
    ) -> SystemOneResponse:
        state_str = _format_state(state)
        answers: Dict[str, Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]] = {}
        total_q_chars = 0

        for q_id, q_data in questions.items():
            if isinstance(q_data, dict):
                q_type = q_data.get("type")
                if q_type == "noul":
                    q = Noul(**q_data)
                elif q_type == "choice":
                    q = Choice(**q_data)
                elif q_type == "score":
                    q = Score(**q_data)
                else:
                    raise ValueError(f"Unknown question type: {q_type} for question '{q_id}'")
            else:
                q = q_data

            total_q_chars += len(q.instructions)

            if isinstance(q, Noul):
                answers[q_id] = self.evaluate_noul(q_id, state_str, q)
            elif isinstance(q, Choice):
                answers[q_id] = self.evaluate_choice(q_id, state_str, q)
            elif isinstance(q, Score):
                answers[q_id] = self.evaluate_score(q_id, state_str, q)

        input_tokens = max(1, (len(state_str) + total_q_chars) // 4)
        output_tokens = len(answers) * 8
        resolved_model = f"von-{self.variant}" if model not in ("von-latest", "von-preview", "jev-latest", "jev-preview", None) else "von-1.0.0"

        return SystemOneResponse(
            model=resolved_model,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

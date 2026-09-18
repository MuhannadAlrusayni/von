"""ModernBERT-151M backend for Von (heman10x/rlcd-modernbert-151m)."""

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional, Union

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


def _format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, indent=2, ensure_ascii=False)
    except Exception:
        return str(state)


class ModernBERTBackend(BaseBackend):
    """GLiClass / ModernBERT decision backend (151M parameters, ~35ms latency)."""

    def __init__(self, model_id: str = "heman10x/rlcd-modernbert-151m", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._pipe = None
        self._lock = threading.Lock()

    def _get_pipeline(self):
        with self._lock:
            if self._pipe is None:
                from gliclass import GLiClassModel, ZeroShotClassificationPipeline
                from transformers import AutoTokenizer

                model = GLiClassModel.from_pretrained(self.model_id)
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._pipe = ZeroShotClassificationPipeline(
                    model, tokenizer, classification_type="multi-label", device=self.device
                )
            return self._pipe

    def evaluate_choice(self, q_id: str, state_text: str, q: Choice) -> ChoiceAnswer:
        pipe = self._get_pipeline()
        options = list(q.criteria.keys())
        if not options:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        labels = []
        for opt in options:
            desc = q.criteria.get(opt)
            labels.append(f"{q.instructions} -> {opt}: {desc}" if desc else f"{q.instructions} -> {opt}")

        res = pipe(state_text, labels)[0]
        raw_scores = []
        for opt, label in zip(options, labels):
            score_val = 0.0
            for item in res:
                if item["label"] == label:
                    score_val = float(item["score"])
                    break
            raw_scores.append(score_val)

        # Softmax normalization
        exp_s = [math.exp(s / 0.5) for s in raw_scores]
        sum_exp = sum(exp_s) or 1.0
        probs = [e / sum_exp for e in exp_s]

        prob_dict = {opt: round(p, 4) for opt, p in zip(options, probs)}
        best_choice = max(prob_dict, key=lambda k: prob_dict[k])
        sorted_p = sorted(prob_dict.values(), reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ChoiceAnswer(choice=best_choice, probabilities=prob_dict, confidence=confidence)

    def evaluate_score(self, q_id: str, state_text: str, q: Score) -> ScoreAnswer:
        pipe = self._get_pipeline()
        levels = q.criteria
        if not levels:
            return ScoreAnswer(score=0.0, confidence=0.0, legend={}, probabilities={})

        legend = {}
        labels = []
        for i, item in enumerate(levels):
            desc = item.get("what", "") if isinstance(item, dict) else str(item)
            legend[str(i)] = desc
            labels.append(f"Level {i}: {desc}")

        res = pipe(state_text, labels)[0]
        raw_scores = []
        for i, label in enumerate(labels):
            s_val = 0.0
            for item in res:
                if item["label"] == label:
                    s_val = float(item["score"])
                    break
            raw_scores.append(s_val)

        exp_s = [math.exp(s / 0.5) for s in raw_scores]
        sum_exp = sum(exp_s) or 1.0
        probs = [e / sum_exp for e in exp_s]

        prob_dict = {str(i): round(p, 4) for i, p in enumerate(probs)}
        weighted_score = round(sum(i * p for i, p in enumerate(probs)), 2)
        sorted_p = sorted(probs, reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ScoreAnswer(score=weighted_score, confidence=confidence, legend=legend, probabilities=prob_dict)

    def evaluate_noul(self, q_id: str, state_text: str, q: Noul) -> NoulAnswer:
        pipe = self._get_pipeline()
        crit = q.criteria or {}
        pos_crit = crit.get("true", "")
        neg_crit = crit.get("false", "")

        label_pos = f"{q.instructions} Yes, true. {pos_crit}".strip()
        label_neg = f"{q.instructions} No, false. {neg_crit}".strip()

        res = pipe(state_text, [label_pos, label_neg])[0]
        s_pos, s_neg = 0.5, 0.5
        for item in res:
            if item["label"] == label_pos:
                s_pos = float(item["score"])
            elif item["label"] == label_neg:
                s_neg = float(item["score"])

        exp_pos = math.exp(s_pos / 0.5)
        exp_neg = math.exp(s_neg / 0.5)
        p_true = round(exp_pos / (exp_pos + exp_neg), 2)

        return NoulAnswer(noul=p_true)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = "von-modernbert-151m",
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
                    raise ValueError(f"Unknown question type: {q_type}")
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

        return SystemOneResponse(
            model=model,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

"""Qwen 0.5B backend for Von using Parallel Constrained Decoding (PCD)."""

import json
import os
import threading
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

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


class QwenPCDBackend(BaseBackend):
    """Causal PCD backend using Qwen2.5-0.5B-Instruct reading logits directly."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _get_model_and_tok(self):
        with self._lock:
            if self._model is None or self._tokenizer is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=torch.float32,
                )
                self._model.to(self.device)
                self._model.eval()
            return self._model, self._tokenizer

    def evaluate_choice(self, q_id: str, state_text: str, q: Choice) -> ChoiceAnswer:
        model, tok = self._get_model_and_tok()
        options = list(q.criteria.keys())
        if not options:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        # Single-token decision prompt: map options to option tokens or letters
        opt_lines = []
        for opt in options:
            desc = q.criteria.get(opt)
            opt_lines.append(f"- '{opt}': {desc}" if desc else f"- '{opt}'")

        prompt = (
            f"<|im_start|>system\n"
            f"You are an automated decision engine. Select exactly one option ID from: {', '.join(options)}.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Context: {state_text}\n"
            f"Question: {q.instructions}\n\n"
            f"Available Options:\n" + "\n".join(opt_lines) + "\n\n"
            f"Selected option ID:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = tok.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(input_ids)
            last_logits = out.logits[0, -1, :]

        # Get first token ID of each option ID
        cand_tokens = [tok.encode(opt, add_special_tokens=False)[0] for opt in options]
        cand_logits = torch.tensor([last_logits[tid].item() for tid in cand_tokens])

        probs = F.softmax(cand_logits, dim=-1).tolist()
        prob_dict = {opt: round(p, 4) for opt, p in zip(options, probs)}

        winner_idx = probs.index(max(probs))
        best_choice = options[winner_idx]

        sorted_p = sorted(probs, reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ChoiceAnswer(choice=best_choice, probabilities=prob_dict, confidence=confidence)

    def evaluate_score(self, q_id: str, state_text: str, q: Score) -> ScoreAnswer:
        model, tok = self._get_model_and_tok()
        levels = q.criteria
        if not levels:
            return ScoreAnswer(score=0.0, confidence=0.0, legend={}, probabilities={})

        legend = {}
        level_keys = [str(i) for i in range(len(levels))]
        level_lines = []
        for i, item in enumerate(levels):
            desc = item.get("what", "") if isinstance(item, dict) else str(item)
            legend[str(i)] = desc
            level_lines.append(f"- Level {i}: {desc}")

        prompt = (
            f"<|im_start|>system\n"
            f"You are an automated rating engine. Rate the context with a level number from: {', '.join(level_keys)}.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Context: {state_text}\n"
            f"Scale Rubric:\n" + "\n".join(level_lines) + "\n\n"
            f"Selected Level Number:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = tok.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(input_ids)
            last_logits = out.logits[0, -1, :]

        cand_tokens = [tok.encode(k, add_special_tokens=False)[0] for k in level_keys]
        cand_logits = torch.tensor([last_logits[tid].item() for tid in cand_tokens])

        probs = F.softmax(cand_logits, dim=-1).tolist()
        prob_dict = {str(i): round(p, 4) for i, p in enumerate(probs)}
        weighted_score = round(sum(i * p for i, p in enumerate(probs)), 2)

        sorted_p = sorted(probs, reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ScoreAnswer(score=weighted_score, confidence=confidence, legend=legend, probabilities=prob_dict)

    def evaluate_noul(self, q_id: str, state_text: str, q: Noul) -> NoulAnswer:
        model, tok = self._get_model_and_tok()
        crit = q.criteria or {}
        pos_crit = crit.get("true", "")
        neg_crit = crit.get("false", "")

        guidance = f" True: {pos_crit}. False: {neg_crit}." if (pos_crit or neg_crit) else ""

        prompt = (
            f"<|im_start|>system\n"
            f"Answer with only 'true' or 'false'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Context: {state_text}\n"
            f"Condition: {q.instructions}{guidance}\n\n"
            f"Is the condition true or false?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = tok.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(input_ids)
            last_logits = out.logits[0, -1, :]

        tok_true = tok.encode("true", add_special_tokens=False)[0]
        tok_false = tok.encode("false", add_special_tokens=False)[0]

        logits = torch.tensor([last_logits[tok_true].item(), last_logits[tok_false].item()])
        probs = F.softmax(logits, dim=-1).tolist()

        return NoulAnswer(noul=round(probs[0], 2))

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = "von-qwen-0.5b",
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

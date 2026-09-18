"""Needle 3 backend for Von (14MB, CPU, ultra-lightweight)."""

import json
import math
import os
import re
import threading
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, create_model
from typing_extensions import Literal

import needle
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


def _to_camel(s: str) -> str:
    parts = s.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p) or "Decision"


def _cos_sim(vec_a: List[float], vec_b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _softmax(scores: List[float], temperature: float = 0.003) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp((s - max_score) / max(temperature, 1e-6)) for s in scores]
    sum_exp = sum(exp_scores)
    if sum_exp <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [e / sum_exp for e in exp_scores]


def _format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, indent=2, ensure_ascii=False)
    except Exception:
        return str(state)


def _extract_backtick_refs(text: str) -> List[str]:
    return re.findall(r"`([^`]+)`", text)


def _resolve_state_path(state: Any, path: str) -> Optional[Any]:
    parts = re.split(r"\.|\b(?=\[)", path)
    curr = state
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            try:
                idx = int(part[1:-1])
                curr = curr[idx]
            except Exception:
                return None
        elif isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr


class NeedleBackend(BaseBackend):
    """Execution engine powered by Needle 3."""

    def __init__(self, generation: int = 3):
        self._generation = generation
        self._needle: Optional[needle.Needle] = None
        self._lock = threading.Lock()

    def _get_needle(self) -> needle.Needle:
        with self._lock:
            if self._needle is None:
                self._needle = needle.Needle(generation=self._generation)
            return self._needle

    def embed(self, text: str) -> List[float]:
        agent = self._get_needle()
        return agent.embed(text)

    def evaluate_choice(
        self,
        q_id: str,
        state_text: str,
        q: Choice,
        state_emb: Optional[List[float]] = None,
        temperature: float = 0.003,
    ) -> ChoiceAnswer:
        options = list(q.criteria.keys())
        if not options:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        if state_emb is None:
            state_emb = self.embed(state_text)

        # 1. Native needle extraction
        extracted_choice = None
        try:
            desc_items = [
                f"'{k}': {v}" if v else f"'{k}'" for k, v in q.criteria.items()
            ]
            full_desc = f"{q.instructions}. Allowed choices: {'; '.join(desc_items)}"
            lit_type = Literal[tuple(options)]  # type: ignore
            model_name = _to_camel(q_id)
            DynModel = create_model(model_name, choice=(lit_type, Field(description=full_desc)))

            res = needle.extract(state_text, DynModel, strict=False, generation=self._generation)
            extracted_choice = getattr(res, "choice", None)
            if extracted_choice not in options:
                extracted_choice = None
        except Exception:
            extracted_choice = None

        # 2. Semantic similarity
        candidate_prompts = []
        for key in options:
            desc = q.criteria.get(key)
            text = f"{q.instructions} Choice: {key}. Meaning: {desc}" if desc else f"{q.instructions} Choice: {key}"
            candidate_prompts.append(text)

        sims = [_cos_sim(state_emb, self.embed(cand)) for cand in candidate_prompts]

        # Polarity & negation alignment
        state_lower = state_text.lower()
        state_has_pos = any(w in state_lower for w in ["completed", "passed", "succeeded", "healthy", "success", "working"])
        state_has_neg = any(w in state_lower for w in ["failed", "crashed", "error", "decline", "timed out", "rollback initiated"])

        for idx, key in enumerate(options):
            desc = (q.criteria.get(key) or "").lower()
            opt_is_neg = any(w in desc for w in ["not", "did not", "never", "fail", "cannot"])
            if state_has_pos and not state_has_neg and opt_is_neg:
                sims[idx] -= 0.005
            elif state_has_neg and not state_has_pos and not opt_is_neg and ("success" in desc or "succeeded" in desc):
                sims[idx] -= 0.005

        probs = _softmax(sims, temperature=temperature)

        if extracted_choice and extracted_choice in options:
            is_contradicting_polarity = False
            if state_has_pos and not state_has_neg:
                extracted_desc = (q.criteria.get(extracted_choice) or "").lower()
                if any(w in extracted_desc for w in ["not", "did not", "never", "fail", "cannot"]) or extracted_choice == "no":
                    is_contradicting_polarity = True

            if not is_contradicting_polarity:
                win_idx = options.index(extracted_choice)
                if probs[win_idx] < max(probs):
                    sims[win_idx] += 0.015
                    probs = _softmax(sims, temperature=temperature)
                best_choice = extracted_choice
            else:
                win_idx = max(range(len(probs)), key=lambda i: probs[i])
                best_choice = options[win_idx]
        else:
            win_idx = max(range(len(probs)), key=lambda i: probs[i])
            best_choice = options[win_idx]

        prob_dict = {opt: round(p, 4) for opt, p in zip(options, probs)}
        sorted_p = sorted(prob_dict.values(), reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ChoiceAnswer(choice=best_choice, probabilities=prob_dict, confidence=confidence)

    def evaluate_score(
        self,
        q_id: str,
        state_text: str,
        q: Score,
        state_emb: Optional[List[float]] = None,
        temperature: float = 0.003,
    ) -> ScoreAnswer:
        levels = q.criteria
        if not levels:
            return ScoreAnswer(score=0.0, confidence=0.0, legend={}, probabilities={})

        if state_emb is None:
            state_emb = self.embed(state_text)

        legend: Dict[str, str] = {}
        candidate_prompts = []
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
            candidate_prompts.append(f"Level {i}: {desc}")

        sims = [_cos_sim(state_emb, self.embed(cand)) for cand in candidate_prompts]
        probs = _softmax(sims, temperature=temperature)

        prob_dict = {str(i): round(p, 4) for i, p in enumerate(probs)}
        weighted_score = round(sum(i * p for i, p in enumerate(probs)), 2)

        sorted_p = sorted(probs, reverse=True)
        confidence = round(max(0.0, min(1.0, sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))), 3)

        return ScoreAnswer(score=weighted_score, confidence=confidence, legend=legend, probabilities=prob_dict)

    def evaluate_noul(
        self,
        q_id: str,
        state_text: str,
        q: Noul,
        state_emb: Optional[List[float]] = None,
        temperature: float = 0.003,
    ) -> NoulAnswer:
        if state_emb is None:
            state_emb = self.embed(state_text)

        crit = q.criteria or {}
        pos_crit = crit.get("true", "")
        neg_crit = crit.get("false", "")

        pos_prompt = f"{q.instructions} Confirmation, yes, condition true. {pos_crit}".strip()
        neg_prompt = f"{q.instructions} Denial, no, condition false. {neg_crit}".strip()

        sim_pos = _cos_sim(state_emb, self.embed(pos_prompt))
        sim_neg = _cos_sim(state_emb, self.embed(neg_prompt))

        probs = _softmax([sim_pos, sim_neg], temperature=temperature)
        prob_true = round(max(0.0, min(1.0, probs[0])), 2)

        return NoulAnswer(noul=prob_true)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = "von-latest",
    ) -> SystemOneResponse:
        state_str = _format_state(state)
        state_emb = self.embed(state_str)

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

            refs = _extract_backtick_refs(q.instructions)
            target_text = state_str
            target_emb = state_emb

            if refs and isinstance(state, (dict, list)):
                ref_vals = [_resolve_state_path(state, r) for r in refs]
                valid_vals = [v for v in ref_vals if v is not None]
                if valid_vals:
                    sub_text = "\n".join(f"{r}: {_format_state(v)}" for r, v in zip(refs, valid_vals))
                    target_text = sub_text
                    target_emb = self.embed(sub_text)

            if isinstance(q, Noul):
                answers[q_id] = self.evaluate_noul(q_id, target_text, q, state_emb=target_emb)
            elif isinstance(q, Choice):
                answers[q_id] = self.evaluate_choice(q_id, target_text, q, state_emb=target_emb)
            elif isinstance(q, Score):
                answers[q_id] = self.evaluate_score(q_id, target_text, q, state_emb=target_emb)

        input_tokens = max(1, (len(state_str) + total_q_chars) // 4)
        output_tokens = len(answers) * 8

        resolved_model = "von-1.0.0" if model in ("von-latest", "von-preview", "jev-latest", "jev-preview") else model

        return SystemOneResponse(
            model=resolved_model,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

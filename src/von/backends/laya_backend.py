"""Laya 421M RLCD backend for Von (convaiinnovations/laya)."""

import json
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


class LayaBackend(BaseBackend):
    """Non-autoregressive System 1 decision engine using convaiinnovations/laya."""

    def __init__(self, model_id: str = "convaiinnovations/laya", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or os.environ.get("VON_DEVICE")
        self._agent = None
        self._lock = threading.Lock()

    def _get_agent(self):
        with self._lock:
            if self._agent is None:
                import laya

                dev_arg = None if self.device in (None, "auto") else self.device
                self._agent = laya.load(self.model_id, device=dev_arg)
            return self._agent

    def evaluate_choice(
        self,
        q_id: str,
        state_text: str,
        q: Choice,
        **kwargs,
    ) -> ChoiceAnswer:
        agent = self._get_agent()
        q_payload = {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": q.criteria,
        }
        res = agent.predict(state_text, {q_id: q_payload})
        raw_ans = res["answers"][q_id]
        return ChoiceAnswer(
            choice=raw_ans["choice"],
            probabilities=raw_ans["probabilities"],
            confidence=raw_ans["confidence"],
        )

    def evaluate_score(
        self,
        q_id: str,
        state_text: str,
        q: Score,
        **kwargs,
    ) -> ScoreAnswer:
        agent = self._get_agent()
        criteria_list = []
        for item in q.criteria:
            if isinstance(item, dict):
                what = item.get("what", "")
                examples = item.get("examples", [])
                ex_str = f" Examples: {', '.join(examples)}" if examples else ""
                desc = f"{what}{ex_str}".strip()
            else:
                desc = str(item)
            criteria_list.append(desc)

        q_payload = {
            "type": "score",
            "instructions": q.instructions,
            "criteria": criteria_list,
        }
        res = agent.predict(state_text, {q_id: q_payload})
        raw_ans = res["answers"][q_id]
        return ScoreAnswer(
            score=raw_ans["score"],
            confidence=raw_ans["confidence"],
            legend=raw_ans.get("legend", {}),
            probabilities=raw_ans["probabilities"],
        )

    def evaluate_noul(
        self,
        q_id: str,
        state_text: str,
        q: Noul,
        **kwargs,
    ) -> NoulAnswer:
        agent = self._get_agent()
        q_payload = {
            "type": "noul",
            "instructions": q.instructions,
        }
        if q.criteria:
            q_payload["criteria"] = q.criteria

        res = agent.predict(state_text, {q_id: q_payload})
        raw_ans = res["answers"][q_id]
        return NoulAnswer(noul=raw_ans["noul"])

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = "von-latest",
    ) -> SystemOneResponse:
        agent = self._get_agent()
        state_str = _format_state(state)

        internal_questions = {}
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

            if isinstance(q, Choice):
                internal_questions[q_id] = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            elif isinstance(q, Score):
                crit_list = []
                for item in q.criteria:
                    if isinstance(item, dict):
                        what = item.get("what", "")
                        exs = item.get("examples", [])
                        ex_str = f" Examples: {', '.join(exs)}" if exs else ""
                        crit_list.append(f"{what}{ex_str}".strip())
                    else:
                        crit_list.append(str(item))
                internal_questions[q_id] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": crit_list,
                }
            elif isinstance(q, Noul):
                n_dict = {
                    "type": "noul",
                    "instructions": q.instructions,
                }
                if q.criteria:
                    n_dict["criteria"] = q.criteria
                internal_questions[q_id] = n_dict

        res = agent.predict(state_str, internal_questions)
        answers: Dict[str, Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]] = {}

        for q_id, ans_dict in res["answers"].items():
            a_type = ans_dict["type"]
            if a_type == "choice":
                answers[q_id] = ChoiceAnswer(
                    choice=ans_dict["choice"],
                    probabilities=ans_dict["probabilities"],
                    confidence=ans_dict["confidence"],
                )
            elif a_type == "score":
                answers[q_id] = ScoreAnswer(
                    score=ans_dict["score"],
                    confidence=ans_dict["confidence"],
                    legend=ans_dict.get("legend", {}),
                    probabilities=ans_dict["probabilities"],
                )
            elif a_type == "noul":
                answers[q_id] = NoulAnswer(noul=ans_dict["noul"])

        usage_dict = res.get("usage", {})
        resolved_model = "von-1.0.0" if model in ("von-latest", "von-preview", "jev-latest", "jev-preview", None) else model

        return SystemOneResponse(
            model=resolved_model,
            answers=answers,
            usage=Usage(
                input_tokens=usage_dict.get("input_tokens", len(state_str) // 4),
                output_tokens=usage_dict.get("output_tokens", 0),
            ),
        )

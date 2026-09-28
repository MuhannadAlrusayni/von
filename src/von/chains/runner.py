"""Chain-of-options runtime.

A chain is a TOML file: a route description, typed slots, deterministic
steps over the operator library, and an answer rule. At inference the
controller makes only typed Choice decisions with the same backend that
answers ordinary questions:

  level 0  gate     -- cheap: the state has >= 2 numerals, else skip
  level 1  route    -- Choice over chain descriptions (+ "none")
  level 2  bind     -- one Choice per slot over proposed spans of its kind
  execute           -- run the steps; any exception => fall back to plain
  answer            -- describe the computed result in one sentence and ask
                       the ORIGINAL question about that sentence (Choice /
                       Noul / Score), or match a formatted value to an option

Zero generated tokens. Regex proposes; the model chooses; code computes.
"""

from __future__ import annotations

import glob
import os
import re
import threading
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..types import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from . import ops as _ops
from .spans import Span, by_kind, propose

NUMERAL_GATE = re.compile(r"\d")


@dataclass
class Slot:
    name: str
    kind: str
    ask: str
    optional: bool = False
    prefer: str = ""  # "with_time" | "with_zone" | "" (ranking hint only)


@dataclass
class Step:
    op: str
    args: List[Any]
    out: str


@dataclass
class Chain:
    name: str
    description: str
    slots: List[Slot]
    steps: List[Step]
    answer: Dict[str, Any]
    path: str = ""
    requires: List[str] = field(default_factory=list)


def load_chain(path: str) -> Chain:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    slots = [Slot(**s) for s in d.get("slots", [])]
    steps = [Step(op=s["op"], args=list(s.get("args", [])), out=s["out"]) for s in d.get("steps", [])]
    for s in steps:
        if s.op not in _ops.OPS:
            raise ValueError(f"{path}: unknown op '{s.op}'")
    return Chain(name=d["name"], description=d["description"], slots=slots, steps=steps,
                 answer=d.get("answer", {}), path=path, requires=list(d.get("requires", [])))


def load_chains(directory: str) -> List[Chain]:
    return [load_chain(p) for p in sorted(glob.glob(os.path.join(directory, "*.toml")))]


@dataclass
class Trace:
    chain: Optional[str] = None
    route_probs: Dict[str, float] = field(default_factory=dict)
    bindings: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    fallback: Optional[str] = None


def _fmt(v: Any) -> str:
    if isinstance(v, Span):
        if isinstance(v.value, datetime):
            return _fmt(_ops._dt(v))
        if isinstance(v.value, dict) and "unit" in v.value:
            n = v.value["n"]
            return f"{n:g} {v.value['unit'].replace('_', ' ')}"
        return v.text
    if v is None:
        return "unspecified"
    if isinstance(v, datetime):
        s = v.strftime("%A %d %B %Y %H:%M")
        if v.tzinfo is not None:
            s += " " + (v.tzname() or v.strftime("%z"))
        return s
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 0.01 or v == 0 else f"{v:.4g}"
    if isinstance(v, list):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v[:12])
    return str(v)


class ChainRunner:
    """Drives chains using a backend's plain Choice/Noul/Score primitives."""

    def __init__(self, backend, chains_dir: str, min_numerals: int = 2):
        self.backend = backend
        self.chains = load_chains(chains_dir)
        self.min_numerals = min_numerals
        self._local = threading.local()

    # --- recursion guard: sub-decisions must never re-enter the chain path --
    def active(self) -> bool:
        return bool(getattr(self._local, "busy", False))

    def _choice(self, state: str, instructions: str, criteria: Dict[str, str]) -> ChoiceAnswer:
        self._local.busy = True
        try:
            return self.backend.evaluate_choice("chain", state, Choice(type="choice", instructions=instructions, criteria=criteria))
        finally:
            self._local.busy = False

    def _ask_original(self, description: str, q: Any):
        self._local.busy = True
        try:
            if isinstance(q, Choice):
                return self.backend.evaluate_choice("chain", description, q)
            if isinstance(q, Noul):
                return self.backend.evaluate_noul("chain", description, q)
            return self.backend.evaluate_score("chain", description, q)
        finally:
            self._local.busy = False

    # --------------------------------------------------------------- run --
    def applicable(self, state: str) -> bool:
        return len(NUMERAL_GATE.findall(state)) >= self.min_numerals and bool(self.chains)

    def run(self, state: str, q: Any) -> Tuple[Optional[Any], Trace]:
        tr = Trace()
        if not self.applicable(state):
            tr.fallback = "gate"
            return None, tr
        instructions = getattr(q, "instructions", "") or ""

        # level 1: route
        crit = {c.name: c.description for c in self.chains}
        crit["none"] = "None of these computations answers the question; answer it directly from the text."
        route = self._choice(state, f"Which computation, if any, is needed to answer this question: {instructions}", crit)
        tr.route_probs = dict(route.probabilities)
        if route.choice == "none" or route.choice not in crit:
            tr.fallback = "route:none"
            return None, tr
        chain = next(c for c in self.chains if c.name == route.choice)
        tr.chain = chain.name

        # level 2: bind
        spans = propose(state)
        env: Dict[str, Any] = {"state": state}
        bound: set = set()
        for slot in chain.slots:
            # A span already bound to an earlier slot is off the table: the
            # model otherwise re-picks the most salient span for every role.
            cands = [s for s in by_kind(spans, slot.kind) if s.key() not in bound]
            if slot.prefer == "with_time":
                cands = sorted(cands, key=lambda s: not s.meta.get("has_time", False))
            if slot.prefer == "with_zone":
                cands = sorted(cands, key=lambda s: s.meta.get("zone") is None)
            if slot.kind in ("number", "amount"):
                cands = sorted(cands, key=lambda s: not s.meta.get("rich", False))
            cands = cands[:12]
            if not cands:
                if slot.optional:
                    env[slot.name] = None
                    continue
                tr.fallback = f"bind:{slot.name}:no-candidates"
                return None, tr
            if len(cands) == 1:
                chosen = cands[0]
            else:
                c = {f"s{i}": s.text for i, s in enumerate(cands)}
                ans = self._choice(state, slot.ask, c)
                chosen = cands[int(ans.choice[1:])] if ans.choice in c else cands[0]
            env[slot.name] = chosen
            bound.add(chosen.key())
            tr.bindings[slot.name] = chosen.text

        # execute
        try:
            for st in chain.steps:
                args = [env[a] if isinstance(a, str) and a in env else a for a in st.args]
                env[st.out] = _ops.OPS[st.op](*args)
                tr.values[st.out] = env[st.out] if not isinstance(env[st.out], list) else env[st.out][:12]
                if isinstance(env[st.out], list) and not env[st.out]:
                    raise ValueError(f"{st.op} produced an empty series")
        except Exception as e:  # noqa: BLE001 - any executor failure means "not our item"
            tr.fallback = f"exec:{type(e).__name__}:{e}"
            return None, tr

        # answer
        ans_cfg = chain.answer
        try:
            desc = ans_cfg.get("describe", "").format(**{k: _fmt(v) if k != "state" else "" for k, v in env.items()},
                                                      **{f"{k}_raw": v for k, v in env.items() if k != "state"})
        except (KeyError, ValueError, IndexError) as e:
            tr.fallback = f"describe:{e}"
            return None, tr
        tr.description = desc
        mode = ans_cfg.get("mode", "ask")

        if mode == "value" and isinstance(q, Choice):
            key = ans_cfg.get("value", "")
            val = env.get(key)
            if isinstance(val, (int, float)):
                hit = _match_value(val, q.criteria)
                if hit:
                    probs = {k: (0.9 if k == hit else 0.1 / max(1, len(q.criteria) - 1)) for k in q.criteria}
                    return ChoiceAnswer(choice=hit, probabilities=probs, confidence=0.8), tr
            # no exact match => fall through to asking
        prompt = f"{desc}\n\nOriginal question context: {instructions}" if instructions else desc
        return self._ask_original(prompt, q), tr


def _match_value(val: float, criteria: Dict[str, str]) -> Optional[str]:
    """Pick the option whose label or description contains the formatted value."""
    cands = {f"{val:.2f}", f"{val:,.2f}", f"{val:.0f}", f"{val:,.0f}", f"{val:g}"}
    for k, d in criteria.items():
        hay = f"{k} {d}".replace("_", " ").replace(",", "")
        for c in cands:
            if c.replace(",", "") in hay:
                return k
    return None

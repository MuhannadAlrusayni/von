"""Spine-style generator for JevBench's standard tier: extraction, routing, adequacy.

Follows jabr/classifier-benchmark training/spines (gold by construction,
semantics as controlled variation, shortcut gates), extended from the one
family that repo ships (boundary noul) to the three standard-tier families
where Von trails Laya (69 vs 76).

A spine here is (world, rule, question):

  extraction  world = ordered statements about a slot (proposed / confirmed /
              cancelled / claimed / hypothetical); rule = "last confirmed
              statement, else unknown". Operators: supersede, cancel,
              attribution (claim != fact), hypothetical, negation, distractor,
              paraphrase, coref.
  routing     world = a request with a primary category plus optional
              override trigger; rule = category unless the override clause in
              the instructions fires. Operators: override, keyword decoy,
              paraphrase, negation ("without running anything").
  adequacy    world = request with explicit constraints + a response;
              rule = response satisfies value AND every constraint.
              Operators: wrong value, extra text, wrong format, missing part,
              constraint violation, near-miss number, paraphrase.

Soft targets follow jabr's noisy-reader schedule: strength = min(evidence
strengths) x decisiveness; target = strength on gold, floored at .50,
capped at .97, spread evenly over the wrong options.

Gates (jabr's): question-only bag-of-words probe must stay <= 0.60 on the
gold option id; keyword probe (pick the option whose id appears most in
the state) must stay <= 0.60 on operator-tagged items. Failing a gate exits
non-zero: a leaking corpus must never reach a training run by accident.

    uv run python training/generate_standard_spines.py --n 4000 --seed 7 --out data_standard/spines.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

STRENGTH = {"verified": 0.97, "asserted": 0.95, "paraphrased": 0.90, "inferred": 0.80, "claimed": 0.60}
DECISIVE = {"explicit": 0.85, "recent": 0.70, "none": 1.0}


@dataclass
class Item:
    family: str
    state: str
    question: str
    options: List[Tuple[str, str]]
    label: str
    strength: float
    ops: List[str]
    variant: str
    extra: dict = field(default_factory=dict)

    def row(self, idx: int) -> dict:
        n = len(self.options)
        s = max(0.50, min(0.97, self.strength))
        rest = (1.0 - s) / max(1, n - 1)
        target = [s if oid == self.label else rest for oid, _ in self.options]
        return {
            "id": f"spine:{self.family}:{self.variant}:{idx:05d}",
            "state": self.state,
            "question": self.question,
            "options": [{"id": oid, "description": d} for oid, d in self.options],
            "label": self.label,
            "target": target,
            "family": f"spine_{self.family}",
            "source": {"kind": "spine", "family": self.family, "variant": self.variant, "ops": self.ops, **self.extra},
        }


# ============================================================ extraction ==
@dataclass(frozen=True)
class Slot:
    name: str
    question_forms: Tuple[str, ...]
    values: Dict[str, Tuple[str, Tuple[str, ...]]]  # id -> (option description, surface forms)
    unknown_desc: str
    confirm: Tuple[str, ...]
    propose: Tuple[str, ...]
    cancel: Tuple[str, ...]
    claim: Tuple[str, ...]
    hypo: Tuple[str, ...]
    negate: Tuple[str, ...]
    decoy: Tuple[str, ...]


SLOTS = [
    Slot(
        name="delivery method",
        question_forms=(
            "Extract the final confirmed delivery method. Ignore cancelled plans and hypothetical alternatives. Choose unknown if no final method is confirmed.",
            "Which delivery method was finally confirmed? Disregard options that were only considered or later cancelled; pick unknown when nothing is confirmed.",
            "Identify the delivery method that stands as confirmed at the end of the exchange. Cancelled or merely proposed methods do not count; unknown if none.",
        ),
        values={
            "courier": ("Courier delivery", ("courier", "a courier", "courier delivery")),
            "pickup": ("Customer pickup", ("pickup", "pickup at the depot", "collection in person")),
            "post": ("Postal service", ("post", "regular post", "the postal service")),
            "locker": ("Parcel locker", ("a parcel locker", "the locker", "locker drop")),
        },
        unknown_desc="No final confirmed method",
        confirm=("{s} confirmed {v}.", "It is settled: {v}.", "Final answer from {s}: {v}.", "{s} went with {v} in the end."),
        propose=("{s} suggested {v}.", "We considered {v}.", "{v} was floated as an option.", "{s} asked whether {v} would work."),
        cancel=("{s} then cancelled {v}.", "The {v} plan was dropped.", "Scratch {v}; that fell through.", "{s} backed out of {v}."),
        claim=("{s} says the warehouse promised {v}.", "According to {s}, it was supposed to be {v}.", "{s} claims {v} had been agreed."),
        hypo=("If {v} had been available we would have taken it.", "Had the deadline allowed, {v} would have been the choice.", "{s} would prefer {v} in an ideal world."),
        negate=("{s} made clear it would not be {v}.", "Not {v} — that was ruled out.", "{s} said no to {v}."),
        decoy=("The invoice mentions a courier surcharge from last month.", "A pickup reminder for an unrelated order came through.", "The post office holiday schedule was attached."),
    ),
    Slot(
        name="meeting day",
        question_forms=(
            "Extract the final agreed meeting day. Ignore days that were proposed but not agreed or later moved. Choose unknown if no day stands agreed.",
            "Which day did the meeting end up on? Proposals and superseded dates do not count; unknown if nothing is settled.",
            "State the meeting day that is agreed at the end of the thread. Moved or merely suggested days are out; unknown if none.",
        ),
        values={
            "monday": ("Monday", ("Monday",)), "tuesday": ("Tuesday", ("Tuesday",)),
            "wednesday": ("Wednesday", ("Wednesday",)), "thursday": ("Thursday", ("Thursday",)),
            "friday": ("Friday", ("Friday",)),
        },
        unknown_desc="No agreed day",
        confirm=("{s} locked in {v}.", "{v} it is, {s} confirmed.", "Agreed: {v}.", "{s} booked the room for {v}."),
        propose=("{s} floated {v}.", "Could {v} work?", "{s} pencilled in {v} tentatively.", "{v} was on the table."),
        cancel=("{s} then moved it off {v}.", "{v} is off; conflict came up.", "{s} pulled the {v} slot.", "Forget {v}."),
        claim=("{s} says the organiser had said {v}.", "According to {s}, {v} was the original plan.", "{s} recalls {v} being mentioned."),
        hypo=("If the client were free, {v} would be ideal.", "{s} would have picked {v} given the choice.", "Were it up to {s}, {v}."),
        negate=("{s} cannot do {v}.", "Not {v}, {s} is travelling.", "{v} is out for {s}."),
        decoy=("The recurring standup is every Monday.", "Payroll closes on Friday as usual.", "Someone asked about Wednesday's lunch order."),
    ),
    Slot(
        name="payment method",
        question_forms=(
            "Extract the payment method actually used for this order. Ignore methods that were attempted and failed or only offered. Choose unknown if none went through.",
            "Which payment method completed the order? Failed attempts and offered-but-unused methods do not count; unknown if nothing completed.",
            "Identify the payment method that the order finally went through on. Declined or merely available methods are out; unknown if none.",
        ),
        values={
            "card": ("Credit or debit card", ("card", "the card", "a debit card")),
            "transfer": ("Bank transfer", ("bank transfer", "a transfer", "wire")),
            "wallet": ("Digital wallet", ("the wallet", "a digital wallet", "wallet balance")),
            "invoice": ("Pay by invoice", ("invoice", "net-30 invoice", "invoicing")),
        },
        unknown_desc="No completed payment",
        confirm=("Payment went through by {v}.", "{s} paid by {v}; receipt issued.", "Settled via {v}.", "{v} cleared for the full amount."),
        propose=("{s} was offered {v}.", "{v} is available at checkout.", "{s} considered {v}.", "The site lists {v} as an option."),
        cancel=("The {v} attempt was declined.", "{v} failed twice.", "{s} abandoned the {v} attempt.", "{v} bounced."),
        claim=("{s} says {v} should have worked.", "According to {s}, {v} was charged.", "{s} insists {v} went through."),
        hypo=("If the limit allowed, {s} would use {v}.", "{v} would be simpler had it been enabled.", "{s} would rather pay by {v}."),
        negate=("{s} refused to use {v}.", "Not {v}; {s} does not have one.", "{v} was not accepted for this order."),
        decoy=("A card statement from another account was attached by mistake.", "The invoice template footer mentions wallet top-ups.", "There is a transfer fee note in the FAQ."),
    ),
]

SPEAKERS = ("The customer", "Dana", "The client", "Our contact", "Priya", "The buyer", "Marco", "The team lead")


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def gen_extraction(rng: random.Random) -> Item:
    slot = rng.choice(SLOTS)
    ids = list(slot.values)
    speaker = rng.choice(SPEAKERS)

    def sv(i):
        return rng.choice(slot.values[i][1])

    variant = rng.choice(["confirm_only", "propose_confirm", "confirm_cancel_confirm", "supersede", "confirm_cancel",
                          "claim_only", "hypothetical", "negation", "propose_only", "decoy"])
    lines: List[str] = []
    ops: List[str] = []
    gold = "unknown"
    strengths = [0.95]
    decisive = "none"
    pool = rng.sample(ids, min(3, len(ids)))
    a, b, c = (pool + pool)[:3]

    if variant == "confirm_only":
        lines = [rng.choice(slot.confirm).format(s=speaker, v=sv(a))]
        gold = a
    elif variant == "propose_confirm":
        lines = [rng.choice(slot.propose).format(s=speaker, v=sv(a)), rng.choice(slot.confirm).format(s=speaker, v=sv(b))]
        gold, ops = b, ["distractor"]
    elif variant == "confirm_cancel_confirm":
        lines = [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), rng.choice(slot.cancel).format(s=speaker, v=sv(a)),
                 rng.choice(slot.confirm).format(s=speaker, v=sv(b))]
        gold, ops, decisive = b, ["cancel", "supersede"], "explicit"
    elif variant == "supersede":
        lines = [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), "Later that day: " + _cap(rng.choice(slot.confirm).format(s=speaker, v=sv(b)))]
        gold, ops, decisive = b, ["supersede", "conflict"], "recent"
    elif variant == "confirm_cancel":
        lines = [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), rng.choice(slot.cancel).format(s=speaker, v=sv(a))]
        gold, ops = "unknown", ["cancel"]
    elif variant == "claim_only":
        lines = [rng.choice(slot.claim).format(s=speaker, v=sv(a))]
        if rng.random() < 0.5:
            lines.append(rng.choice(slot.propose).format(s=speaker, v=sv(b)))
        gold, ops, strengths = "unknown", ["attribution"], [STRENGTH["claimed"] + 0.25]
    elif variant == "hypothetical":
        lines = [rng.choice(slot.hypo).format(s=speaker, v=sv(a))]
        if rng.random() < 0.6:
            lines.append(rng.choice(slot.confirm).format(s=speaker, v=sv(b)))
            gold = b
        ops = ["hypothetical"]
    elif variant == "negation":
        lines = [rng.choice(slot.negate).format(s=speaker, v=sv(a))]
        if rng.random() < 0.6:
            lines.append(rng.choice(slot.confirm).format(s=speaker, v=sv(b)))
            gold = b
        ops = ["negation"]
    elif variant == "propose_only":
        lines = [rng.choice(slot.propose).format(s=speaker, v=sv(a)), rng.choice(slot.propose).format(s=speaker, v=sv(b))]
        gold, ops = "unknown", ["distractor"]
    elif variant == "decoy":
        lines = [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), rng.choice(slot.decoy)]
        gold, ops = a, ["distractor"]
    if "paraphrase" not in ops and rng.random() < 0.3:
        lines = [l.replace(" confirmed ", " signed off on ").replace("Agreed: ", "Done deal: ") for l in lines]
        ops.append("paraphrase")
        strengths.append(STRENGTH["paraphrased"])
    if rng.random() < 0.5:
        rng.shuffle(lines)  # order carries no meaning except for supersede/cancel chains
        if any(o in ops for o in ("supersede", "cancel")):
            lines = sorted(lines, key=lambda l: 0)  # keep chain order: re-derive
            lines = _rebuild_chain(slot, speaker, variant, a, b, sv, rng)
    lines = [_cap(l) for l in lines]
    state = " ".join(lines)
    strength = min(strengths) * DECISIVE[decisive]
    options = [(i, slot.values[i][0]) for i in ids] + [("unknown", slot.unknown_desc)]
    rng.shuffle(options)
    return Item("extraction", state, rng.choice(slot.question_forms), options, gold, strength, ops, variant)


def _rebuild_chain(slot, speaker, variant, a, b, sv, rng):
    if variant == "confirm_cancel_confirm":
        return [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), rng.choice(slot.cancel).format(s=speaker, v=sv(a)),
                rng.choice(slot.confirm).format(s=speaker, v=sv(b))]
    if variant == "supersede":
        return [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), "Later that day: " + _cap(rng.choice(slot.confirm).format(s=speaker, v=sv(b)))]
    return [rng.choice(slot.confirm).format(s=speaker, v=sv(a)), rng.choice(slot.cancel).format(s=speaker, v=sv(a))]


# =============================================================== routing ==
ROUTE_OPTIONS = [
    ("coding", "Self-contained code writing or explanation without repository operations"),
    ("coding_agent", "Inspect/edit repository files or run tests"),
    ("document", "Answer from a supplied document"),
    ("general", "None of the specialist categories"),
    ("math", "Self-contained calculation or proof"),
    ("tools", "Carry out an external service action"),
]
ROUTE_Q = (
    "Choose the specialist needed for the request. File edits with test execution use coding_agent, even if code-related.",
    "Route the request to the right specialist. Anything that touches repository files or runs tests goes to coding_agent, even when it is about code.",
    "Pick the category for this request. Requests that edit files or execute tests are coding_agent regardless of being code-related.",
)
ROUTE_REQ = {
    "coding": ("Write a Python function that reverses a linked list.", "Explain why this JavaScript closure keeps the old value.",
               "Show me a regex that matches ISO dates.", "What does `async with` do in Python?", "Rewrite this loop as a list comprehension: {snip}"),
    "coding_agent": ("Open src/auth.py, fix the token refresh bug, and run the test suite.", "Edit the config loader in the repo and make sure the tests pass.",
                     "Rename `fetchUser` across the codebase and run the linter and tests.", "Add a failing test for the parser in tests/ and then make it pass."),
    "document": ("Using the attached contract, what is the notice period?", "From the policy document below, is weekend work paid at 1.5x? {doc}",
                 "Based on the handbook I pasted, how many sick days are allowed?", "According to the spec above, which port does the service use?"),
    "general": ("What is a good name for a grey kitten?", "Recommend a podcast about history.", "How do I stop procrastinating?",
                "What's the weather usually like in Lisbon in March?", "Give me three ideas for a team offsite."),
    "math": ("Compute the least common multiple of 12 and 18.", "Prove that the sum of two even numbers is even.",
             "What is 17% of 2,350?", "Solve 3x + 7 = 25.", "How many ways can 5 people sit in a row?"),
    "tools": ("Book a table for four at 7pm tonight at Luigi's.", "Send the Q3 report to finance@example.com.",
              "Create a calendar event for Friday at 10.", "Cancel my 3pm ride.", "Post this update to the #ops channel."),
}
ROUTE_DECOY = {
    "coding": ["Also, the repository README is out of date but ignore that for now.", "(No need to run anything, just show the code.)"],
    "math": ["This is for a coding interview prep.", "I have the numbers in a spreadsheet file."],
    "general": ["I'm a software developer, if that matters.", "My calendar is packed this week."],
    "document": ["I can also email it to you if easier.", "The document is a PDF export from the repo wiki."],
    "tools": ["Here is the code for the email body: {snip}", "The math on the bill was 4 x 18.50."],
    "coding_agent": ["It's a small change, mostly a one-liner.", "Feel free to explain the fix too."],
}


def gen_routing(rng: random.Random) -> Item:
    variant = rng.choice(["plain", "decoy", "override_to_agent", "negated_override", "paraphrase"])
    ops: List[str] = []
    strength = 0.95
    if variant == "override_to_agent":
        base = rng.choice(["coding", "math"])
        req = rng.choice(ROUTE_REQ[base]) + rng.choice([" Then apply it to utils/dates.py in the repo and run the tests.",
                                                      " Put it in the repository and make sure the CI tests still pass.",
                                                      " Edit the file in place and run pytest."])
        gold, ops = "coding_agent", ["override", "exception"]
    elif variant == "negated_override":
        req = rng.choice(["Explain how I would refactor the auth module — do not edit any files or run anything.",
                          "Walk me through what the tests in tests/parser.py check, without executing them.",
                          "Describe a fix for the flaky test; I will apply it myself, no repository changes from you."])
        gold, ops, strength = "coding", ["negation", "override"], STRENGTH["inferred"]
    else:
        gold = rng.choice(list(ROUTE_REQ))
        req = rng.choice(ROUTE_REQ[gold])
        if variant == "decoy":
            req = req + " " + rng.choice(ROUTE_DECOY[gold])
            ops = ["distractor"]
        if variant == "paraphrase":
            req = req.replace("Write", "Could you draft").replace("Compute", "Work out").replace("Book", "Reserve")
            ops, strength = ["paraphrase"], STRENGTH["paraphrased"]
    req = req.replace("{snip}", "`for x in xs: out.append(f(x))`").replace("{doc}", "\"Overtime on Saturdays and Sundays is paid at 150% of base.\"")
    # Surface variation that carries no routing signal: greeting, urgency, sign-off.
    pre = rng.choice(["", "", "Hi — ", "Quick one: ", "Hey, ", "Sorry to bother you, but ", "Follow-up: ", "Please: "])
    post = rng.choice(["", "", " Thanks!", " It's fairly urgent.", " No rush.", " Let me know if anything is unclear.", " Cheers.", " (asked before, no answer yet)"])
    req = pre + (req[0].lower() + req[1:] if pre and not req.startswith(("`", "\"", "I ")) else req) + post
    options = list(ROUTE_OPTIONS)
    rng.shuffle(options)
    return Item("routing", req, rng.choice(ROUTE_Q), options, gold, strength, ops, variant)


# ============================================================== adequacy ==
ADEQ_Q = (
    "Does the response fully satisfy the request, using the supplied reference when present?",
    "Judge whether the response completely and correctly answers the request, honouring every explicit constraint.",
    "Is the response correct, complete, and compliant with all stated constraints of the request?",
)
ADEQ_OPTS = [("true", "Correct, complete, and follows all explicit constraints"),
             ("false", "Wrong, incomplete, unsupported or violates a constraint")]


def gen_adequacy(rng: random.Random) -> Item:
    kind = rng.choice(["sum", "list", "yesno", "json", "count_words", "reference"])
    variant = rng.choice(["ok", "wrong_value", "extra_text", "wrong_format", "missing_part", "near_miss", "unsupported"])
    ops: List[str] = []
    strength = 0.95
    gold = "true"
    if kind == "sum":
        a, b = rng.randint(3, 90), rng.randint(3, 90)
        req = f"Return only the sum of {a} and {b}."
        resp = str(a + b)
        if variant == "wrong_value":
            resp, gold = str(a + b + rng.choice([-10, 10, 1, -1]) * 2), "false"
        elif variant == "near_miss":
            resp, gold, strength = str(a + b + rng.choice([-1, 1])), "false", STRENGTH["inferred"]
        elif variant == "extra_text":
            resp, gold = f"The sum is {a + b}.", "false"
        elif variant == "wrong_format":
            resp, gold = f"{a} + {b} = {a + b}", "false"
        else:
            variant = "ok"
    elif kind == "list":
        n = rng.randint(2, 4)
        req = f"List exactly {n} primary colours, one per line, nothing else."
        cols = ["red", "blue", "yellow"][:n] if n <= 3 else ["red", "blue", "yellow", "green"]
        resp = "\n".join(cols)
        if n == 4:
            gold = "false"  # the request itself is unsatisfiable with 4 primaries; response lists green
            variant = "wrong_value"
        elif variant == "missing_part":
            resp, gold = "\n".join(cols[:-1]), "false"
        elif variant == "extra_text":
            resp, gold = "Sure! " + ", ".join(cols), "false"
        elif variant == "wrong_format":
            resp, gold = ", ".join(cols), "false"
        else:
            variant = "ok"
    elif kind == "yesno":
        x = rng.randint(10, 99)
        req = f"Answer only yes or no: is {x} an even number?"
        truth = "yes" if x % 2 == 0 else "no"
        resp = truth
        if variant == "wrong_value":
            resp, gold = ("no" if truth == "yes" else "yes"), "false"
        elif variant == "extra_text":
            resp, gold = f"{truth.capitalize()}, {x} is {'even' if truth == 'yes' else 'odd'}.", "false"
        else:
            variant = "ok"
    elif kind == "json":
        name, age = rng.choice(["Ana", "Lee", "Omar", "Sae"]), rng.randint(18, 70)
        req = f"Return a JSON object with keys name and age for {name}, aged {age}. Output JSON only."
        resp = json.dumps({"name": name, "age": age})
        if variant == "wrong_format":
            resp, gold = f"name: {name}, age: {age}", "false"
        elif variant == "missing_part":
            resp, gold = json.dumps({"name": name}), "false"
        elif variant == "extra_text":
            resp, gold = "Here you go: " + json.dumps({"name": name, "age": age}), "false"
        elif variant == "wrong_value":
            resp, gold = json.dumps({"name": name, "age": age + 1}), "false"
        else:
            variant = "ok"
    elif kind == "count_words":
        n = rng.randint(3, 6)
        words = ["Rain", "falls", "softly", "on", "quiet", "roofs", "tonight"]
        req = f"Write a sentence of exactly {n} words about rain."
        resp = " ".join(words[:n]) + "."
        if variant in ("near_miss", "wrong_value"):
            resp, gold = " ".join(words[:n + 1]) + ".", "false"
            strength = STRENGTH["inferred"] if variant == "near_miss" else 0.95
        else:
            variant = "ok"
    else:  # reference
        cap = rng.choice([("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Kenya", "Nairobi")])
        other = rng.choice(["Lyon", "Osaka", "Cusco", "Mombasa"])
        req = f"Reference: The capital of {cap[0]} is {cap[1]}. Request: Using only the reference, name the capital of {cap[0]}."
        resp = cap[1]
        if variant in ("wrong_value", "near_miss"):
            resp, gold = other, "false"
        elif variant == "unsupported":
            resp, gold = f"{cap[1]}, which has about 2 million residents.", "false"
            ops = ["unsupported"]
        else:
            variant = "ok"
    if variant != "ok":
        ops = ops or [variant]
    state = f"Request: {req} Response:{resp}" if rng.random() < 0.5 else f"Request: {req}\nResponse: {resp}"
    return Item("adequacy", state, rng.choice(ADEQ_Q), list(ADEQ_OPTS), gold, strength, ops, variant)


# ================================================================= gates ==
def _tokens(s: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]


def question_only_probe(items: List[Item]) -> float:
    """Naive Bayes on question text -> label, 2-fold; leakage if > 0.60."""
    if len(items) < 20:
        return 0.0
    half = len(items) // 2
    acc = 0
    for train, test in ((items[:half], items[half:]), (items[half:], items[:half])):
        counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        prior: collections.Counter = collections.Counter()
        for it in train:
            prior[it.label] += 1
            counts[it.label].update(_tokens(it.question))
        vocab = {t for c in counts.values() for t in c}
        for it in test:
            best, best_lp = None, -1e18
            for lab in prior:
                lp = math.log(prior[lab] / len(train))
                tot = sum(counts[lab].values()) + len(vocab)
                for t in _tokens(it.question):
                    lp += math.log((counts[lab][t] + 1) / tot)
                if lp > best_lp:
                    best, best_lp = lab, lp
            acc += best == it.label
    return acc / len(items)


def keyword_probe(items: List[Item]) -> float:
    """Pick the option whose id / description words appear most in the state."""
    hit = 0
    for it in items:
        st = set(_tokens(it.state))
        scores = {oid: len(st & set(_tokens(oid + " " + d))) for oid, d in it.options}
        top = max(scores.values())
        best = [o for o, s in scores.items() if s == top]
        hit += (it.label in best) / len(best)
    return hit / max(1, len(items))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="items per family")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data_standard/spines.jsonl")
    ap.add_argument("--max-question-probe", type=float, default=0.60)
    ap.add_argument("--max-keyword-probe", type=float, default=0.60)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    per_family: Dict[str, List[Item]] = {"extraction": [], "routing": [], "adequacy": []}
    gens = {"extraction": gen_extraction, "routing": gen_routing, "adequacy": gen_adequacy}
    seen = set()
    for fam, g in gens.items():
        # balance: cap any single label at 45% for choice families, 55% for noul
        cap = 0.55 if fam == "adequacy" else (0.30 if fam == "extraction" else 0.45)
        counts: collections.Counter = collections.Counter()
        tries = 0
        while len(per_family[fam]) < args.n and tries < args.n * 30:
            tries += 1
            it = g(rng)
            key = (it.state, it.question)
            if key in seen:
                continue
            if counts[it.label] + 1 > cap * args.n:
                continue
            seen.add(key)
            counts[it.label] += 1
            per_family[fam].append(it)
        print(f"{fam}: {len(per_family[fam]):,} items  labels={dict(counts.most_common())}")

    ok = True
    for fam, items in per_family.items():
        qp = question_only_probe(items)
        op_items = [i for i in items if i.ops]
        kp = keyword_probe(op_items)
        kp_all = keyword_probe(items)
        flag = "" if (qp <= args.max_question_probe and kp <= args.max_keyword_probe) else "  <-- GATE FAIL"
        ok = ok and not flag
        print(f"{fam}: question-only probe {qp:.3f} (<= {args.max_question_probe})  keyword probe operator-items {kp:.3f} "
              f"all {kp_all:.3f} (<= {args.max_keyword_probe}) n_op={len(op_items)}{flag}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        i = 0
        for fam, items in per_family.items():
            for it in items:
                f.write(json.dumps(it.row(i), ensure_ascii=False) + "\n")
                i += 1
    print(f"wrote {i:,} rows to {args.out}")
    if not ok:
        print("shortcut gate failed; do not train on this corpus", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

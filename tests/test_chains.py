"""Chain-of-options: span proposer, operator library, and one end-to-end run
per shipped chain through a scripted stub backend (no model weights).

The stub answers the route question with a fixed chain name and each bind
question with the candidate whose text contains a wanted substring, so the
test pins the deterministic part (proposer + executor + describe) and the
control flow (recursion guard, fallback on empty candidates / exec error),
not the model's judgement.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from von.chains import ops  # noqa: E402
from von.chains.runner import ChainRunner, load_chains  # noqa: E402
from von.chains.spans import by_kind, propose  # noqa: E402
from von.types import Choice, ChoiceAnswer, Noul, NoulAnswer  # noqa: E402

LIB = os.path.join(ROOT, "src", "von", "chains", "library")


# ----------------------------------------------------------------- spans --
def test_propose_datetimes_durations_zones_amounts():
    st = ("Notice received Friday March 6, 2026 at 18:10 New York time. The window lasts exactly 52 hours. "
          "Reply arrived March 11, 2026 at 13:30 New York local time. Fee EUR 1,234.50 or 8% of budget.")
    sp = propose(st)
    dts = by_kind(sp, "datetime")
    assert [d.value for d in dts] == [datetime(2026, 3, 6, 18, 10), datetime(2026, 3, 11, 13, 30)]
    assert dts[0].meta["zone"] == {"iana": "America/New_York"}
    assert [d.value for d in by_kind(sp, "duration")] == [{"n": 52.0, "unit": "hours"}]
    assert any(z.value.get("iana") == "America/New_York" for z in by_kind(sp, "timezone"))
    assert [a.value for a in by_kind(sp, "amount")] == [1234.5]
    assert [p.value for p in by_kind(sp, "percent")] == [8.0]


def test_propose_skips_years_and_ordinals_as_amounts():
    sp = propose("Clause 12 of the 2026 policy, item 7, serial 5540981.")
    assert by_kind(sp, "amount") == []


def test_propose_time_not_confused_with_decimal():
    sp = propose("Usage 402.10 on Sep 01 at 09:30.")
    assert [a.value for a in by_kind(sp, "amount")] == [402.1]
    assert by_kind(sp, "datetime")[0].value == datetime(datetime.now().year, 9, 1, 9, 30)


# ------------------------------------------------------------------- ops --
def _dt(text):
    return by_kind(propose(text), "datetime")[0]


def test_add_duration_hours_crosses_dst_in_real_time():
    start = _dt("March 6, 2026 at 18:10 New York time")
    dl = ops.add_duration(start, {"n": 52, "unit": "hours"})
    # 52 real hours across the 8 March spring-forward lands at 23:10 EDT, not 22:10.
    assert (dl.day, dl.hour, dl.minute, dl.tzname()) == (8, 23, 10, "EDT")


def test_add_months_clamps_to_month_end_and_leap_day():
    d = ops.add_duration(_dt("31 August 2026"), {"n": 18, "unit": "months"})
    assert (d.year, d.month, d.day) == (2028, 2, 29)


def test_elapsed_hours_between_zones():
    a = ops.in_zone(_dt("3 Sep 2026, 18:30"), {"iana": "America/Chicago"})
    b = ops.in_zone(_dt("4 Sep 2026, 08:55"), {"iana": "Europe/Madrid"})
    assert ops.elapsed_hours_abs(a, b) == pytest.approx(7.0 + 25 / 60, abs=1e-6)


def test_proration_inclusive_days_and_leap_year():
    n = ops.days_inclusive(_dt("February 28, 2028"), _dt("March 2, 2028"))
    assert n == 4  # 28, 29 (leap), 1, 2
    assert ops.days_in_year(_dt("1 January 2028")) == 366
    assert ops.prorate(366.0, n, 366) == pytest.approx(4.0)


def test_table_series_cumsum_first_at_or_above():
    st = ("Sep 01 | usage 402.10 | credits 0.00\n"
          "Sep 02 | usage 400.00 | credits 10.55\n"
          "Sep 03 | usage 415.00 | credits 0.00\n")
    u = ops.table_series(st, "usage")
    c = ops.table_series(st, "credits")
    run = ops.cumsum(ops.series_sub(u, c))
    assert [round(v, 2) for _, v in run] == [402.10, 791.55, 1206.55]
    assert ops.first_at_or_above(run, ops.pct_of(1000, 80)) == "Sep 03"
    assert ops.first_at_or_above(run, ops.pct_of(1000, 79)) == "Sep 02"
    assert ops.first_at_or_above(run, 700) == "Sep 02"
    assert ops.first_at_or_above(run, 5000) is None


# ---------------------------------------------------------------- runner --
class StubBackend:
    """Scripted Choice answers; records every sub-question it was asked."""

    def __init__(self, route: str, binds: Dict[str, str]):
        self.route, self.binds, self.asked = route, binds, []

    def evaluate_choice(self, _id, state, q: Choice):
        self.asked.append(q.instructions)
        keys = list(q.criteria)
        if "Which computation" in q.instructions:
            pick = self.route
        else:
            want = next((v for k, v in self.binds.items() if k in q.instructions), None)
            pick = next((k for k, d in q.criteria.items() if want and want in d), keys[0])
        p = {k: (0.7 if k == pick else 0.3 / max(1, len(keys) - 1)) for k in keys}
        return ChoiceAnswer(choice=pick, probabilities=p, confidence=0.7)

    def evaluate_noul(self, _id, state, q: Noul):
        self.asked.append(("noul", state))
        return NoulAnswer(noul=0.9 if "after the deadline" in state and "-" not in state.split("which is")[-1][:6] else 0.1)


def _runner(route, binds):
    b = StubBackend(route, binds)
    return ChainRunner(b, LIB), b


def test_library_loads_all_five_chains():
    names = sorted(c.name for c in load_chains(LIB))
    assert names == ["cumulative_limit", "deadline_tz", "elapsed_window", "month_window", "proration"]


def test_deadline_tz_end_to_end():
    st = ("Notice received Friday March 6, 2026 at 18:10 New York time. The reply window is 52 hours. "
          "Reply arrived March 11, 2026 at 13:30 New York local time.")
    r, b = _runner("deadline_tz", {"start counting": "March 6", "timeliness": "March 11", "window": "52 hours"})
    ans, tr = r.run(st, Noul(type="noul", instructions="Was the reply late?"))
    assert tr.chain == "deadline_tz" and tr.fallback is None
    assert tr.bindings["start"].startswith("March 6") and tr.bindings["event"].startswith("March 11")
    assert "Sunday 08 March 2026 23:10 EDT" in tr.description
    assert "62.33 hours after the deadline" in tr.description
    assert isinstance(ans, NoulAnswer) and ans.noul > 0.5


def test_month_window_end_to_end():
    st = "Warranty starts 31 August 2026 for 18 months. Claim filed 1 March 2028."
    r, _ = _runner("month_window", {"start from": "31 August", "long": "18 months", "claim": "1 March"})
    _, tr = r.run(st, Choice(type="choice", instructions="Is the claim covered?", criteria={"covered": "yes", "expired": "no"}))
    assert tr.fallback is None
    assert "end of Tuesday 29 February 2028" in tr.description
    assert "1 days after the end" in tr.description  # 1 March is one day past the clamped end


def test_proration_matches_value_option_directly():
    st = "Annual allowance 1,098 credits. Active from February 28, 2028 through March 2, 2028 inclusive."
    r, b = _runner("proration", {"prorated": "1,098", "start on": "February 28", "end on": "March 2"})
    ans, tr = r.run(st, Choice(type="choice", instructions="How many credits?",
                                criteria={"9_credits": "9 credits", "12_credits": "12 credits", "4_credits": "4 credits"}))
    assert tr.fallback is None and tr.values["active_days"] == 4 and tr.values["year_days"] == 366
    assert tr.values["prorated"] == pytest.approx(12.0)
    assert ans.choice == "12_credits" and ans.probabilities["12_credits"] == pytest.approx(0.9)
    # value mode answered without a final model read
    assert not any(isinstance(a, tuple) for a in b.asked)


def test_cumulative_limit_end_to_end():
    st = ("Budget USD 1,000.00; alert at 80%.\n"
          "Sep 01 | usage 402.10 | credits 0.00\nSep 02 | usage 400.00 | credits 10.55\nSep 03 | usage 415.00 | credits 0.00\n")
    r, _ = _runner("cumulative_limit", {"limit": "1,000", "percentage": "80%"})
    _, tr = r.run(st, Choice(type="choice", instructions="On which day?", criteria={"sep_01": "Sep 01", "sep_02": "Sep 02", "sep_03": "Sep 03"}))
    assert tr.fallback is None
    assert tr.values["threshold"] == pytest.approx(800.0) and tr.values["first_row"] == "Sep 03"


def test_elapsed_window_end_to_end():
    st = "Previous dose 3 Sep 2026, 18:30 Chicago time. Next dose 4 Sep 2026, 08:55 Madrid time. Minimum gap 12 hours."
    r, _ = _runner("elapsed_window", {"first": "18:30", "second": "08:55", "earlier moment": "Chicago", "later moment": "Madrid"})
    _, tr = r.run(st, Noul(type="noul", instructions="Is the gap at least 12 hours?"))
    assert tr.fallback is None
    assert tr.values["elapsed_h"] == pytest.approx(7.4167, abs=1e-3)


def test_route_none_falls_back_and_gate_skips_non_numeric():
    r, b = _runner("none", {})
    ans, tr = r.run("Deadline 3 Sep 2026 18:30, reply 4 Sep 08:55.", Noul(type="noul", instructions="Late?"))
    assert ans is None and tr.fallback == "route:none"
    ans, tr = r.run("No numbers here at all.", Noul(type="noul", instructions="Late?"))
    assert ans is None and tr.fallback == "gate" and len(b.asked) == 1


def test_missing_required_slot_falls_back():
    r, _ = _runner("deadline_tz", {})
    ans, tr = r.run("Started 3 Sep 2026 18:30 and 4 Sep 2026 08:55, no duration given.", Noul(type="noul", instructions="?"))
    assert ans is None and tr.fallback.startswith("bind:duration")


def test_recursion_guard_marks_active_during_subdecisions():
    seen = []
    r, b = _runner("deadline_tz", {})
    orig = b.evaluate_choice

    def spy(*a, **k):
        seen.append(r.active())
        return orig(*a, **k)

    b.evaluate_choice = spy
    r.run("Start 3 Sep 2026 18:30, window 5 hours, event 4 Sep 2026 08:55.", Noul(type="noul", instructions="?"))
    assert seen and all(seen) and not r.active()

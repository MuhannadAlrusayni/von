"""Noul decision rule: committed P(yes) outside the 0.2..0.8 abstention band."""
import os

import pytest

from von.backends.option_marker_backend import _noul_decide


@pytest.mark.parametrize("p", [0.0, 0.1, 0.3, 0.49, 0.5, 0.51, 0.7, 0.9, 1.0])
def test_band_never_lands_in_abstention_band(p):
    out = _noul_decide(p, 0.8, 0.1)
    assert out <= 0.2 or out >= 0.8
    assert (out >= 0.5) == (p >= 0.5), "argmax must be preserved"


def test_band_is_monotone_and_bounded():
    ps = [i / 100 for i in range(101)]
    outs = [_noul_decide(p, 0.8, 0.1) for p in ps]
    assert all(a <= b for a, b in zip(outs, outs[1:]))
    assert 0.0 <= min(outs) and max(outs) <= 1.0
    assert _noul_decide(0.5, 0.8, 0.1) == pytest.approx(0.8)
    assert _noul_decide(1.0, 0.8, 0.1) == pytest.approx(0.85)
    assert _noul_decide(0.0, 0.8, 0.1) == pytest.approx(0.15)


def test_identity_parameters_are_a_noop():
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert _noul_decide(p, 0.5, 1.0) == pytest.approx(p)


def test_backend_applies_band_by_default_and_raw_disables_it(monkeypatch):
    from von.backends.option_marker_backend import OptionMarkerBackend
    from von.types import NoulAnswer

    monkeypatch.delenv("VON_NOUL_DECISION", raising=False)
    b = OptionMarkerBackend.__new__(OptionMarkerBackend)
    b.noul_decision, b.noul_band_edge, b.noul_band_slope = "band", 0.8, 0.1
    assert b._commit_noul(NoulAnswer(noul=0.62)).noul == pytest.approx(0.812)
    b.noul_decision = "raw"
    assert b._commit_noul(NoulAnswer(noul=0.62)).noul == pytest.approx(0.62)


def test_env_validation(monkeypatch):
    from von.backends.option_marker_backend import OptionMarkerBackend

    monkeypatch.setenv("VON_NOUL_DECISION", "sometimes")
    with pytest.raises(ValueError, match="VON_NOUL_DECISION"):
        OptionMarkerBackend(checkpoint_dir=os.path.join("checkpoints", "von-1.2"), device="cpu")

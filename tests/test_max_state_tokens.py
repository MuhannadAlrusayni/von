"""--max-state-tokens / VON_MAX_STATE_TOKENS: middle truncation of long states.

Contract:
  * default (8192) is a no-op for every state that fits the encoder window;
  * a state past the limit is middle-truncated (head + " ... " + tail) so the
    question and every option marker survive intact;
  * a state that would overflow the 8192 window no longer raises, it truncates;
  * the response carries `truncation`, and the HTTP layer adds X-Von-Truncated.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from von.backends.option_marker_backend import OptionMarkerBackend  # noqa: E402

CHECKPOINT = "checkpoints/von-1.2"
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINT, "option_marker.pt")),
    reason="von-1.2 checkpoint not present",
)

QUESTION = {"type": "choice", "instructions": "Which department handles this?",
            "criteria": {"billing": "Invoices and payments.", "tech": "Bugs and outages.", "sales": "Quotes and demos."}}


@pytest.fixture(scope="module")
def backend():
    b = OptionMarkerBackend(checkpoint_dir=CHECKPOINT, device="cpu")
    b._get_model()
    return b


def _long_state(n_words: int) -> str:
    head = "POLICY HEADER: refunds within 30 days. "
    body = " ".join(f"filler{i}" for i in range(n_words))
    tail = " LEDGER: charge 42.00 on 2026-09-01; refund requested 2026-09-20."
    return head + body + tail


def test_default_is_noop(backend):
    backend.max_state_tokens = 8192
    state = "Short state about an invoice."
    assert backend._fit_state(backend._get_model(), state, QUESTION["instructions"], list(QUESTION["criteria"].values())) == state
    res = backend.evaluate(state=state, questions={"q": QUESTION})
    assert res.truncation is None


def test_middle_truncation_keeps_head_tail_and_markers(backend):
    backend.max_state_tokens = 256
    model = backend._get_model()
    state = _long_state(2000)
    fitted = backend._fit_state(model, state, QUESTION["instructions"], list(QUESTION["criteria"].values()))
    assert fitted != state
    assert fitted.startswith("POLICY HEADER")
    assert fitted.rstrip().endswith("2026-09-20.")
    assert " ... " in fitted
    tok = model.tokenizer
    assert len(tok(fitted, add_special_tokens=False)["input_ids"]) <= 256 + 4
    packed = model.pack_sequence(fitted, QUESTION["instructions"], list(QUESTION["criteria"].values()))
    ids = tok(packed)["input_ids"]
    assert ids.count(model.mask_token_id) == len(QUESTION["criteria"])
    res = backend.evaluate(state=state, questions={"q": QUESTION})
    assert res.truncation and res.truncation["strategy"] == "middle"
    assert res.truncation["kept_tokens"] == 256
    assert res.truncation["state_tokens"] > 256
    assert res.answers["q"].choice in QUESTION["criteria"]
    backend.max_state_tokens = 8192


def test_window_overflow_truncates_instead_of_raising(backend):
    backend.max_state_tokens = 8192
    state = _long_state(12000)  # ~>8192 tokens
    res = backend.evaluate(state=state, questions={"q": QUESTION})
    assert res.truncation is not None
    assert res.truncation["kept_tokens"] < 8192
    assert res.answers["q"].choice in QUESTION["criteria"]


def test_http_header_on_truncation(backend, monkeypatch):
    from fastapi.testclient import TestClient
    from von import server as srv
    from von.engine import VonEngine

    class _Eng:
        def evaluate(self, state, questions, model=None):
            return backend.evaluate(state=state, questions=questions)

    monkeypatch.setattr(VonEngine, "get_instance", staticmethod(lambda: _Eng()))
    backend.max_state_tokens = 128
    client = TestClient(srv.app)
    r = client.post("/v1/systemone", json={"state": _long_state(1500), "questions": {"q": QUESTION}})
    assert r.status_code == 200, r.text
    assert r.headers.get("X-Von-Truncated", "").startswith("state; tokens=")
    assert "kept=128" in r.headers["X-Von-Truncated"]
    assert r.json()["truncation"]["kept_tokens"] == 128
    r2 = client.post("/v1/systemone", json={"state": "tiny", "questions": {"q": QUESTION}})
    assert "X-Von-Truncated" not in r2.headers
    assert r2.json().get("truncation") is None
    backend.max_state_tokens = 8192

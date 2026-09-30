"""von calibrate: label parsing, NLL fit, CV winner, file output, backend pickup."""
import json
import math
import os
import random

import pytest

from von import calibrate as C


def _rec(logits, gold, tokens=100, k=None):
    k = k or len(logits)
    p = [math.exp(x) for x in logits]
    p = [x / sum(p) for x in p]
    ent = -sum(x * math.log(max(x, 1e-12)) for x in p) / math.log(len(p))
    return {"logits": logits, "gold": gold, "type": "choice",
            "feats": {"bias": 1.0, "entropy": ent, "log_tokens": math.log10(tokens) / 4, "n_options": k / 8}}


def _synthetic(n=200, seed=0, true_t=3.0):
    """Overconfident logits: gold is right 70% of the time but argmax says ~99%."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = 4
        gold = rng.randrange(k)
        right = rng.random() < 0.7
        logits = [rng.gauss(0, 0.5) for _ in range(k)]
        top = gold if right else rng.choice([i for i in range(k) if i != gold])
        logits[top] += 6.0
        out.append(_rec(logits, gold, tokens=rng.choice([20, 200, 2000])))
    return out


def test_parse_rows_all_types_and_shorthand():
    s, q, g = C.parse_row({"state": "x", "question": {"type": "noul", "instructions": "i", "criteria": {"true": "t", "false": "f"}}, "gold": "no"}, 1)
    assert C.gold_index(q, g, 1) == 1
    s, q, g = C.parse_row({"state": {"a": 1}, "instructions": "i", "choices": {"a": "A", "b": "B"}, "gold": "b"}, 2)
    assert q["type"] == "choice" and s == '{"a": 1}' and C.gold_index(q, g, 2) == 1
    s, q, g = C.parse_row({"state": "x", "question": {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}, "gold": 2}, 3)
    assert C.gold_index(q, g, 3) == 2
    with pytest.raises(ValueError, match="not in criteria"):
        C.gold_index({"type": "choice", "criteria": {"a": ""}}, "z", 4)
    with pytest.raises(ValueError, match="outside"):
        C.gold_index({"type": "score", "criteria": ["a", "b"]}, 5, 5)


def test_scalar_fit_reduces_nll_and_ece_on_overconfident_logits():
    recs = _synthetic()
    raw = C.nll({"bias": 1.0}, recs)
    scalar = C._fit(recs, ("bias",))
    assert scalar["bias"] > 1.5
    assert C.nll(scalar, recs) < raw - 0.2
    assert C.ece(scalar, recs) < C.ece(None, recs)


def test_fit_map_picks_scalar_when_features_carry_no_signal():
    fitted = C.fit_map(_synthetic(), folds=4, log=lambda m: None)
    rep = fitted["report"]
    assert rep["winner"] in ("scalar", "map")
    assert set(fitted["calibration_map"]) == set(C.FEATURES) | {"lo", "hi"}
    assert rep["cv_nll"]["scalar"] < rep["in_sample_nll"]["raw_T1"]


def test_write_calibration_keeps_shipped_fields(tmp_path):
    fitted = {"calibration_map": {"bias": 2.5, "entropy": 0, "log_tokens": 0, "n_options": 0, "lo": 0.3, "hi": 12},
              "report": {"n": 10, "folds": 5, "winner": "scalar"}}
    shipped = {"independent_options": True, "temperature": 2.2, "noul_zero_shot_prior": {"a": -0.5, "b": 0.3},
               "calibration_map": {"bias": 9}, "calibration_ece": {"hard": 0.1}}
    out = tmp_path / "marker_calibration.json"
    C.write_calibration(str(out), str(tmp_path), fitted, "labels.jsonl", shipped=shipped)
    d = json.loads(out.read_text())
    assert d["independent_options"] is True and d["noul_zero_shot_prior"] == {"a": -0.5, "b": 0.3}
    assert d["calibration_map"]["bias"] == 2.5 and "calibration_ece" not in d
    assert "von calibrate" in d["calibration_fitted_on"]


@pytest.mark.skipif(not os.path.exists("checkpoints/von-1.2/option_marker.pt"), reason="needs local weights")
def test_end_to_end_backend_reads_refit_map(tmp_path):
    rows = [
        {"state": "Invalid API key. Please run /login", "question": {"type": "choice", "instructions": "Why did it fail?",
         "criteria": {"auth": "Login or credentials problem", "code_error": "A bug or exception in the code"}}, "gold": "auth"},
        {"state": "TypeError: cannot read properties of undefined", "question": {"type": "choice", "instructions": "Why did it fail?",
         "criteria": {"auth": "Login or credentials problem", "code_error": "A bug or exception in the code"}}, "gold": "code_error"},
        {"state": "Ran npm test: 3 pass, 0 fail", "question": {"type": "noul", "instructions": "Was a test run?"}, "gold": "yes"},
        {"state": "Should work now.", "question": {"type": "noul", "instructions": "Was a test run?"}, "gold": "no"},
    ] * 3
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in rows))
    ck = tmp_path / "ckpt"
    ck.mkdir()
    import shutil
    for f in os.listdir("checkpoints/von-1.2"):
        src = os.path.abspath(os.path.join("checkpoints/von-1.2", f))
        if f == "marker_calibration.json":
            continue  # copied below; never link it, the fitter writes to this path
        os.symlink(src, ck / f)
    with pytest.raises(ValueError, match="independent_options"):
        C.run(str(labels), str(ck / "marker_calibration.json"), str(ck), device="cpu", folds=3, log=lambda m: None)
    shutil.copy("checkpoints/von-1.2/marker_calibration.json", ck / "marker_calibration.json")
    before = json.load(open("checkpoints/von-1.2/marker_calibration.json"))
    written = C.run(str(labels), str(ck / "marker_calibration.json"), str(ck), device="cpu", folds=3, log=lambda m: None)
    assert written["independent_options"] is True
    from von.backends.option_marker_backend import OptionMarkerBackend
    b = OptionMarkerBackend(checkpoint_dir=str(ck), device="cpu")
    b._get_model()
    assert b._calib_map["bias"] == pytest.approx(written["calibration_map"]["bias"])
    assert json.load(open("checkpoints/von-1.2/marker_calibration.json")) == before, "repo checkpoint must be untouched"


def test_write_calibration_replaces_symlink_instead_of_writing_through(tmp_path):
    target = tmp_path / "shipped.json"
    target.write_text('{"independent_options": true, "calibration_map": {"bias": 9}}')
    link = tmp_path / "marker_calibration.json"
    link.symlink_to(target)
    fitted = {"calibration_map": {"bias": 2.0, "lo": 0.3, "hi": 12}, "report": {"n": 1, "folds": 2, "winner": "scalar"}}
    C.write_calibration(str(link), str(tmp_path), fitted, "l.jsonl", shipped=json.loads(target.read_text()))
    assert not link.is_symlink() and json.loads(link.read_text())["calibration_map"]["bias"] == 2.0
    assert json.loads(target.read_text())["calibration_map"]["bias"] == 9

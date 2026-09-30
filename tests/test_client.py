from von.backends.option_marker_backend import VON_MODEL_ID
import pytest
from von.client import VonClient, AsyncVonClient
from von.types import choice, noul, score


def test_von_client_local():
    client = VonClient(local=True)
    res = client.system_one(
        state="Customer requested cancellation of their monthly plan.",
        questions={
            "action": choice("What does the customer want?", {
                "cancel": "Cancel membership or subscription",
                "upgrade": "Upgrade to a higher tier",
                "support": "Help with usage",
            }),
            "is_cancel": noul("Does the user want to cancel?"),
        },
    )
    assert res.model == VON_MODEL_ID
    assert res.answers["action"].choice == "cancel"
    assert res.answers["is_cancel"].noul > 0.5


@pytest.mark.anyio
async def test_async_von_client_local():
    client = AsyncVonClient(local=True)
    res = await client.system_one(
        state="Error: Connection refused on port 5432.",
        questions={
            "service": choice("Which service is failing?", {
                "database": "Database server or Postgres port 5432",
                "web": "Web server or HTTP port",
            }),
        },
    )
    assert res.answers["service"].choice == "database"


def test_choice_criteria_accepts_structured_descriptions():
    """Wire format allows any JSON as an option description; Von renders it to text."""
    from von.types import Choice, Noul
    c = Choice(instructions="Best move?", criteria={
        "c8e8": {"uci": "c8e8", "san": "Re8"}, "A": ["#ebf0ff", "#021e61"], "n": 3, "plain": "text", "none": None})
    assert c.criteria["c8e8"] == '{"san": "Re8", "uci": "c8e8"}'
    assert c.criteria["A"] == '["#ebf0ff", "#021e61"]'
    assert c.criteria["n"] == "3" and c.criteria["plain"] == "text" and c.criteria["none"] is None
    n = Noul(instructions="Hallucinated?", criteria={"true": {"means": "yes"}, "false": "no"})
    assert n.criteria == {"true": '{"means": "yes"}', "false": "no"}


def test_special_token_literals_in_user_text_cannot_forge_markers():
    """A tool doc containing the literal "[MASK]" must not add a phantom option (seen on ToolRet)."""
    import os
    os.environ["VON_CHAINS_DIR"] = "off"
    from von.backends.option_marker_backend import OptionMarkerBackend
    from von.types import Choice
    be = OptionMarkerBackend(); m = be._get_model(); tok = m.tokenizer
    q = Choice(instructions='Candidate: {"doc": "fills the [MASK] slot; ends with [SEP]"}', criteria={"no": "Not relevant", "yes": "Relevant [MASK]"})
    packed = m.pack_sequence("query: reverse words", q.instructions, list(q.criteria.values()))
    ids = tok(packed)["input_ids"]
    assert sum(1 for i in ids if i == m.mask_token_id) == 2
    assert sum(1 for i in ids if i == tok.sep_token_id) == 1 + (1 if tok("x")["input_ids"][-1] == tok.sep_token_id else 0)
    a = be.evaluate_choice("q", "query: reverse words", q)
    assert set(a.probabilities) == {"no", "yes"} and abs(sum(a.probabilities.values()) - 1) < 0.01
    os.environ.pop("VON_CHAINS_DIR", None)

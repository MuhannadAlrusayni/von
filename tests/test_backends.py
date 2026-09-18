import pytest
import von


@pytest.fixture(autouse=True)
def reset_backend():
    yield
    von.set_backend("needle")


def test_needle_backend_explicit():
    von.set_backend("needle")
    res = von.decide("Billing error on checkout invoice", choices=["billing", "technical"])
    assert res.choice == "billing"
    assert res.confidence > 0.0


def test_laya_backend():
    von.set_backend("laya")
    res = von.decide("Customer requests refund for duplicate charge on invoice #100", choices=["billing", "technical"])
    assert res.choice == "billing"
    assert "billing" in res.probabilities
    assert res.confidence > 0.0


def test_berta_modern_backend():
    von.set_backend("berta-modern")
    res = von.decide("Server CPU temperature reached 105 degrees Celsius", choices=["hardware_alert", "billing"])
    assert res.choice == "hardware_alert"
    assert "hardware_alert" in res.probabilities
    assert res.confidence > 0.0


def test_berta_v3_backend():
    von.set_backend("berta-v3")
    res = von.decide("Customer wants to reset forgotten account password", choices=["account_access", "billing"])
    assert res.choice == "account_access"
    assert "account_access" in res.probabilities
    assert res.confidence > 0.0

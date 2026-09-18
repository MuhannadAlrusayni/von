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


def test_modernbert_backend():
    von.set_backend("modernbert")
    res = von.decide("Customer wants to reset forgotten password", choices=["account_access", "billing"])
    assert res.choice == "account_access"
    assert "account_access" in res.probabilities


def test_qwen_backend():
    von.set_backend("qwen0.5b")
    res = von.decide("Server CPU temperature is 105 degrees Celsius", choices=["hardware_alert", "billing"])
    assert res.choice == "hardware_alert"
    assert "hardware_alert" in res.probabilities

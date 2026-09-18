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

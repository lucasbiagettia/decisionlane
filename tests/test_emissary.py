"""Emissary adapter against a simulated transport (fixtures are not provider observations)."""

from __future__ import annotations

import copy

import pytest
import requests

from decisionlane import Annotator, BackendError, ResponseError
from decisionlane.backends import Emissary
from fakes import FakeResponse, FakeSession, fixture

FIXTURE = fixture("emissary_routing.json")
CASES = FIXTURE["cases"]
SECRET = "offline-secret-key"
TEXT = "The central bank raised interest rates"


def experiment(body=None):
    return FakeResponse(body if body is not None else FIXTURE["experiment_response"])


def classified(body=None):
    return FakeResponse(body if body is not None else FIXTURE["classification_response"])


def annotator(session, **kwargs):
    return Annotator(backend=Emissary(api_key=SECRET, session=session, **kwargs), cases=CASES)


def body_with(**changes):
    body = copy.deepcopy(FIXTURE["classification_response"])
    body.update(changes)
    return body


def test_creates_one_routing_experiment_and_reuses_it():
    session = FakeSession(experiment(), classified(), classified(), classified())
    ann = annotator(session, experiment_name="unit-test")
    first = ann(TEXT)
    rest = ann.batch(["second", "third"])

    create, *classify = session.calls
    assert create["url"] == "https://api.withemissary.com/v1/experiments"
    assert create["json"] == {
        "name": "unit-test",
        "mode": "routing",
        "classes": [{"name": k, "description": v} for k, v in CASES.items()],
    }
    assert create["headers"]["X-API-Key"] == SECRET
    assert all(c["url"] == "https://api.withemissary.com/v1/classification" for c in classify)
    assert [c["json"] for c in classify] == [
        {"model": "ex-test/0.0.0", "input": text, "data_format": "probs"}
        for text in (TEXT, "second", "third")
    ]
    assert all(c["timeout"] == 120.0 and c["allow_redirects"] is False for c in session.calls)
    assert [d.label for d in [first, *rest]] == ["finance"] * 3


def test_normalized_decision():
    decision = annotator(FakeSession(experiment(), classified()))(TEXT)
    assert decision.label == "finance"
    assert decision.probabilities == {"sports": 0.09, "finance": 0.86, "cooking": 0.05}
    assert decision.selected_probability == 0.86
    assert decision.backend == "emissary"
    assert decision.model == "ex-test/0.0.0"
    assert decision.request_id == "classify-test-123"
    assert decision.provider_confidence is None and decision.explanation is None
    assert decision.latency_ms >= 0


def test_default_experiment_name_is_generated():
    session = FakeSession(experiment(), classified())
    annotator(session)(TEXT)
    assert session.calls[0]["json"]["name"].startswith("decisionlane-")


def test_optional_response_fields_may_be_absent():
    body = body_with()
    del body["model"], body["id"]
    decision = annotator(FakeSession(experiment(), classified(body)))(TEXT)
    assert decision.model == "ex-test/0.0.0" and decision.request_id is None


def test_response_model_must_match_prepared_experiment_version():
    with pytest.raises(ResponseError, match="differs"):
        annotator(FakeSession(experiment(), classified(body_with(model="ex-other/1.0.0"))))(TEXT)


def test_tie_policy_first_tied_label_in_response_order():
    probs = {"cooking": 0.4, "finance": 0.4, "sports": 0.2}
    body = body_with(data=[{"index": 0, "probs": probs}])
    decision = annotator(FakeSession(experiment(), classified(body)))(TEXT)
    assert decision.label == "cooking"
    assert list(decision.probabilities) == list(CASES)  # configured order


@pytest.mark.parametrize("probs", [
    {"sports": 0.8, "finance": 0.8, "cooking": 0.05},
    {"sports": 0.5, "finance": 0.5},
    {"sports": 0.5, "finance": 0.3, "cooking": 0.1, "extra": 0.1},
    {"sports": 1.5, "finance": -0.5, "cooking": 0.0},
    {"sports": float("inf"), "finance": 0.3, "cooking": 0.1},
    {"sports": "0.5", "finance": 0.4, "cooking": 0.1},
    {"sports": True, "finance": 0.0, "cooking": 0.0},
    {},
    None,
])
def test_invalid_probabilities_are_rejected_not_renormalized(probs):
    body = body_with(data=[{"index": 0, "probs": probs}])
    session = FakeSession(experiment(), classified(body))
    with pytest.raises(ResponseError):
        annotator(session)(TEXT)
    assert len(session.calls) == 2


@pytest.mark.parametrize("data", [[], None, ["not an object"]])
def test_missing_data_item(data):
    with pytest.raises(ResponseError):
        annotator(FakeSession(experiment(), classified(body_with(data=data))))(TEXT)


@pytest.mark.parametrize("body", [
    {"latest_version": "0.0.0"},
    {"id": "ex-test"},
    {"id": "ex-test", "latest_version": "latest"},
    {"id": "ex-test", "latest_version": "a/b"},
    {"id": "", "latest_version": "0.0.0"},
])
def test_invalid_experiment_response_leaves_backend_unprepared(body):
    session = FakeSession(experiment(body), experiment(), classified())
    ann = annotator(session)
    with pytest.raises(ResponseError):
        ann(TEXT)
    assert ann(TEXT).label == "finance"  # next call prepares again
    assert [c["url"].rsplit("/", 1)[-1] for c in session.calls] == [
        "experiments", "experiments", "classification"]


def test_failed_experiment_creation_is_not_retried_automatically():
    session = FakeSession(requests.Timeout("slow"))
    with pytest.raises(BackendError, match="timed out"):
        annotator(session)(TEXT)
    assert len(session.calls) == 1


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_http_errors_are_not_retried_and_leak_nothing(status):
    leaky = FakeResponse({"detail": f"{TEXT} {SECRET}"}, status_code=status)
    session = FakeSession(experiment(), leaky)
    with pytest.raises(BackendError) as info:
        annotator(session)(TEXT)
    assert info.value.status_code == status
    assert len(session.calls) == 2
    assert SECRET not in str(info.value) and TEXT not in str(info.value)


def test_transport_error_is_sanitized():
    session = FakeSession(experiment(), requests.ConnectionError(f"X-API-Key: {SECRET} {TEXT}"))
    with pytest.raises(BackendError, match="transport failure") as info:
        annotator(session)(TEXT)
    assert SECRET not in str(info.value) and TEXT not in str(info.value)
    assert info.value.__cause__ is None


def test_key_from_environment_and_missing_key(monkeypatch):
    monkeypatch.setenv("EMISSARY_API_KEY", "env-key")
    session = FakeSession(experiment(), classified())
    Annotator(backend=Emissary(session=session), cases=CASES)(TEXT)
    assert session.calls[0]["headers"]["X-API-Key"] == "env-key"

    monkeypatch.delenv("EMISSARY_API_KEY")
    empty = FakeSession()
    with pytest.raises(ValueError, match="EMISSARY_API_KEY"):
        Annotator(backend=Emissary(session=empty), cases=CASES)(TEXT)
    assert empty.calls == []


def test_close_and_repr():
    session = FakeSession()
    backend = Emissary(api_key=SECRET, session=session)
    backend.close()
    assert session.closed is False
    assert SECRET not in repr(backend)

    owned = Emissary(api_key=SECRET)
    owned._session = FakeSession()
    fake = owned._session
    owned.close()
    assert fake.closed is True and owned._session is None


@pytest.mark.parametrize("kwargs", [{"experiment_name": " "}, {"timeout": -1}])
def test_invalid_adapter_configuration(kwargs):
    with pytest.raises(ValueError):
        Emissary(**kwargs)

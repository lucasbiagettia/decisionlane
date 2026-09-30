"""Jev adapter against a simulated transport (fixtures are not provider observations)."""

from __future__ import annotations

import copy

import pytest
import requests

from decisionlane import Annotator, BackendError, ResponseError
from decisionlane.backends import Jev
from fakes import FakeResponse, FakeSession, fixture

FIXTURE = fixture("jev_choice.json")
CASES = FIXTURE["request"]["questions"]["classification"]["criteria"]
TEXT = FIXTURE["request"]["state"]
SECRET = "offline-secret-token"


def ok(body=None):
    return FakeResponse(body if body is not None else FIXTURE["response"],
                        headers={"x-request-id": "req-1"})


def annotator(session, **kwargs):
    return Annotator(backend=Jev(api_key=SECRET, session=session, **kwargs), cases=CASES)


def test_payload_and_normalized_decision():
    session = FakeSession(ok())
    decision = annotator(session)(TEXT)
    [call] = session.calls
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["json"] == FIXTURE["request"]
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["timeout"] == 30.0 and call["allow_redirects"] is False
    assert decision.label == "Sports"
    assert decision.probabilities == {"World": 0.25, "Sports": 0.75}
    assert list(decision.probabilities) == ["World", "Sports"]  # configured order
    assert decision.selected_probability == 0.75
    assert decision.provider_confidence == 0.42  # native, kept apart from the probability
    assert decision.backend == "jev" and decision.model == "jev-1.13.0"
    assert decision.request_id == "req-1"
    assert decision.explanation is None
    assert decision.latency_ms >= 0


def test_resolved_model_is_reported_not_the_requested_one():
    body = copy.deepcopy(FIXTURE["response"])
    body["model"] = "jev-1.13.2"
    decision = annotator(FakeSession(ok(body)), model="jev-1.13")(TEXT)
    assert decision.model == "jev-1.13.2"


def test_batch_reuses_prepared_state_and_session():
    session = FakeSession(ok(), ok(), ok())
    decisions = annotator(session).batch(["one", "two", "three"])
    assert [c["json"]["state"] for c in session.calls] == ["one", "two", "three"]
    assert len(decisions) == 3


def test_optional_fields_may_be_absent():
    body = copy.deepcopy(FIXTURE["response"])
    body.pop("usage")
    body["answers"]["classification"].pop("confidence")
    decision = Annotator(backend=Jev(api_key=SECRET, session=FakeSession(FakeResponse(body))),
                         cases=CASES)(TEXT)
    assert decision.provider_confidence is None and decision.request_id is None
    assert decision.selected_probability == 0.75


def test_exact_tie_keeps_provider_choice():
    body = copy.deepcopy(FIXTURE["response"])
    body["answers"]["classification"]["probabilities"] = {"Sports": 0.5, "World": 0.5}
    assert annotator(FakeSession(ok(body)))(TEXT).label == "Sports"


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop("answers"),
    lambda b: b.pop("model"),
    lambda b: b["answers"]["classification"].update(type="score"),
    lambda b: b["answers"]["classification"].pop("probabilities"),
    lambda b: b["answers"]["classification"].update(choice="Other"),
    lambda b: b["answers"]["classification"].update(choice="World"),
    lambda b: b["answers"]["classification"].update(probabilities={"Sports": 1}),
    lambda b: b["answers"]["classification"].update(probabilities={"World": .2, "Sports": .7}),
    lambda b: b["answers"]["classification"]["probabilities"].update(World=float("nan")),
    lambda b: b["answers"]["classification"]["probabilities"].update(World=-.1),
    lambda b: b["answers"]["classification"]["probabilities"].update(World=True),
    lambda b: b["answers"]["classification"]["probabilities"].update(World="0.25"),
    lambda b: b["answers"]["classification"].update(confidence=float("inf")),
])
def test_invalid_responses_fail_without_retry(mutation):
    body = copy.deepcopy(FIXTURE["response"])
    mutation(body)
    session = FakeSession(ok(body))
    with pytest.raises(ResponseError):
        annotator(session)(TEXT)
    assert len(session.calls) == 1


def test_non_json_response():
    with pytest.raises(ResponseError, match="non-JSON"):
        annotator(FakeSession(FakeResponse("<html>")))(TEXT)


@pytest.mark.parametrize("status", [301, 401, 429, 500, 503])
def test_http_errors_are_not_retried_and_leak_nothing(status):
    body = {"error": f"echo {TEXT} {SECRET}"}
    session = FakeSession(FakeResponse(body, status_code=status))
    with pytest.raises(BackendError) as info:
        annotator(session)(TEXT)
    assert info.value.status_code == status
    assert len(session.calls) == 1
    assert SECRET not in str(info.value) and TEXT not in str(info.value)


@pytest.mark.parametrize("error, message", [
    (requests.Timeout(f"read timed out {SECRET}"), "timed out"),
    (requests.ConnectionError(f"Authorization: Bearer {SECRET}"), "transport failure"),
])
def test_transport_errors_are_sanitized(error, message):
    session = FakeSession(error)
    with pytest.raises(BackendError, match=message) as info:
        annotator(session)(TEXT)
    assert SECRET not in str(info.value) and info.value.__cause__ is None
    assert len(session.calls) == 1


def test_token_from_environment_is_read_at_preparation(monkeypatch):
    monkeypatch.setenv("JEV_TOKEN", "env-token")
    session = FakeSession(ok())
    Annotator(backend=Jev(session=session), cases=CASES)(TEXT)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer env-token"


def test_missing_token_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("JEV_TOKEN", raising=False)
    session = FakeSession()
    with pytest.raises(ValueError, match="JEV_TOKEN"):
        Annotator(backend=Jev(session=session), cases=CASES)(TEXT)
    assert session.calls == []


def test_choice_option_limit_is_checked_locally():
    cases = {f"case{i}": f"description {i}" for i in range(256)}
    session = FakeSession()
    with pytest.raises(ValueError, match="255"):
        Annotator(backend=Jev(api_key=SECRET, session=session), cases=cases)("text")
    assert session.calls == []


def test_context_estimate_applies_only_to_the_documented_model():
    long_text = "x" * 32_000
    session = FakeSession()
    with pytest.raises(ValueError, match="never truncated"):
        annotator(session)(long_text)
    assert session.calls == []

    other = FakeSession(ok(dict(FIXTURE["response"], model="jev-2.0.0")))
    assert annotator(other, model="jev-2.0.0")(long_text).label == "Sports"
    assert other.calls[0]["json"]["state"] == long_text


def test_owned_session_has_no_retries_and_is_closed():
    backend = Jev(api_key=SECRET)
    backend.prepare(CASES)  # local: builds a session, sends nothing
    session = backend._session
    assert session.get_adapter("https://api.typesafe.ai").max_retries.total == 0
    backend.close()
    assert backend._session is None


def test_injected_session_is_not_closed_by_the_adapter():
    session = FakeSession()
    backend = Jev(api_key=SECRET, session=session)
    backend.close()
    assert session.closed is False


def test_repr_hides_the_token():
    assert SECRET not in repr(Jev(api_key=SECRET))


@pytest.mark.parametrize("kwargs", [{"model": ""}, {"timeout": 0}, {"timeout": float("inf")}])
def test_invalid_adapter_configuration(kwargs):
    with pytest.raises(ValueError):
        Jev(**kwargs)

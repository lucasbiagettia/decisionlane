"""Decisions adapter with offline transport, using the benchmark's contract."""

from __future__ import annotations

import copy

import pytest
import requests

from decisionlane import Annotator, BackendError, DecisionModel, ResponseError
from decisionlane.backends import OpenAI
from fakes import FakeResponse, FakeSession, fixture

FIXTURE = fixture("openai_choice.json")
CASES = {entry["value"]: entry["description"] for entry in FIXTURE["request"]["questions"][0]["choices"]}
TEXT = FIXTURE["request"]["input"]
SECRET = "offline-secret-key"


def ok(body=None):
    return FakeResponse(body if body is not None else FIXTURE["response"],
                        headers={"x-request-id": "req-1"})


def annotator(session, **kwargs):
    return Annotator(backend=OpenAI(api_key=SECRET, session=session, **kwargs), cases=CASES)


def test_payload_and_normalized_decision_match_benchmark_contract():
    session = FakeSession(ok())
    decision = annotator(session)(TEXT)
    [call] = session.calls
    assert call["url"] == "https://api.openai.com/v1/decisions"
    assert call["json"] == FIXTURE["request"]
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["timeout"] == 120.0 and call["allow_redirects"] is False
    assert decision.label == "Sports"
    assert decision.probabilities == {"World": .25, "Sports": .75}
    assert list(decision.probabilities) == list(CASES)
    assert decision.selected_probability == .75
    assert decision.provider_confidence == .42
    assert decision.backend == "openai" and decision.model == "gpt-6-luna"
    assert decision.request_id == "req-1" and decision.explanation is None
    assert decision.latency_ms >= 0


def test_resolved_model_and_explicit_configuration():
    body = copy.deepcopy(FIXTURE["response"])
    body["model"] = "resolved-model"
    session = FakeSession(ok(body))
    decision = annotator(session, model="requested-alias", timeout=45)(TEXT)
    assert session.calls[0]["json"]["model"] == "requested-alias"
    assert session.calls[0]["timeout"] == 45
    assert decision.model == "resolved-model"


def test_named_backend_prepares_lazily_and_batch_reuses_session(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    session = FakeSession(ok(), ok(), ok())
    created = []

    def new_session():
        created.append(session)
        return session

    monkeypatch.setattr("decisionlane.backends.openai.new_session", new_session)
    classifier = Annotator(backend="openai", cases=CASES)
    assert isinstance(classifier._backend, DecisionModel)
    assert classifier.batch([]) == [] and created == []
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        classifier(TEXT)
    assert not session.calls and created == []
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_ORG_ID", "org-1")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-1")
    classifier(TEXT)
    assert [d.backend for d in classifier.batch(["one", "two"])] == ["openai", "openai"]
    assert created == [session]
    assert [c["json"]["input"] for c in session.calls] == [TEXT, "one", "two"]
    assert session.calls[0]["headers"]["OpenAI-Organization"] == "org-1"
    assert session.calls[0]["headers"]["OpenAI-Project"] == "proj-1"
    classifier.close()
    assert session.closed


def test_text_and_case_descriptions_are_unchanged_and_copied():
    cases = {"World": "  Déjà vu\npolitics  ", "Sports": "Sporting events"}
    session = FakeSession(ok())
    classifier = Annotator(backend=OpenAI(api_key=SECRET, session=session), cases=cases)
    cases["World"] = "changed"
    classifier("  Texto con acentos áéí\n  ")
    assert session.calls[0]["json"]["input"] == "  Texto con acentos áéí\n  "
    assert session.calls[0]["json"]["questions"][0]["choices"][0]["description"] == "  Déjà vu\npolitics  "


def test_optional_fields_and_rounding_preserve_provider_values():
    body = copy.deepcopy(FIXTURE["response"])
    body.pop("usage")
    body["answers"][0].pop("confidence")
    body["answers"][0]["probabilities"][0]["probability"] = .74995
    decision = annotator(FakeSession(FakeResponse(body)))(TEXT)
    assert decision.selected_probability == .74995
    assert decision.probabilities["World"] == .25
    assert decision.provider_confidence is None and decision.request_id is None


def test_exact_tie_keeps_provider_choice():
    body = copy.deepcopy(FIXTURE["response"])
    for entry in body["answers"][0]["probabilities"]:
        entry["probability"] = .5
    assert annotator(FakeSession(ok(body)))(TEXT).label == "Sports"


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop("model"),
    lambda b: b.update(model="  "),
    lambda b: b.pop("answers"),
    lambda b: b.update(answers={}),
    lambda b: b.update(answers=[None]),
    lambda b: b["answers"].append(b["answers"][0]),
    lambda b: b["answers"][0].update(type="score"),
    lambda b: b["answers"][0].update(name="other-question"),
    lambda b: b["answers"][0].update(choice="Unknown"),
    lambda b: b["answers"][0].update(choice="World"),
    lambda b: b["answers"][0].update(choice=True),
    lambda b: b["answers"][0].pop("probabilities"),
    lambda b: b["answers"][0].update(probabilities={"World": .25, "Sports": .75}),
    lambda b: b["answers"][0]["probabilities"].pop(),
    lambda b: b["answers"][0]["probabilities"].__setitem__(0, None),
    lambda b: b["answers"][0]["probabilities"][0].update(value="World"),
    lambda b: b["answers"][0]["probabilities"][0].update(value=True),
    lambda b: b["answers"][0]["probabilities"][0].update(value="Unknown"),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=.1),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=float("nan")),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=float("inf")),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=-.1),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=1.1),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=True),
    lambda b: b["answers"][0]["probabilities"][0].update(probability="0.75"),
    lambda b: b["answers"][0].update(confidence=float("inf")),
    lambda b: b["answers"][0].update(confidence=True),
])
def test_invalid_responses_fail_without_retry(mutation):
    body = copy.deepcopy(FIXTURE["response"])
    mutation(body)
    session = FakeSession(ok(body))
    with pytest.raises(ResponseError):
        annotator(session)(TEXT)
    assert len(session.calls) == 1


def test_refusal_stops_batch_without_inventing_label_or_leaking_text():
    body = {"answers": [{"type": "refusal", "name": "classification", "reason": TEXT + SECRET}]}
    session = FakeSession(ok(), ok(body), ok())
    with pytest.raises(ResponseError, match="refused classification") as caught:
        annotator(session).batch(["first", "second", "third"])
    assert len(session.calls) == 2
    assert SECRET not in str(caught.value) and TEXT not in str(caught.value)


@pytest.mark.parametrize("body", ["<html>", [], None])
def test_invalid_json_body(body):
    with pytest.raises(ResponseError):
        annotator(FakeSession(FakeResponse(body)))(TEXT)


@pytest.mark.parametrize("status", [302, 401, 429, 500, 503])
def test_http_errors_have_status_and_do_not_leak_or_retry(status):
    session = FakeSession(FakeResponse({"error": TEXT + SECRET}, status_code=status))
    with pytest.raises(BackendError) as caught:
        annotator(session)(TEXT)
    assert caught.value.status_code == status and len(session.calls) == 1
    assert SECRET not in str(caught.value) and TEXT not in str(caught.value)


@pytest.mark.parametrize("error", [requests.Timeout(SECRET), requests.ConnectionError(SECRET)])
def test_transport_errors_are_sanitized_and_not_retried(error):
    session = FakeSession(error)
    with pytest.raises(BackendError) as caught:
        annotator(session)(TEXT)
    assert SECRET not in str(caught.value) and len(session.calls) == 1


def test_owned_session_has_no_retries_and_closed_backend_cannot_predict():
    backend = OpenAI(api_key=SECRET)
    with pytest.raises(RuntimeError, match="not prepared"):
        backend.predict(TEXT)
    backend.prepare(CASES)
    assert backend._session.get_adapter("https://api.openai.com").max_retries.total == 0
    backend.close()
    assert backend._session is None
    with pytest.raises(RuntimeError, match="closed"):
        backend.predict(TEXT)


def test_choice_limit_is_checked_before_credentials_or_requests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    session = FakeSession()
    cases = {str(i): f"Case {i}" for i in range(256)}
    with pytest.raises(ValueError, match="at most 255"):
        Annotator(backend=OpenAI(session=session), cases=cases)(TEXT)
    assert not session.calls
    cases.pop("255")
    backend = OpenAI(api_key=SECRET, session=session)
    backend.prepare(cases)
    assert len(backend._cases) == 255 and not session.calls


def test_injected_session_is_not_closed_and_repr_hides_key():
    session = FakeSession()
    backend = OpenAI(api_key=SECRET, session=session)
    backend.close()
    assert not session.closed and SECRET not in repr(backend)


@pytest.mark.parametrize("kwargs", [{"model": ""}, {"model": "  "}, {"model": None},
                                    {"timeout": 0}, {"timeout": float("inf")}, {"timeout": float("nan")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OpenAI(**kwargs)

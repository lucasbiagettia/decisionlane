"""Facade contract, exercised with an external fake backend (no network)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Mapping

import pytest

from decisionlane import Annotator, Decision, DecisionModel, ResponseError
from decisionlane.backends import Emissary, Jev

CASES = {
    "refund": "The customer asks for a refund",
    "card_lost": "Reports a lost or stolen card",
    "other": "Anything else",
}


class KeywordBackend:
    """A third-party backend defined outside the package, without subclassing."""

    def __init__(self, fail_on: str | None = None, fail_prepare: int = 0):
        self.prepared_with: list[dict] = []
        self.texts: list[str] = []
        self.fail_on = fail_on
        self.fail_prepare = fail_prepare

    def prepare(self, cases: Mapping[str, str]) -> None:
        if self.fail_prepare:
            self.fail_prepare -= 1
            raise ConnectionError("simulated preparation failure")
        self.prepared_with.append(dict(cases))
        self.labels = list(cases)

    def predict(self, text: str) -> Decision:
        if text == self.fail_on:
            raise RuntimeError("simulated provider failure")
        self.texts.append(text)
        label = "card_lost" if "card" in text else self.labels[-1]
        rest = (1 - 0.7) / (len(self.labels) - 1)
        probabilities = {name: 0.7 if name == label else rest for name in self.labels}
        return Decision(label=label, probabilities=probabilities, backend="keyword", latency_ms=0.1)


def test_single_and_batch_preserve_order_and_prepare_once():
    backend = KeywordBackend()
    annotator = Annotator(backend=backend, cases=CASES)
    decision = annotator("My card was stolen")
    assert decision.label == "card_lost"
    assert decision.selected_probability == pytest.approx(0.7)
    decisions = annotator.batch(["refund please", "lost card", "change address"])
    assert [d.label for d in decisions] == ["other", "card_lost", "other"]
    assert backend.texts == ["My card was stolen", "refund please", "lost card", "change address"]
    annotator.batch(("another card",))
    assert len(backend.prepared_with) == 1


def test_fake_backend_satisfies_protocol():
    assert isinstance(KeywordBackend(), DecisionModel)


def test_empty_batch_returns_empty_list_without_preparing():
    backend = KeywordBackend()
    assert Annotator(backend=backend, cases=CASES).batch([]) == []
    assert backend.prepared_with == []


def test_string_is_not_accepted_as_a_batch():
    with pytest.raises(TypeError, match="sequence of texts"):
        Annotator(backend=KeywordBackend(), cases=CASES).batch("card")


@pytest.mark.parametrize("cases", [
    {"only": "one case"},
    {},
    {"a": "fine", "": "empty id"},
    {"a": "fine", "   ": "blank id"},
    {"a": "fine", "b": ""},
    {"a": "fine", "b": "  "},
    {"a": "fine", "b": None},
    {"a": "fine", 1: "non-string id"},
])
def test_invalid_cases_rejected_at_construction(cases):
    with pytest.raises(ValueError):
        Annotator(backend=KeywordBackend(), cases=cases)


def test_cases_must_be_a_mapping():
    with pytest.raises(TypeError):
        Annotator(backend=KeywordBackend(), cases=[("a", "x"), ("b", "y")])


@pytest.mark.parametrize("text", ["", "   ", None, 3])
def test_invalid_texts_rejected_before_preparation(text):
    backend = KeywordBackend()
    annotator = Annotator(backend=backend, cases=CASES)
    with pytest.raises((ValueError, TypeError)):
        annotator(text)
    with pytest.raises((ValueError, TypeError)):
        annotator.batch(["valid text", text])
    assert backend.prepared_with == [] and backend.texts == []


def test_unknown_backend_lists_supported_names():
    with pytest.raises(ValueError, match="'emissary', 'jev'"):
        Annotator(backend="openai", cases=CASES)


def test_non_backend_object_rejected():
    with pytest.raises(TypeError):
        Annotator(backend=object(), cases=CASES)


def test_named_backends_build_without_network_or_credentials(monkeypatch):
    monkeypatch.delenv("JEV_TOKEN", raising=False)
    monkeypatch.delenv("EMISSARY_API_KEY", raising=False)
    assert isinstance(Annotator(backend="jev", cases=CASES)._backend, Jev)
    assert isinstance(Annotator(backend="emissary", cases=CASES)._backend, Emissary)


def test_configuration_is_independent_of_the_original_mapping():
    cases = dict(CASES)
    backend = KeywordBackend()
    annotator = Annotator(backend=backend, cases=cases)
    cases["injected"] = "added after construction"
    del cases["refund"]
    annotator("hello")
    assert backend.prepared_with == [CASES]


def test_ids_descriptions_and_texts_are_passed_unchanged():
    cases = {" Refund ": "  Déjà vu — refund  ", "other": "x\ny"}
    backend = KeywordBackend()
    Annotator(backend=backend, cases=cases).batch(["  padded  text\t"])
    assert backend.prepared_with == [cases]
    assert backend.texts == ["  padded  text\t"]


def test_failed_preparation_is_not_marked_ready():
    backend = KeywordBackend(fail_prepare=1)
    annotator = Annotator(backend=backend, cases=CASES)
    with pytest.raises(ConnectionError):
        annotator("first")
    assert annotator("second").label == "other"
    assert len(backend.prepared_with) == 1


def test_batch_is_fail_fast_without_partial_results():
    backend = KeywordBackend(fail_on="boom")
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        Annotator(backend=backend, cases=CASES).batch(["a", "boom", "c"])
    assert backend.texts == ["a"]


class _Returning:
    def __init__(self, decision):
        self.decision = decision

    def prepare(self, cases):
        pass

    def predict(self, text):
        return self.decision


VALID = Decision(label="refund", probabilities={"refund": 0.5, "card_lost": 0.3, "other": 0.2},
                 backend="stub", latency_ms=1.0)


@pytest.mark.parametrize("decision", [
    replace(VALID, label="unknown"),
    replace(VALID, probabilities={"refund": 0.5, "card_lost": 0.5}),
    replace(VALID, probabilities={"refund": 0.6, "card_lost": 0.3, "other": 0.2}),
    replace(VALID, probabilities={"refund": 1.2, "card_lost": -0.1, "other": -0.1}),
    replace(VALID, probabilities={"refund": float("nan"), "card_lost": 0.3, "other": 0.2}),
    replace(VALID, provider_confidence=2.0),
])
def test_invalid_backend_decisions_are_rejected(decision):
    with pytest.raises(ResponseError):
        Annotator(backend=_Returning(decision), cases=CASES)("text")


def test_backend_must_return_a_decision():
    with pytest.raises(TypeError):
        Annotator(backend=_Returning({"label": "refund"}), cases=CASES)("text")


def test_decision_without_distribution_and_optional_fields():
    decision = Decision(label="other", probabilities=None, backend="stub", latency_ms=1.0)
    assert Annotator(backend=_Returning(decision), cases=CASES)("x") is decision
    assert decision.selected_probability is None
    assert decision.model is None and decision.request_id is None
    assert decision.provider_confidence is None and decision.explanation is None
    with pytest.raises(FrozenInstanceError):
        decision.label = "refund"  # type: ignore[misc]


def test_explanation_is_preserved_when_a_backend_returns_one():
    decision = replace(VALID, explanation="Mentions money back")
    assert Annotator(backend=_Returning(decision), cases=CASES)("x").explanation == "Mentions money back"


def test_close_only_closes_owned_backends():
    class Closable(KeywordBackend):
        closed = False

        def close(self):
            self.closed = True

    external = Closable()
    Annotator(backend=external, cases=CASES).close()
    assert external.closed is False

    owned = Annotator(backend="jev", cases=CASES)
    calls = []
    owned._backend.close = lambda: calls.append("closed")  # type: ignore[method-assign]
    owned.close()
    assert calls == ["closed"]

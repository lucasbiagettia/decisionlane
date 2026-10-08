"""The public facade: configure cases once, classify one text or a sequence."""

from __future__ import annotations

from typing import Mapping, Sequence

from .backends import Emissary, Jev, OpenAI
from .core import Decision, DecisionModel, ResponseError, check_probabilities, check_unit_interval

SUPPORTED_BACKENDS = ("emissary", "jev", "openai")


class Annotator:
    """Single-label, closed-set zero-shot classifier over user-defined cases.

    ``backend`` is ``"jev"``, ``"emissary"``, ``"openai"`` or any object implementing
    ``DecisionModel``. Construction is local: no network, no credentials.
    The backend is prepared lazily before the first classification and only
    once after a successful preparation.
    """

    def __init__(self, backend: str | DecisionModel, cases: Mapping[str, str]) -> None:
        self._cases = _validated_cases(cases)
        if isinstance(backend, str):
            self._backend: DecisionModel = _resolve(backend)
            self._owns_backend = True
        elif isinstance(backend, DecisionModel):
            self._backend = backend
            self._owns_backend = False
        else:
            raise TypeError("backend must be a backend name or implement prepare() and predict()")
        self._prepared = False

    def __call__(self, text: str) -> Decision:
        """Classify one text."""

        _check_text(text)
        self._prepare()
        return self._predict(text)

    def batch(self, texts: Sequence[str]) -> list[Decision]:
        """Classify texts sequentially, preserving length and order.

        This is one request per text, not a remote batch. All texts are
        validated before any request. Execution is fail-fast: the first error
        is raised and no partial list is returned, although earlier requests
        may already have completed (and been billed); retrying repeats them.
        """

        if isinstance(texts, (str, bytes)):
            raise TypeError("batch() expects a sequence of texts, not a single string")
        texts = list(texts)
        for text in texts:
            _check_text(text)
        if not texts:
            return []
        self._prepare()
        return [self._predict(text) for text in texts]

    def close(self) -> None:
        """Close HTTP sessions of a backend created from a name.

        A backend object passed by the caller remains the caller's to close.
        """

        if self._owns_backend:
            self._backend.close()  # type: ignore[attr-defined]

    def _prepare(self) -> None:
        if not self._prepared:
            self._backend.prepare(dict(self._cases))
            self._prepared = True

    def _predict(self, text: str) -> Decision:
        decision = self._backend.predict(text)
        if not isinstance(decision, Decision):
            raise TypeError("backend predict() must return a Decision")
        if decision.label not in self._cases:
            raise ResponseError("backend returned a label outside the configured cases")
        if decision.probabilities is not None:
            check_probabilities(decision.probabilities, list(self._cases))
        if decision.provider_confidence is not None:
            check_unit_interval(decision.provider_confidence, "provider_confidence")
        return decision


def _resolve(name: str) -> DecisionModel:
    if name == "jev":
        return Jev()
    if name == "emissary":
        return Emissary()
    if name == "openai":
        return OpenAI()
    supported = ", ".join(repr(item) for item in SUPPORTED_BACKENDS)
    raise ValueError(f"Unknown backend {name!r}. Supported backends: {supported}.")


def _validated_cases(cases: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(cases, Mapping):
        raise TypeError("cases must be a mapping of case id to description")
    copied = dict(cases)
    if len(copied) < 2:
        raise ValueError("cases must define at least two cases")
    for label, description in copied.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("case ids must be non-empty strings")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"case {label!r} needs a non-empty description")
    return copied


def _check_text(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text cannot be empty")

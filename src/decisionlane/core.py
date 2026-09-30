"""Result type, backend protocol, errors and shared validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

# Absolute tolerance for provider distributions that do not sum exactly to one
# because of rounding. Invalid distributions are rejected, never renormalized.
PROBABILITY_SUM_TOLERANCE = 1e-4


class BackendError(RuntimeError):
    """A provider request failed: transport error, timeout or non-2xx status.

    Messages never include credentials, headers or request/response bodies.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ResponseError(ValueError):
    """A provider answered, but the response does not match the expected contract."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    """One normalized classification result.

    ``probabilities`` is the provider's distribution over exactly the configured
    labels, or ``None`` when the provider returns none. ``provider_confidence``
    is a provider-native score with provider-specific meaning; it is not the
    probability of ``label``. ``explanation`` is only set when the provider
    returns one in the same response.
    """

    label: str
    probabilities: Mapping[str, float] | None
    backend: str
    model: str | None = None
    request_id: str | None = None
    latency_ms: float
    provider_confidence: float | None = None
    explanation: str | None = None

    @property
    def selected_probability(self) -> float | None:
        """Probability of ``label``, or ``None`` without a distribution."""

        if self.probabilities is None:
            return None
        return self.probabilities[self.label]


@runtime_checkable
class DecisionModel(Protocol):
    """What a backend must implement to be used by ``Annotator``.

    ``prepare`` receives the validated cases once, before the first prediction.
    ``predict`` classifies one text and measures its own ``latency_ms``.
    """

    def prepare(self, cases: Mapping[str, str]) -> None: ...

    def predict(self, text: str) -> Decision: ...


def check_unit_interval(value: object, what: str) -> float:
    """Return ``value`` as float if it is a finite number in [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseError(f"{what} is not a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ResponseError(f"{what} is not a finite number in [0, 1]")
    return number


def check_probabilities(raw: object, labels: Sequence[str]) -> dict[str, float]:
    """Validate a distribution over exactly ``labels``; keep the input key order."""

    if not isinstance(raw, Mapping):
        raise ResponseError("probabilities must be a mapping")
    if set(raw) != set(labels) or len(raw) != len(labels):
        raise ResponseError("probabilities must contain exactly the configured labels")
    probabilities = {
        label: check_unit_interval(value, f"probability for {label!r}")
        for label, value in raw.items()
    }
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise ResponseError(f"probabilities sum to {total}, not 1")
    return probabilities

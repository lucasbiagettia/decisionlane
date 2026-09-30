"""Jev zero-shot adapter over the TypeSafe Choice API."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Mapping

import requests

from ..core import Decision, ResponseError, check_probabilities, check_unit_interval
from ._http import check_timeout, encode, new_session, post_json

DEFAULT_MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = "Which of the configured classes best describes this text?"

# https://docs.typesafe.ai/primitives/choice: a Choice question accepts up to 255 options.
MAX_CHOICE_OPTIONS = 255
# https://docs.typesafe.ai/models documents 32k tokens for state plus the longest
# question for jev-1.13.0 only. The check is a conservative local estimate
# (serialized UTF-8 bytes plus framing), not the provider tokenizer, and is not
# applied to other model versions, whose limits must be verified separately.
CONTEXT_LIMITS = {"jev-1.13.0": 32_000}
FRAMING_ALLOWANCE = 1_024


class Jev:
    """Zero-shot classification with Jev through one TypeSafe Choice question.

    Each case id becomes a Choice option and its description the criterion.
    The token is read from ``JEV_TOKEN`` at preparation unless ``api_key`` is
    given. Preparation is local: it makes no request.
    """

    name = "jev"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.timeout = check_timeout(timeout)
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._cases: dict[str, str] | None = None

    def __repr__(self) -> str:
        return f"Jev(model={self.model!r}, timeout={self.timeout:g})"

    def prepare(self, cases: Mapping[str, str]) -> None:
        if len(cases) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"Jev Choice accepts at most {MAX_CHOICE_OPTIONS} cases, got {len(cases)}")
        api_key = self._api_key or os.environ.get("JEV_TOKEN")
        if not api_key:
            raise ValueError("Jev needs a token: set JEV_TOKEN or pass Jev(api_key=...)")
        self._api_key = api_key
        if self._session is None:
            self._session = new_session()
        self._cases = dict(cases)

    def predict(self, text: str) -> Decision:
        started = perf_counter()
        if self._cases is None or self._session is None:
            raise RuntimeError("Jev is not prepared (prepare() must succeed first) or was closed")
        body = encode({
            "model": self.model,
            "state": text,
            "questions": {"classification": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": self._cases,
            }},
        })
        limit = CONTEXT_LIMITS.get(self.model)
        if limit is not None and len(body) + FRAMING_ALLOWANCE > limit:
            raise ValueError(
                f"Input too long for {self.model}: conservative estimate exceeds {limit} tokens "
                "for state plus question. Input is never truncated."
            )
        response_body, response = post_json(
            self._session, ENDPOINT, body,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.timeout, provider="Jev",
        )
        label, probabilities, confidence, model = self._parse(response_body)
        request_id = response.headers.get("x-request-id")
        return Decision(
            label=label,
            probabilities=probabilities,
            backend=self.name,
            model=model,
            request_id=request_id if isinstance(request_id, str) else None,
            latency_ms=(perf_counter() - started) * 1000,
            provider_confidence=confidence,
        )

    def close(self) -> None:
        """Close the HTTP session if this adapter created it."""

        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def _parse(self, body: dict) -> tuple[str, dict[str, float], float | None, str]:
        assert self._cases is not None
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ResponseError("Jev response does not report the resolved model")
        answers = body.get("answers")
        answer = answers.get("classification") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ResponseError("Jev response lacks the classification Choice answer")
        label = answer.get("choice")
        if not isinstance(label, str) or label not in self._cases:
            raise ResponseError("Jev choice is not one of the configured cases")
        probabilities = check_probabilities(answer.get("probabilities"), list(self._cases))
        probabilities = {case: probabilities[case] for case in self._cases}
        # Ties: an exact maximum chosen by the provider is kept as is.
        if probabilities[label] != max(probabilities.values()):
            raise ResponseError("Jev choice is not a maximum-probability case")
        # Native confidence describes how concentrated the distribution is;
        # it is not the probability of the chosen label.
        confidence = answer.get("confidence")
        if confidence is not None:
            confidence = check_unit_interval(confidence, "Jev confidence")
        return label, probabilities, confidence, model


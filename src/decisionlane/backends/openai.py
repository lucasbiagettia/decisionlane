"""OpenAI zero-shot adapter over the Decisions Choice API."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Mapping

import requests

from ..core import Decision, ResponseError, check_probabilities, check_unit_interval
from ._http import check_timeout, encode, new_session, post_json

DEFAULT_MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/decisions"
# https://developers.openai.com/api/reference/resources/decisions/methods/create
MAX_CHOICE_OPTIONS = 255
# Same classification rubric as llm-classifier-bench's Decisions adapter.
INSTRUCTIONS = (
    "Classify the input text into exactly one of the allowed labels. "
    "Use the class descriptions as the decision rubric. Do not invent labels."
)


class OpenAI:
    """Zero-shot classification through one named Decisions Choice question.

    Case ids become choice values and descriptions become the decision rubric.
    Preparation is local. The key is read from ``OPENAI_API_KEY`` at preparation
    unless ``api_key`` is given. No OpenAI SDK is required.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
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
        return f"OpenAI(model={self.model!r}, timeout={self.timeout:g})"

    def prepare(self, cases: Mapping[str, str]) -> None:
        if len(cases) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"OpenAI Decisions Choice accepts at most {MAX_CHOICE_OPTIONS} cases, got {len(cases)}")
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI needs a key: set OPENAI_API_KEY or pass OpenAI(api_key=...)")
        self._api_key = api_key
        if self._session is None:
            self._session = new_session()
        self._cases = dict(cases)

    def predict(self, text: str) -> Decision:
        started = perf_counter()
        if self._cases is None or self._session is None:
            raise RuntimeError("OpenAI is not prepared (prepare() must succeed first) or was closed")
        body = encode({
            "model": self.model,
            "input": text,
            "questions": [{
                "type": "choice",
                "name": "classification",
                "instructions": INSTRUCTIONS,
                "choices": [
                    {"value": label, "description": description}
                    for label, description in self._cases.items()
                ],
            }],
        })
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for variable, header in (
            ("OPENAI_ORG_ID", "OpenAI-Organization"),
            ("OPENAI_PROJECT_ID", "OpenAI-Project"),
        ):
            value = os.environ.get(variable)
            if value:
                headers[header] = value
        response_body, response = post_json(
            self._session, ENDPOINT, body,
            headers=headers, timeout=self.timeout, provider="OpenAI Decisions",
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
        answers = body.get("answers")
        if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict):
            raise ResponseError("OpenAI Decisions requires exactly one classification answer")
        answer = answers[0]
        if answer.get("type") == "refusal":
            raise ResponseError("OpenAI Decisions refused classification; no label or probabilities returned")
        if answer.get("type") != "choice" or answer.get("name") != "classification":
            raise ResponseError("OpenAI Decisions lacks the named classification Choice answer")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ResponseError("OpenAI Decisions response does not report the resolved model")
        label = answer.get("choice")
        if not isinstance(label, str) or label not in self._cases:
            raise ResponseError("OpenAI Decisions choice is not one of the configured cases")
        entries = answer.get("probabilities")
        if not isinstance(entries, list) or len(entries) != len(self._cases):
            raise ResponseError("OpenAI Decisions probabilities must contain exactly the configured labels")
        raw = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
                raise ResponseError("OpenAI Decisions probability entries must have string labels")
            value = entry["value"]
            if value not in self._cases or value in raw:
                raise ResponseError("OpenAI Decisions probabilities contain duplicate or unknown labels")
            raw[value] = entry.get("probability")
        probabilities = check_probabilities(raw, list(self._cases))
        probabilities = {case: probabilities[case] for case in self._cases}
        # Keep the provider's chosen maximum, including exact ties.
        if probabilities[label] != max(probabilities.values()):
            raise ResponseError("OpenAI Decisions choice is not a maximum-probability case")
        confidence = answer.get("confidence")
        if confidence is not None:
            confidence = check_unit_interval(confidence, "OpenAI Decisions confidence")
        return label, probabilities, confidence, model

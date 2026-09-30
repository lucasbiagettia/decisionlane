"""Emissary zero-shot adapter over a routing experiment."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Mapping
from uuid import uuid4

import requests

from ..core import Decision, ResponseError, check_probabilities
from ._http import check_timeout, encode, new_session, post_json

BASE_URL = "https://api.withemissary.com"


class Emissary:
    """Zero-shot classification with an Emissary routing experiment.

    Preparation creates one remote experiment (``POST /v1/experiments``,
    ``mode="routing"``) from the case ids and descriptions. This is
    configuration, not training: no labeled examples are sent. The resulting
    experiment/version is reused for every prediction of this instance.
    Another instance, or another process, creates another experiment.

    The key is read from ``EMISSARY_API_KEY`` at preparation unless
    ``api_key`` is given.
    """

    name = "emissary"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        experiment_name: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if experiment_name is not None and (
            not isinstance(experiment_name, str) or not experiment_name.strip()
        ):
            raise ValueError("experiment_name must be a non-empty string")
        self.timeout = check_timeout(timeout)
        self.experiment_name = experiment_name
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._labels: list[str] | None = None
        self.model_id: str | None = None
        """``experiment_id/version`` created by the last successful preparation."""

    def __repr__(self) -> str:
        return f"Emissary(model_id={self.model_id!r}, timeout={self.timeout:g})"

    def prepare(self, cases: Mapping[str, str]) -> None:
        api_key = self._api_key or os.environ.get("EMISSARY_API_KEY")
        if not api_key:
            raise ValueError("Emissary needs an API key: set EMISSARY_API_KEY or pass Emissary(api_key=...)")
        self._api_key = api_key
        if self._session is None:
            self._session = new_session()
        self._labels, self.model_id = None, None
        body, _ = self._post("/v1/experiments", {
            "name": self.experiment_name or f"decisionlane-{uuid4().hex[:12]}",
            "mode": "routing",
            "classes": [{"name": label, "description": text} for label, text in cases.items()],
        })
        experiment_id = body.get("id")
        version = body.get("latest_version")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ResponseError("Emissary experiment response lacks a valid 'id'")
        if (not isinstance(version, str) or not version.strip()
                or version.lower() == "latest" or "/" in version):
            raise ResponseError("Emissary experiment response lacks a valid 'latest_version'")
        self._labels = list(cases)
        self.model_id = f"{experiment_id}/{version}"

    def predict(self, text: str) -> Decision:
        started = perf_counter()
        if self._labels is None or self.model_id is None or self._session is None:
            raise RuntimeError("Emissary is not prepared (prepare() must succeed first) or was closed")
        body, _ = self._post("/v1/classification", {
            "model": self.model_id,
            "input": text,
            "data_format": "probs",
        })
        data = body.get("data")
        item = data[0] if isinstance(data, list) and data else None
        if not isinstance(item, dict):
            raise ResponseError("Emissary classification response has no 'data' item")
        raw = check_probabilities(item.get("probs"), self._labels)
        # Emissary returns a distribution, not a chosen label. The label is the
        # maximum; on an exact tie, the first tied label in response order wins.
        label = max(raw, key=raw.__getitem__)
        model = body.get("model")
        if isinstance(model, str) and model != self.model_id:
            raise ResponseError("Emissary response model differs from the prepared experiment version")
        request_id = body.get("id")
        return Decision(
            label=label,
            probabilities={case: raw[case] for case in self._labels},
            backend=self.name,
            model=self.model_id,
            request_id=request_id if isinstance(request_id, str) else None,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def close(self) -> None:
        """Close the HTTP session if this adapter created it."""

        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def _post(self, path: str, payload: dict) -> tuple[dict, requests.Response]:
        assert self._session is not None
        return post_json(
            self._session, BASE_URL + path, encode(payload),
            headers={"X-API-Key": self._api_key or ""},
            timeout=self.timeout, provider="Emissary",
        )

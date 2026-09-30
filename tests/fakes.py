"""Simulated HTTP transport for offline adapter tests. Nothing here reaches a provider."""

from __future__ import annotations

import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = copy.deepcopy(body)
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return copy.deepcopy(self._body)


class FakeSession:
    """Returns queued outcomes in order and records every POST."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs, "json": json.loads(kwargs["data"])})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True

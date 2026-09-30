"""Minimal JSON-over-HTTPS helpers shared by the adapters.

No retries, no redirects, no logging. Errors never carry credentials, headers
or bodies, because request and response bodies can contain the input text.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import requests
from requests.adapters import HTTPAdapter

from ..core import BackendError, ResponseError


def new_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=0))
    return session


def check_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a number of seconds")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    return float(timeout)


def encode(payload: Mapping[str, Any]) -> bytes:
    """Serialize as UTF-8 JSON without escaping or altering the text."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def post_json(
    session: requests.Session,
    url: str,
    body: bytes,
    *,
    headers: Mapping[str, str],
    timeout: float,
    provider: str,
) -> tuple[dict[str, Any], requests.Response]:
    """POST once and return the JSON object body with the raw response."""

    try:
        response = session.post(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.Timeout:
        raise BackendError(f"{provider} request timed out after {timeout:g} s") from None
    except requests.RequestException:
        raise BackendError(f"{provider} transport failure") from None
    if not 200 <= response.status_code < 300:
        raise BackendError(
            f"{provider} returned HTTP {response.status_code}", status_code=response.status_code
        )
    try:
        parsed = response.json()
    except ValueError:
        raise ResponseError(f"{provider} returned a non-JSON response") from None
    if not isinstance(parsed, dict):
        raise ResponseError(f"{provider} returned JSON that is not an object")
    return parsed, response

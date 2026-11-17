"""How long to wait when a provider says "too many requests".

A free tier says no often, and it usually says exactly how long to wait:
in a Retry-After header, in a retryDelay field, or in plain words inside the
error message. Giving up a second later wastes the whole test for no reason.

So on a 429 we wait the time the provider asked for, up to a minute, and try
once more. If it says no again, that one test becomes ERROR and the run goes on.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Never wait longer than this, however long the provider asks for. A run that
# sits still for five minutes is worse than a test marked ERROR.
MAX_WAIT_SECONDS = 60.0

# "retryDelay": "17s"   (Google)
_RETRY_DELAY_FIELD = re.compile(r'retry_?delay"?\s*:\s*"?([0-9]+(?:\.[0-9]+)?)\s*(ms|s)', re.IGNORECASE)
# "Please retry in 17.75s." / "Please try again in 7.5s"
_RETRY_IN_WORDS = re.compile(
    r"(?:retry|try again)[^0-9]{0,24}?([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\b", re.IGNORECASE
)


def _as_seconds(amount: str, unit: str) -> float | None:
    try:
        number = float(amount)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number / 1000 if unit.lower() == "ms" else number


def _from_header(problem: Any) -> float | None:
    answer = getattr(problem, "response", None)
    headers = getattr(answer, "headers", None)
    if headers is None:
        return None
    for name in ("retry-after", "retry-after-ms", "x-ratelimit-reset-requests"):
        try:
            raw = headers.get(name)
        except (AttributeError, TypeError):
            return None
        if not raw:
            continue
        unit = "ms" if name.endswith("-ms") else "s"
        found = _as_seconds(str(raw).strip().rstrip("s"), unit)
        if found is not None:
            return found
    return None


def _from_text(text: str) -> float | None:
    for pattern in (_RETRY_DELAY_FIELD, _RETRY_IN_WORDS):
        found = pattern.search(text)
        if found:
            seconds = _as_seconds(found.group(1), found.group(2))
            if seconds is not None:
                return seconds
    return None


def seconds_to_wait(problem: Any) -> float | None:
    """How long the provider asked us to wait, or None when it did not say.

    Never more than MAX_WAIT_SECONDS, and never less than nothing.
    """
    found = _from_header(problem)
    if found is None:
        body = getattr(problem, "body", None)
        text = ""
        if body is not None:
            try:
                text = json.dumps(body)
            except (TypeError, ValueError):
                text = str(body)
        text = text + " " + str(getattr(problem, "message", "") or "") + " " + str(problem)
        found = _from_text(text)

    if found is None:
        return None
    return min(max(found, 0.0), MAX_WAIT_SECONDS)

"""Remembers judge answers, so the same reply is never judged twice.

The key is a hash of everything that could change the answer: the judge model,
the prompt version, the user request, the reply and the expectation. Change any
of those and the key changes, so a stale answer can never be reused.

The file is .agentqa_cache/judge_cache.json, which git ignores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentqa.project_paths import PROJECT_ROOT

CACHE_FOLDER_NAME = ".agentqa_cache"
CACHE_FILE_NAME = "judge_cache.json"
DEFAULT_CACHE_FILE = PROJECT_ROOT / CACHE_FOLDER_NAME / CACHE_FILE_NAME


def cache_key(
    judge_model: str,
    prompt_version: str,
    user_request: str,
    reply: str,
    expectation: str,
) -> str:
    """One short key for one judging question."""
    parts = [judge_model, prompt_version, user_request, reply or "", expectation]
    joined = "\n--\n".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class JudgeCache:
    """Judge answers on disk. A broken file is ignored, never a crash."""

    def __init__(self, path: str | Path | None = None, enabled: bool = True) -> None:
        self.path = Path(path) if path else DEFAULT_CACHE_FILE
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._answers: dict[str, Any] = {} if not enabled else self._read()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return saved if isinstance(saved, dict) else {}

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        found = self._answers.get(key)
        if isinstance(found, dict):
            self.hits += 1
            return dict(found)
        self.misses += 1
        return None

    def set(self, key: str, answer: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._answers[key] = dict(answer)
        self.save()

    def save(self) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._answers, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # A cache that cannot be written is a lost saving, not a failed run.
            pass

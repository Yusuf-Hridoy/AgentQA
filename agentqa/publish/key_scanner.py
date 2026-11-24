"""Looks for an API key before anything is copied into the public folder.

Publishing is the one step that cannot be taken back. Once a key is pushed to
a public repository it is burned, whatever the next commit says, so the scan
runs before the copy and a single hit stops the whole command.

Two kinds of thing are looked for:

    the shapes keys come in     AIza… for Gemini, gsk_… for Groq
    the keys on this machine    the real value of every *_API_KEY set now

The second catches a provider this file has never heard of. Nothing here ever
prints, logs or returns the matched text: a scanner that quotes the key it
found has leaked it to the terminal, to the scrollback and to CI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Key shapes, as the providers write them.
KEY_PATTERNS = (
    ("a Gemini key", re.compile(r"AIza[0-9A-Za-z_\-]{20,}")),
    ("a Groq key", re.compile(r"gsk_[0-9A-Za-z]{20,}")),
    ("an OpenAI-style key", re.compile(r"sk-[0-9A-Za-z_\-]{20,}")),
)

# An environment value shorter than this is not a key, it is a setting.
SHORTEST_REAL_KEY = 12

# Files that are not text are not read.
TEXT_SUFFIXES = (".json", ".html", ".md", ".txt", ".yaml", ".yml", ".csv", ".js", ".css", "")


@dataclass(frozen=True)
class KeyFound:
    """Where a key turned up. The key itself is never kept."""

    path: Path
    what: str

    def as_text(self) -> str:
        return str(self.path) + " looks like it holds " + self.what


def key_values_in_environment() -> list[tuple[str, str]]:
    """Every real key set right now, as (variable name, value)."""
    found = []
    for name, value in os.environ.items():
        if not name.endswith("_API_KEY") and not name.endswith("_TOKEN"):
            continue
        cleaned = (value or "").strip()
        if len(cleaned) >= SHORTEST_REAL_KEY:
            found.append((name, cleaned))
    return found


def scan_text(text: str, environment_keys: list[tuple[str, str]] | None = None) -> str | None:
    """What kind of key this text holds, or None. Never gives back the key."""
    for what, pattern in KEY_PATTERNS:
        if pattern.search(text):
            return what
    for name, value in environment_keys if environment_keys is not None else []:
        if value and value in text:
            return "the value of " + name
    return None


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def scan_folder(folder: str | Path) -> list[KeyFound]:
    """Every file under the folder that looks like it holds a key."""
    base = Path(folder)
    environment_keys = key_values_in_environment()
    found = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        what = scan_text(text, environment_keys)
        if what:
            found.append(KeyFound(path=path, what=what))
    return found

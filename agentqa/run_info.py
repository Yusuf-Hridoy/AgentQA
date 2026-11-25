"""What a run was made of: which brain, which judge, which rules, which tests.

Two results are only comparable when these match. The hashes make that easy to
see: change one word of AGENT_RULES or one test case file, and the hash changes.
`agentqa compare` refuses runs whose hashes or prompt versions differ, so this
block is what makes a scorecard fair.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agentqa import __version__
from agentqa.agent_loop import AGENT_RULES
from agentqa.agents.agent_interface import AGENT_BUILTIN, NO_FRAMEWORK
from agentqa.project_paths import TEST_CASES_DIR

HASH_LENGTH = 8


def short_hash(text: str) -> str:
    """The first 8 characters of the SHA-256. Enough to spot a change."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def agent_rules_hash() -> str:
    return short_hash(AGENT_RULES)


def test_cases_hash(folder: str | Path | None = None) -> str:
    """One hash over every test case file, in file name order."""
    directory = Path(folder) if folder else TEST_CASES_DIR
    files = sorted(
        list(directory.glob("*.yaml")) + list(directory.glob("*.yml")),
        key=lambda path: path.name,
    )
    parts = []
    for path in files:
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            body = ""
        parts.append(path.name + "\n" + body)
    return short_hash("\n--\n".join(parts))


def build_run_info(
    brain_name: str,
    provider: str | None,
    model: str,
    judge_mode: str,
    judge_model: str | None,
    judge_prompt_version: str | None,
    cases_folder: str | Path | None = None,
    claim_prompt_version: str | None = None,
    repeat: int = 1,
    agent: str = AGENT_BUILTIN,
    agent_framework: str | None = None,
    guard_prompt_version: str | None = None,
) -> dict[str, Any]:
    return {
        "brain": brain_name,
        # Which agent was tested, and what built it. The agent may differ
        # between two runs on one scorecard: it is one of the things being
        # compared. The rules and the test cases may not.
        "agent": agent,
        "agent_framework": agent_framework or NO_FRAMEWORK,
        "guard_prompt_version": guard_prompt_version,
        "provider": provider,
        "model": model,
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "claim_prompt_version": claim_prompt_version,
        "repeat": int(repeat),
        "agent_rules_hash": agent_rules_hash(),
        "test_cases_hash": test_cases_hash(cases_folder),
        "agentqa_version": __version__,
    }

"""One check result: what was checked, did it pass, why not, and who said so."""

from __future__ import annotations

from pydantic import BaseModel

# The three areas a check can belong to.
AREA_TASK = "task"      # did the agent finish the job?
AREA_STEPS = "steps"    # did it take the right steps?
AREA_SAFETY = "safety"  # was it safe?

ALL_AREAS = (AREA_TASK, AREA_STEPS, AREA_SAFETY)

# Who decided this check.
SOURCE_CODE = "code"    # plain code looked at a fact
SOURCE_JUDGE = "judge"  # the AI judge read the meaning


class CheckResult(BaseModel):
    check_name: str
    passed: bool
    message: str
    area: str
    source: str = SOURCE_CODE
    # The judge's own sentence. Empty for code checks.
    reason: str | None = None


def passed(check_name: str, message: str, area: str, source: str = SOURCE_CODE) -> CheckResult:
    return CheckResult(check_name=check_name, passed=True, message=message, area=area, source=source)


def failed(check_name: str, message: str, area: str, source: str = SOURCE_CODE) -> CheckResult:
    return CheckResult(check_name=check_name, passed=False, message=message, area=area, source=source)


def judged(check_name: str, is_pass: bool, reason: str, area: str) -> CheckResult:
    """A result the AI judge decided. The reason is shown to the person."""
    return CheckResult(
        check_name=check_name,
        passed=is_pass,
        message="judge: " + reason,
        area=area,
        source=SOURCE_JUDGE,
        reason=reason,
    )

"""One check result: what was checked, did it pass, and why not."""

from __future__ import annotations

from pydantic import BaseModel

# The three areas a check can belong to.
AREA_TASK = "task"      # did the agent finish the job?
AREA_STEPS = "steps"    # did it take the right steps?
AREA_SAFETY = "safety"  # was it safe?

ALL_AREAS = (AREA_TASK, AREA_STEPS, AREA_SAFETY)


class CheckResult(BaseModel):
    check_name: str
    passed: bool
    message: str
    area: str


def passed(check_name: str, message: str, area: str) -> CheckResult:
    return CheckResult(check_name=check_name, passed=True, message=message, area=area)


def failed(check_name: str, message: str, area: str) -> CheckResult:
    return CheckResult(check_name=check_name, passed=False, message=message, area=area)

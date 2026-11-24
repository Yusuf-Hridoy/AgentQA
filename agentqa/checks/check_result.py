"""One check result: what was checked, did it pass, why not, and who said so."""

from __future__ import annotations

from pydantic import BaseModel

# The four areas a check can belong to.
AREA_TASK = "task"        # did the agent finish the job?
AREA_STEPS = "steps"      # did it take the right steps?
AREA_SAFETY = "safety"    # was it safe?
AREA_HONESTY = "honesty"  # did it tell the truth about what it did?

ALL_AREAS = (AREA_TASK, AREA_STEPS, AREA_SAFETY, AREA_HONESTY)

# The areas again, most serious first. A run line shows the worst failed check,
# so a lie is never hidden behind a wrong step.
AREAS_WORST_FIRST = (AREA_HONESTY, AREA_SAFETY, AREA_TASK, AREA_STEPS)


def area_seriousness(area: str) -> int:
    """How serious an area is. Lower is worse. An unknown area comes last."""
    if area in AREAS_WORST_FIRST:
        return AREAS_WORST_FIRST.index(area)
    return len(AREAS_WORST_FIRST)


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
    # Only on the honesty check: what the reply said it had done, and who read
    # those claims out of the reply.
    claimed_actions: list[str] | None = None
    claim_reader: str | None = None


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

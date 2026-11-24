"""agentqa check-judge: does the judge agree with the human labels?

The command talks to a real judge, so here it is given a fake one. No network,
no key.
"""

from __future__ import annotations

import json

import pytest
from fake_openai_client import FakeOpenAIClient, text_answer
from typer.testing import CliRunner

from agentqa import cli
from agentqa.claims.claim_calibration_loader import load_claim_examples
from agentqa.cli import app
from agentqa.judge.judge_cache import JudgeCache
from agentqa.judge.judge_calibration_loader import (
    JudgeExampleFileError,
    load_judge_examples,
)
from agentqa.judge.reply_judge import ReplyJudge

runner = CliRunner()


def verdict_text(verdict: str, reason: str = "that is what it means"):
    return text_answer(json.dumps({"verdict": verdict, "reason": reason}))


class ReaderThatSays:
    """A claim reader that gives back a fixed list per example id."""

    name = "ai"

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = dict(answers)

    def read_claims(self, user_request: str, reply: str) -> list[str]:
        return list(self.answers.get(reply, []))


def reader_that_agrees_with_every_label():
    return ReaderThatSays({one.reply: one.expected for one in load_claim_examples()})


def judge_that_says(verdicts, tmp_path):
    client = FakeOpenAIClient(answers=[verdict_text(one) for one in verdicts])
    return ReplyJudge(
        model="test-judge",
        client=client,
        cache=JudgeCache(path=tmp_path / "judge_cache.json", enabled=False),
    )


# ---------- the example file ----------


def test_there_are_ten_examples_four_pass_and_six_fail():
    examples = load_judge_examples()

    assert len(examples) == 10
    assert len([one for one in examples if one.should_pass]) == 4
    assert len([one for one in examples if not one.should_pass]) == 6


def test_every_example_has_everything_the_judge_needs():
    for example in load_judge_examples():
        assert example.id
        assert example.user_asks and example.reply and example.expectation
        assert example.label in ("pass", "fail")


def test_a_missing_example_file_says_so(tmp_path):
    with pytest.raises(JudgeExampleFileError) as problem:
        load_judge_examples(tmp_path / "nothing.yaml")

    assert "No judge example file" in str(problem.value)


def test_a_bad_label_is_refused(tmp_path):
    path = tmp_path / "judge_examples.yaml"
    path.write_text(
        "examples:\n  - id: a\n    user_asks: q\n    reply: r\n    expectation: e\n    label: maybe\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgeExampleFileError) as problem:
        load_judge_examples(path)

    assert "label must be pass or fail" in str(problem.value)


# ---------- the command ----------


def test_a_judge_that_agrees_with_every_label_scores_ten_of_ten(tmp_path, monkeypatch):
    labels = [example.label for example in load_judge_examples()]
    monkeypatch.setattr(cli, "build_judge", lambda *a, **k: judge_that_says(labels, tmp_path))
    monkeypatch.setattr(cli, "build_claim_reader", lambda *a, **k: reader_that_agrees_with_every_label())

    result = runner.invoke(app, ["check-judge"])

    assert result.exit_code == 0
    assert "Judge agreed with the human label on 10 of 10." in result.output
    assert "Claim reader matched the human label on 10 of 10." in result.output


def test_a_disagreement_is_counted_and_listed(tmp_path, monkeypatch):
    labels = [example.label for example in load_judge_examples()]
    labels[0] = "fail"  # the judge gets the first example wrong
    monkeypatch.setattr(cli, "build_judge", lambda *a, **k: judge_that_says(labels, tmp_path))
    monkeypatch.setattr(cli, "build_claim_reader", lambda *a, **k: reader_that_agrees_with_every_label())

    result = runner.invoke(app, ["check-judge"])

    assert "Judge agreed with the human label on 9 of 10." in result.output
    assert "Where the judge differs:" in result.output
    assert "plain-pass" in result.output


def test_a_claim_reader_that_reads_a_reply_wrongly_is_listed(tmp_path, monkeypatch):
    labels = [example.label for example in load_judge_examples()]
    monkeypatch.setattr(cli, "build_judge", lambda *a, **k: judge_that_says(labels, tmp_path))
    # This reader reads nothing in any reply, so every claim example differs.
    monkeypatch.setattr(cli, "build_claim_reader", lambda *a, **k: ReaderThatSays({}))

    result = runner.invoke(app, ["check-judge"])

    assert "Claim reader matched the human label on 4 of 10." in result.output
    assert "Where the claim reader differs:" in result.output
    assert "expected create_ticket, read nothing" in result.output


def test_a_missing_gemini_key_gives_a_setup_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, ["check-judge"])

    assert result.exit_code == 2
    assert "Set GEMINI_API_KEY in your .env file." in result.output

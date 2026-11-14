"""A broken test case file must give a clear error that names the file."""

from __future__ import annotations

import pytest

from agentqa.brains.fake_brain import FakeBrain, FakeBrainScriptError
from agentqa.project_paths import FAKE_BRAIN_SCRIPTS_DIR, TEST_CASES_DIR
from agentqa.test_case_loader import TestCaseFileError, load_test_case_file, load_test_cases

GOOD_FILE = """
id: demo-case
type: normal
acting_user: alice
user_asks: "Please make a ticket."
should_use:
  - create_ticket
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_good_file_loads_and_remembers_its_file_name(tmp_path):
    case = load_test_case_file(write(tmp_path, "demo-case.yaml", GOOD_FILE))

    assert case.id == "demo-case"
    assert case.source_file == "demo-case.yaml"
    assert case.max_steps == 8


def test_a_missing_field_names_the_file_and_the_field(tmp_path):
    path = write(tmp_path, "no-user.yaml", "id: x\ntype: normal\nuser_asks: hello\n")

    with pytest.raises(TestCaseFileError) as problem:
        load_test_case_file(path)

    assert "no-user.yaml" in str(problem.value)
    assert "acting_user" in str(problem.value)


def test_an_unknown_tool_name_is_refused(tmp_path):
    path = write(tmp_path, "bad-tool.yaml", GOOD_FILE.replace("create_ticket", "delete_database"))

    with pytest.raises(TestCaseFileError) as problem:
        load_test_case_file(path)

    assert "bad-tool.yaml" in str(problem.value)
    assert "Unknown tool" in str(problem.value)


def test_an_unknown_type_is_refused(tmp_path):
    path = write(tmp_path, "bad-type.yaml", GOOD_FILE.replace("type: normal", "type: whatever"))

    with pytest.raises(TestCaseFileError) as problem:
        load_test_case_file(path)

    assert "bad-type.yaml" in str(problem.value)


def test_an_extra_field_is_refused(tmp_path):
    path = write(tmp_path, "extra.yaml", GOOD_FILE + "reply_shuld_say: [ticket]\n")

    with pytest.raises(TestCaseFileError) as problem:
        load_test_case_file(path)

    assert "extra.yaml" in str(problem.value)


def test_broken_yaml_is_refused(tmp_path):
    path = write(tmp_path, "broken.yaml", "id: demo\n  type: [normal\n")

    with pytest.raises(TestCaseFileError) as problem:
        load_test_case_file(path)

    assert "broken.yaml" in str(problem.value)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(TestCaseFileError):
        load_test_case_file(tmp_path / "not-here.yaml")


def test_two_files_with_the_same_id_are_refused(tmp_path):
    write(tmp_path, "one.yaml", GOOD_FILE)
    write(tmp_path, "two.yaml", GOOD_FILE)

    with pytest.raises(TestCaseFileError) as problem:
        load_test_cases(tmp_path)

    assert "use the id demo-case" in str(problem.value)
    assert "one.yaml" in str(problem.value) and "two.yaml" in str(problem.value)


def test_only_picks_one_test_case():
    picked = load_test_cases(TEST_CASES_DIR, only="printer-ticket")

    assert len(picked) == 1 and picked[0].id == "printer-ticket"


def test_only_with_an_unknown_id_lists_the_known_ids():
    with pytest.raises(TestCaseFileError) as problem:
        load_test_cases(TEST_CASES_DIR, only="does-not-exist")

    assert "printer-ticket" in str(problem.value)


def test_the_project_has_ten_test_cases_and_ten_scripts():
    cases = load_test_cases(TEST_CASES_DIR)

    assert len(cases) == 10
    for case in cases:
        assert (FAKE_BRAIN_SCRIPTS_DIR / (case.id + ".yaml")).is_file()


def test_a_broken_fake_brain_script_is_refused(tmp_path):
    path = write(tmp_path, "demo-case.yaml", "actions:\n  - note: hello\n")

    with pytest.raises(FakeBrainScriptError) as problem:
        FakeBrain.from_script_file(path)

    assert "demo-case.yaml" in str(problem.value)


def test_a_missing_fake_brain_script_is_refused(tmp_path):
    with pytest.raises(FakeBrainScriptError):
        FakeBrain.for_test_case("not-here", folder=tmp_path)

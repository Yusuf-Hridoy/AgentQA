"""Where the project files live.

Paths are worked out from this file, so they also work after an editable install.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

FAKE_DATA_DIR = PROJECT_ROOT / "fake_data"
TEST_CASES_DIR = PROJECT_ROOT / "test_cases"
FAKE_BRAIN_SCRIPTS_DIR = PROJECT_ROOT / "fake_brain_scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
JUDGE_CALIBRATION_DIR = PROJECT_ROOT / "judge_calibration"
CACHE_DIR = PROJECT_ROOT / ".agentqa_cache"

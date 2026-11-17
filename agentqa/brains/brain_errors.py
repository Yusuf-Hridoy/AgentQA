"""What can go wrong with a real brain.

Two kinds, on purpose:

BrainSetupError  the brain cannot be used at all (no key, Ollama not running).
                 AgentQA stops before the first test and exits with code 2.
BrainError       one API call failed after the SDK already retried.
                 Only that one test becomes ERROR. The run goes on.
"""

from __future__ import annotations


class BrainSetupError(Exception):
    """The brain is not ready. Nothing can run."""


class BrainError(Exception):
    """One call to the model failed. One test is lost, not the run."""

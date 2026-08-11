import json
import subprocess
from typing import Any

import pytest

from planner.claude_code_client import ClaudeCodeClient, ClaudeCodeError


class FakeRunner:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.argv: list[str] = []

    def __call__(self, argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.argv = argv
        return subprocess.CompletedProcess(argv, 0, json.dumps(self.payload), "")


def test_client_runs_claude_with_structured_output_and_allowed_directories() -> None:
    runner = FakeRunner({"job_id": "job-1", "search_policy": {"max_candidates": 1}})
    client = ClaudeCodeClient(runner=runner)

    payload = client.run(
        prompt="Create a search plan.",
        json_schema={"type": "object"},
        add_dirs=["/data/LLMOptAgent/specs", "/data/LLMOptAgent/references"],
    )

    assert payload["job_id"] == "job-1"
    assert runner.argv[:3] == ["claude", "-p", "Create a search plan."]
    assert "--output-format" in runner.argv
    assert "--json-schema" in runner.argv
    assert runner.argv.count("--add-dir") == 2
    assert "--dangerously-skip-permissions" not in runner.argv


def test_client_parses_claude_result_envelope() -> None:
    runner = FakeRunner({"result": '{"job_id": "job-1"}'})

    payload = ClaudeCodeClient(runner=runner).run("prompt", {}, [])

    assert payload == {"job_id": "job-1"}


def test_client_reports_nonzero_exit_and_invalid_json() -> None:
    def failed_runner(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 7, "", "permission denied")

    with pytest.raises(ClaudeCodeError, match="status 7"):
        ClaudeCodeClient(runner=failed_runner).run("prompt", {}, [])

    def invalid_runner(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "not-json", "")

    with pytest.raises(ClaudeCodeError, match="invalid JSON"):
        ClaudeCodeClient(runner=invalid_runner).run("prompt", {}, [])

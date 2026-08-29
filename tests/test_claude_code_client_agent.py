"""锁定 L2 生成压测命令用的 CLI 选择逻辑(默认 tclaude,与 L1 gen_configs.sh 一致)。

只测 executable 的解析优先级,不真调 CLI:
  显式传入 > BENCH_AGENT > LLM_INFER_AGENT > 默认 tclaude
"""

from __future__ import annotations

import pytest

from planner.claude_code_client import ClaudeCodeClient, _default_executable


@pytest.fixture(autouse=True)
def _clear_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个用例都从干净环境起,免得外部 shell 里已 export 的值污染断言。"""
    monkeypatch.delenv("BENCH_AGENT", raising=False)
    monkeypatch.delenv("LLM_INFER_AGENT", raising=False)


def test_default_is_tclaude() -> None:
    """不设任何环境变量、不显式传:默认 tclaude(与 gen_configs.sh 统一)。"""
    assert _default_executable() == "tclaude"
    assert ClaudeCodeClient().executable == "tclaude"


def test_bench_agent_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_AGENT", "claude")
    assert _default_executable() == "claude"
    assert ClaudeCodeClient().executable == "claude"


def test_bench_agent_takes_priority_over_llm_infer_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCH_AGENT", "tclaude")
    monkeypatch.setenv("LLM_INFER_AGENT", "claude")
    assert _default_executable() == "tclaude"


def test_llm_infer_agent_used_when_bench_agent_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_INFER_AGENT", "claude")
    assert _default_executable() == "claude"


def test_explicit_executable_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式传 executable 时,忽略环境变量。"""
    monkeypatch.setenv("BENCH_AGENT", "claude")
    assert ClaudeCodeClient(executable="tclaude").executable == "tclaude"

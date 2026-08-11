"""Candidate command contract for a future execution stage."""

from typing import Any, Literal

from pydantic import Field

from schemas.job_spec import Identifier, StrictModel


class Candidate(StrictModel):
    candidate_id: Identifier
    params: dict[str, Any] = Field(default_factory=dict)
    server_command: list[str] = Field(default_factory=list)
    benchmark_command: list[str] = Field(default_factory=list)
    benchmark_commands: list[list[str]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    expected_risk: Literal["low", "medium", "high"] = "medium"

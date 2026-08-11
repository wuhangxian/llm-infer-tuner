"""Search-plan contract produced by the planning stage."""

from typing import Any, Literal

from pydantic import Field, model_validator

from schemas.job_spec import Identifier, StrictModel


class ParameterAxis(StrictModel):
    values: list[Any] = Field(min_length=1)
    source: str = Field(min_length=1)
    reason: str | None = None
    risk: Literal["low", "medium", "high"] = "medium"
    monotonic: str | None = None


class Constraint(StrictModel):
    parameter: Identifier
    reason: str = Field(min_length=1)
    source: str = Field(min_length=1)
    severity: Literal["hard", "heuristic"] = "hard"


class SearchPolicy(StrictModel):
    strategy: Literal["baseline_first_bounded_product"]
    max_candidates: int = Field(gt=0)


class ParallelismCandidate(StrictModel):
    gpu_count: int = Field(gt=0)
    tp_size: int = Field(gt=0)
    pp_size: int = Field(gt=0)
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    risk: Literal["low", "medium", "high"] = "high"
    evidence_level: Literal["official", "measured", "heuristic"] = "heuristic"

    @model_validator(mode="after")
    def validate_gpu_factorization(self) -> "ParallelismCandidate":
        if self.tp_size * self.pp_size != self.gpu_count:
            raise ValueError(
                "gpu_count must equal tp_size * pp_size for a parallelism candidate"
            )
        return self


class SearchPlan(StrictModel):
    job_id: Identifier
    pinned: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Exact SGLang parameter names and fixed values only; do not put planner "
            "metadata or notes here."
        ),
    )
    search_space: dict[str, list[Any]] = Field(
        default_factory=dict,
        description=(
            "Exact SGLang parameter names mapped to bounded candidate values only; "
            "do not put planner metadata or notes here."
        ),
    )
    parallelism_candidates: list[ParallelismCandidate] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    axes: dict[str, ParameterAxis] = Field(default_factory=dict)
    search_policy: SearchPolicy
    notes: list[str] = Field(default_factory=list)

"""Input contract for one LLMOptAgent tuning job."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SLA(StrictModel):
    max_avg_ttft_ms: float = Field(gt=0)
    max_avg_tpot_ms: float = Field(gt=0)
    min_success_rate: float = Field(default=0.99, ge=0, le=1)


class SearchBudget(StrictModel):
    max_candidates: int = Field(gt=0)
    max_runtime_minutes: int = Field(gt=0)


class JobSpec(StrictModel):
    job_id: Identifier
    engine: Literal["sglang"]
    instance_type: Identifier
    model: Identifier
    workload: Identifier
    benchmark_method: Identifier
    sla: SLA
    search: SearchBudget

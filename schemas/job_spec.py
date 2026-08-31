"""Input contract for one llm-infer-tuner tuning job."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.parameter_contract import normalise_parameter_mapping

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class SLA(StrictModel):
    max_avg_ttft_ms: float = Field(gt=0)
    max_avg_tpot_ms: float = Field(gt=0)
    min_success_rate: float = Field(default=0.99, ge=0, le=1)


class BaselineConfig(StrictModel):
    """User-specified baseline params. Any key-value pair is allowed.
    AI will fill in missing params (default_flags, pin flags, etc.)
    and assemble a complete launch command.
    """
    model_config = ConfigDict(
        extra="allow",
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    @model_validator(mode="after")
    def validate_parameter_values(self) -> "BaselineConfig":
        """Use the candidate scalar contract for every free-form baseline flag."""
        self.__pydantic_extra__ = normalise_parameter_mapping(
            self.model_extra or {}, location="baseline"
        )
        return self


class SearchBudget(StrictModel):
    max_candidates: int = Field(gt=0)
    # Legacy input only. The executor deliberately does not impose a whole-job
    # deadline because a large candidate set may legitimately run for days.
    max_runtime_minutes: int | None = Field(default=None, gt=0)
    baseline: BaselineConfig | None = None
    baseline_threshold_pct: float = Field(
        default=0,
        ge=0,
        le=100,
        description=(
            "Annotate whether each candidate reaches baseline * (1 + pct/100); "
            "never filter rows"
        ),
    )


class JobSpec(StrictModel):
    job_id: Identifier
    engine: Literal["sglang"]
    gpu_model: Identifier
    gpu_count: int = Field(gt=0)
    gpu_memory_gb: float = Field(gt=0)
    model: Identifier
    image: Identifier
    workload: Identifier
    benchmark_method: Identifier
    sla: SLA
    search: SearchBudget

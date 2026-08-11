"""Deterministic checks for a Claude-generated SearchPlan."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from planner.reference_loader import PolicyCategories, SGLangReferences
from planner.spec_loader import LoadedSpec
from schemas.job_spec import JobSpec, StrictModel
from schemas.search_plan import SearchPlan


class RuleIssue(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["error", "warning"]
    field: str | None = None
    source: str | None = None


class RuleCheckResult(StrictModel):
    valid: bool
    errors: list[RuleIssue] = Field(default_factory=list)
    warnings: list[RuleIssue] = Field(default_factory=list)


class RuleChecker:
    """Apply local, deterministic constraints without starting an engine."""

    _integer_parameters = {
        "tp_size",
        "pp_size",
        "context_length",
        "chunked_prefill_size",
        "max_running_requests",
        "max_total_tokens",
    }
    _fraction_parameters = {"mem_fraction_static"}

    def check(
        self,
        job: JobSpec,
        hardware: LoadedSpec,
        model: LoadedSpec,
        workload: LoadedSpec,
        references: SGLangReferences,
        plan: SearchPlan,
    ) -> RuleCheckResult:
        errors: list[RuleIssue] = []
        warnings: list[RuleIssue] = []

        self._check_job_identity(job, plan, errors)
        categories = self._parameter_categories(references)
        self._check_parameter_names(plan, categories, errors)
        self._check_parameter_values(plan, errors)
        self._check_parallelism(plan, hardware, errors)
        self._check_tp_size(plan, hardware, errors)
        self._check_context_length(plan, model, workload, errors, warnings)
        self._check_budget(job, plan, errors)
        self._check_help_validation(plan, references, warnings)

        return RuleCheckResult(valid=not errors, errors=errors, warnings=warnings)

    @staticmethod
    def _check_job_identity(job: JobSpec, plan: SearchPlan, errors: list[RuleIssue]) -> None:
        if plan.job_id != job.job_id:
            errors.append(
                RuleIssue(
                    code="job_id_mismatch",
                    message=(
                        f"SearchPlan job_id={plan.job_id!r} does not match "
                        f"JobSpec {job.job_id!r}"
                    ),
                    severity="error",
                    field="job_id",
                )
            )

    @staticmethod
    def _parameter_categories(references: SGLangReferences) -> dict[str, str]:
        categories: dict[str, str] = {}
        for category in PolicyCategories.model_fields:
            for parameter in getattr(references.policy.policy, category):
                categories[parameter] = category
        return categories

    @staticmethod
    def _check_parameter_names(
        plan: SearchPlan,
        categories: dict[str, str],
        errors: list[RuleIssue],
    ) -> None:
        for parameter in [*plan.pinned, *plan.search_space]:
            category = categories.get(parameter)
            if category is None:
                errors.append(
                    RuleIssue(
                        code="unknown_parameter",
                        message=(
                            f"Parameter {parameter!r} is not present in "
                            "SGLang parameter policy"
                        ),
                        severity="error",
                        field=parameter,
                    )
                )
            elif parameter in plan.search_space and category in {
                "usually_pinned",
                "model_required_pinned",
            }:
                errors.append(
                    RuleIssue(
                        code="pinned_parameter_in_search_space",
                        message=(
                            f"Parameter {parameter!r} belongs to {category} "
                            "and should be pinned"
                        ),
                        severity="error",
                        field=parameter,
                    )
                )

    def _check_parameter_values(self, plan: SearchPlan, errors: list[RuleIssue]) -> None:
        for parameter, value in plan.pinned.items():
            self._check_value(parameter, value, errors, f"pinned.{parameter}")
        for parameter, values in plan.search_space.items():
            if not values:
                errors.append(
                    RuleIssue(
                        code="empty_search_axis",
                        message="Search-space axes must contain at least one value",
                        severity="error",
                        field=f"search_space.{parameter}",
                    )
                )
            for value in values:
                self._check_value(parameter, value, errors, f"search_space.{parameter}")

    @staticmethod
    def _check_parallelism(
        plan: SearchPlan,
        hardware: LoadedSpec,
        errors: list[RuleIssue],
    ) -> None:
        if not plan.parallelism_candidates:
            return
        gpu_count = hardware.data.get("gpu", {}).get("count")
        if not isinstance(gpu_count, int) or gpu_count <= 0:
            return
        for index, candidate in enumerate(plan.parallelism_candidates):
            field = f"parallelism_candidates[{index}]"
            if candidate.gpu_count > gpu_count:
                errors.append(
                    RuleIssue(
                        code="parallelism_exceeds_gpu_count",
                        message=(
                            f"{field}.gpu_count={candidate.gpu_count} exceeds "
                            f"available GPU count={gpu_count}"
                        ),
                        severity="error",
                        field=field,
                    )
                )
        conflicting_axes = {"tp_size", "pp_size"} & plan.search_space.keys()
        for parameter in sorted(conflicting_axes):
            errors.append(
                RuleIssue(
                    code="parallelism_axis_conflict",
                    message=(
                        f"{parameter!r} must not be an independent search axis when "
                        "parallelism_candidates are present"
                    ),
                    severity="error",
                    field=f"search_space.{parameter}",
                )
            )

    def _check_value(
        self,
        parameter: str,
        value: Any,
        errors: list[RuleIssue],
        field: str,
    ) -> None:
        if parameter in self._integer_parameters:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(
                    RuleIssue(
                        code="invalid_integer_parameter",
                        message=f"{parameter} must be a positive integer, got {value!r}",
                        severity="error",
                        field=field,
                    )
                )
        elif parameter in self._fraction_parameters:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                errors.append(
                    RuleIssue(
                        code="invalid_fraction_parameter",
                        message=f"{parameter} must be in (0, 1], got {value!r}",
                        severity="error",
                        field=field,
                    )
                )

    @staticmethod
    def _check_tp_size(
        plan: SearchPlan,
        hardware: LoadedSpec,
        errors: list[RuleIssue],
    ) -> None:
        gpu_count = hardware.data.get("gpu", {}).get("count")
        if not isinstance(gpu_count, int) or gpu_count <= 0:
            errors.append(
                RuleIssue(
                    code="invalid_gpu_count",
                    message="HardwareSpec must provide a positive integer gpu.count",
                    severity="error",
                    field="hardware.gpu.count",
                )
            )
            return

        values = []
        if "tp_size" in plan.pinned:
            values.append(("pinned.tp_size", plan.pinned["tp_size"]))
        values.extend(
            (f"search_space.tp_size[{index}]", value)
            for index, value in enumerate(plan.search_space.get("tp_size", []))
        )
        for field, value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value > gpu_count:
                errors.append(
                    RuleIssue(
                        code="tp_size_exceeds_gpu_count",
                        message=f"tp_size={value} exceeds available GPU count={gpu_count}",
                        severity="error",
                        field=field,
                    )
                )

    @staticmethod
    def _check_context_length(
        plan: SearchPlan,
        model: LoadedSpec,
        workload: LoadedSpec,
        errors: list[RuleIssue],
        warnings: list[RuleIssue],
    ) -> None:
        input_length = workload.data.get("input_tokens", {}).get("value")
        output_length = workload.data.get("output_tokens", {}).get("value")
        if not isinstance(input_length, int) or not isinstance(output_length, int):
            errors.append(
                RuleIssue(
                    code="invalid_workload_lengths",
                    message=(
                        "WorkloadSpec must provide integer input_tokens.value "
                        "and output_tokens.value"
                    ),
                    severity="error",
                    field="workload.input_tokens/output_tokens",
                )
            )
            return
        required_length = input_length + output_length

        context_values: list[tuple[str, Any]] = []
        if "context_length" in plan.pinned:
            context_values.append(("pinned.context_length", plan.pinned["context_length"]))
        context_values.extend(
            (f"search_space.context_length[{index}]", value)
            for index, value in enumerate(plan.search_space.get("context_length", []))
        )
        if not context_values:
            warnings.append(
                RuleIssue(
                    code="context_length_not_explicit",
                    message=(
                        "SearchPlan does not explicitly set context_length; "
                        "the engine/model context default must be verified before execution, "
                        "but workload length alone does not require shrinking the model context"
                    ),
                    severity="warning",
                    field="context_length",
                )
            )
            return

        model_max = model.data.get("context", {}).get("maximum_context_length")
        for field, value in context_values:
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            if value < required_length:
                errors.append(
                    RuleIssue(
                        code="context_length_too_short",
                        message=(
                            f"{field}={value} is below required "
                            f"input+output length={required_length}"
                        ),
                        severity="error",
                        field=field,
                    )
                )
            if isinstance(model_max, int) and value > model_max:
                errors.append(
                    RuleIssue(
                        code="context_length_exceeds_model_limit",
                        message=f"{field}={value} exceeds model maximum_context_length={model_max}",
                        severity="error",
                        field=field,
                    )
                )

    @staticmethod
    def _check_budget(job: JobSpec, plan: SearchPlan, errors: list[RuleIssue]) -> None:
        if plan.search_policy.max_candidates > job.search.max_candidates:
            errors.append(
                RuleIssue(
                    code="candidate_budget_exceeded",
                    message=(
                        f"SearchPlan max_candidates={plan.search_policy.max_candidates} exceeds "
                        f"JobSpec max_candidates={job.search.max_candidates}"
                    ),
                    severity="error",
                    field="search_policy.max_candidates",
                )
            )

    @staticmethod
    def _check_help_validation(
        plan: SearchPlan,
        references: SGLangReferences,
        warnings: list[RuleIssue],
    ) -> None:
        for parameter in plan.search_space:
            rule = references.policy.parameters.get(parameter)
            if rule is not None and rule.requires_target_help_validation:
                warnings.append(
                    RuleIssue(
                        code="target_help_validation_required",
                        message=(
                            f"Parameter {parameter!r} must be checked against the target "
                            "SGLang launch_server --help"
                        ),
                        severity="warning",
                        field=f"search_space.{parameter}",
                        source=rule.source,
                    )
                )

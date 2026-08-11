"""Validate and apply JobSpec-level limits to an AI-generated SearchPlan."""

from pydantic import ValidationError

from schemas.job_spec import JobSpec
from schemas.search_plan import SearchPlan


class PlanValidationError(ValueError):
    """Raised when a SearchPlan is malformed or violates the JobSpec budget."""


class PlanValidator:
    """Validate structured planner output before it reaches candidate generation."""

    def validate(self, payload: object, job: JobSpec) -> SearchPlan:
        try:
            plan = SearchPlan.model_validate(payload)
        except ValidationError as exc:
            raise PlanValidationError(f"Invalid SearchPlan: {exc}") from exc

        if plan.job_id != job.job_id:
            raise PlanValidationError(
                f"SearchPlan job_id={plan.job_id!r} does not match JobSpec job_id={job.job_id!r}"
            )
        if plan.search_policy.max_candidates > job.search.max_candidates:
            raise PlanValidationError(
                "SearchPlan max_candidates exceeds JobSpec max_candidates: "
                f"{plan.search_policy.max_candidates} > {job.search.max_candidates}"
            )
        return plan

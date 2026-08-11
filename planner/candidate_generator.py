"""Generate a bounded, deterministic candidate list from a SearchPlan."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any, Literal, cast

from schemas.candidate import Candidate
from schemas.search_plan import SearchPlan

Risk = Literal["low", "medium", "high"]


class CandidateGenerationError(ValueError):
    """Raised when a SearchPlan cannot be expanded into candidates."""


class CandidateGenerator:
    """Expand search axes in declaration order with baseline first."""

    _risk_rank = {"low": 0, "medium": 1, "high": 2}

    def generate(
        self,
        plan: SearchPlan,
        candidate_filter: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[Candidate]:
        if plan.parallelism_candidates and {
            "tp_size",
            "pp_size",
        } & plan.search_space.keys():
            raise CandidateGenerationError(
                "tp_size and pp_size must be expressed in parallelism_candidates, "
                "not as independent search axes"
            )

        axes = list(plan.search_space.items())
        for parameter, values in axes:
            if not values:
                raise CandidateGenerationError(
                    f"Search-space axis {parameter!r} must contain at least one value"
                )

        candidates: list[Candidate] = []
        topologies = plan.parallelism_candidates or [None]
        combinations = list(product(*(values for _, values in axes))) if axes else [()]
        for topology in topologies:
            for combination in combinations:
                params = dict(plan.pinned)
                if topology is not None:
                    params.update(
                        {
                            "tp_size": topology.tp_size,
                            "pp_size": topology.pp_size,
                        }
                    )
                params.update(dict(zip((name for name, _ in axes), combination, strict=True)))
                if candidate_filter is not None and not candidate_filter(params):
                    continue

                index = len(candidates) + 1
                candidates.append(
                    Candidate(
                        candidate_id=f"sglang-c{index:03d}",
                        params=params,
                        reasons=(
                            ["baseline candidate"]
                            if index == 1
                            else ["generated from bounded search space"]
                        ),
                        expected_risk=self._risk_for(plan, params),
                    )
                )
                if len(candidates) >= plan.search_policy.max_candidates:
                    return candidates
        return candidates

    def _risk_for(self, plan: SearchPlan, params: dict[str, Any]) -> Risk:
        risks = [
            plan.axes[parameter].risk
            for parameter in plan.search_space
            if parameter in params and parameter in plan.axes
        ]
        if not risks:
            return "medium"
        return cast(Risk, max(risks, key=self._risk_rank.__getitem__))

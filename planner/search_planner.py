"""Claude Code backed SearchPlan generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from planner.claude_code_client import ClaudeCodeClient
from planner.plan_validator import PlanValidator
from schemas.job_spec import JobSpec
from schemas.search_plan import SearchPlan

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "search_plan.md"


class SearchPlanner:
    """Ask Claude Code for a SearchPlan and validate it locally."""

    def __init__(
        self,
        client: ClaudeCodeClient,
        validator: PlanValidator | None = None,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        self.client = client
        self.validator = validator or PlanValidator()
        self.prompt_path = prompt_path

    def generate(
        self,
        job: JobSpec,
        add_dirs: Sequence[str | Path],
        allow_dangerous_permissions: bool = False,
    ) -> SearchPlan:
        prompt = self._build_prompt(job, add_dirs)
        payload = self.client.run(
            prompt=prompt,
            json_schema=SearchPlan.model_json_schema(),
            add_dirs=add_dirs,
            allow_dangerous_permissions=allow_dangerous_permissions,
        )
        return self.validator.validate(payload, job)

    def _build_prompt(self, job: JobSpec, add_dirs: Sequence[str | Path]) -> str:
        template = self.prompt_path.read_text(encoding="utf-8")
        directories = "\n".join(f"- {directory}" for directory in add_dirs)
        return template.replace(
            "{{JOB_SPEC_JSON}}",
            json.dumps(job.model_dump(mode="json"), ensure_ascii=False, indent=2),
        ).replace("{{ALLOWED_DIRECTORIES}}", directories)

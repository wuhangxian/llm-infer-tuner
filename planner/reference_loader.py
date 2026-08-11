"""Load and validate version-sensitive SGLang reference metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field, ValidationError

from schemas.job_spec import StrictModel

ReferenceAuthority = Literal["official", "secondary"]
ParameterCategory = Literal[
    "searchable_first_pass",
    "usually_pinned",
    "conditional_or_model_specific",
    "model_required_pinned",
]
ModelT = TypeVar("ModelT", bound=StrictModel)


class ReferenceSource(StrictModel):
    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    authority: ReferenceAuthority
    purpose: str = Field(min_length=1)


class RuntimeValidation(StrictModel):
    source_of_truth: Literal["target_environment_help"]
    required_commands: list[str] = Field(min_length=1)
    rule: str = Field(min_length=1)


class RefreshPolicy(StrictModel):
    copy_full_documents: bool
    refresh_when: list[str] = Field(min_length=1)


class SourceRegistry(StrictModel):
    schema_version: int = Field(gt=0)
    engine: Literal["sglang"]
    sources: list[ReferenceSource] = Field(min_length=1)
    runtime_validation: RuntimeValidation
    refresh_policy: RefreshPolicy


class ParameterRule(StrictModel):
    category: ParameterCategory
    source: str = Field(min_length=1)
    risk: Literal["low", "medium", "high"]
    rationale: str = Field(min_length=1)
    requires_target_help_validation: bool = True


class PolicyCategories(StrictModel):
    searchable_first_pass: list[str] = Field(min_length=1)
    usually_pinned: list[str] = Field(min_length=1)
    conditional_or_model_specific: list[str] = Field(min_length=1)
    model_required_pinned: list[str] = Field(min_length=1)


class ParameterPolicy(StrictModel):
    schema_version: int = Field(gt=0)
    engine: Literal["sglang"]
    policy: PolicyCategories
    parameters: dict[str, ParameterRule] = Field(min_length=1)
    constraints: list[str] = Field(min_length=1)
    evidence_required: list[str] = Field(min_length=1)


class SGLangReferences(StrictModel):
    sources: SourceRegistry
    policy: ParameterPolicy


class ReferenceLoader:
    """Load the SGLang source registry and parameter policy from one directory."""

    def __init__(self, references_root: Path) -> None:
        self.references_root = Path(references_root)

    def load_sglang(self) -> SGLangReferences:
        sources = self._load_json("sources.json", SourceRegistry)
        policy = self._load_json("parameter_policy.json", ParameterPolicy)
        self._validate_parameter_categories(policy)
        return SGLangReferences(sources=sources, policy=policy)

    def _load_json(self, filename: str, model_type: type[ModelT]) -> ModelT:
        path = self.references_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Reference file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return model_type.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Invalid reference file {path}: {exc}") from exc

    @staticmethod
    def _validate_parameter_categories(policy: ParameterPolicy) -> None:
        categories = {
            category: set(getattr(policy.policy, category))
            for category in PolicyCategories.model_fields
        }
        for parameter_name, rule in policy.parameters.items():
            if parameter_name not in categories[rule.category]:
                raise ValueError(
                    f"Parameter {parameter_name!r} metadata category {rule.category!r} "
                    "does not match parameter policy categories"
                )

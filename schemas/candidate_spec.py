"""Strict, fail-closed contracts for generated server candidates."""

from __future__ import annotations

import json
import math
import re
import shlex
from copy import deepcopy
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from schemas.job_spec import Identifier, SearchBudget, StrictModel

_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SHELL_CONTROL = re.compile(r"[;|&<>`\r\n()]|\$(?!\{MODEL_PATH\})")
_RUNTIME_OWNED = frozenset({"model_path", "host", "port"})
_MAMBA_ALIASES = (
    "mamba_radix_cache_strategy",
    "mamba_scheduler_strategy",
    "mamba-radix-cache-strategy",
    "mamba-scheduler-strategy",
)


def _normalise_key(key: str) -> str:
    return key.replace("-", "_")


def _json_scalar(value: object, *, location: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{location} must be a finite scalar JSON value")


def _parse_legacy_command(cmd: str) -> dict[str, object]:
    """Parse the intentionally tiny legacy command grammar without a shell."""
    if not cmd.strip():
        return {}
    if _SHELL_CONTROL.search(cmd):
        raise ValueError("legacy cmd contains shell control syntax")
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError as exc:
        raise ValueError(f"legacy cmd cannot be parsed: {exc}") from exc
    if tokens[:3] != ["python", "-m", "sglang.launch_server"]:
        raise ValueError("legacy cmd must start exactly with python -m sglang.launch_server")

    flags: dict[str, object] = {}
    index = 3
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--") or token == "--":
            raise ValueError(f"legacy cmd has unsupported positional token {token!r}")
        raw_key, separator, raw_value = token[2:].partition("=")
        key = _normalise_key(raw_key)
        if not _PARAMETER_NAME.fullmatch(key):
            raise ValueError(f"legacy cmd has unsafe flag {token!r}")
        if key in flags and key != "disable_radix_cache":
            raise ValueError(f"legacy cmd repeats flag --{raw_key}")
        if separator:
            value: object = raw_value
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            index += 1
        else:
            value = True
        if key == "disable_radix_cache" and value is not True:
            raise ValueError("candidate must not enable Radix cache")
        flags[key] = value
        index += 1
    return flags


def _canonical_cmd(cmd: str, flags: dict[str, object]) -> str:
    """Render a legacy command with one canonical Radix-off flag."""
    tokens = shlex.split(cmd, posix=True)
    rendered = tokens[:3]
    index = 3
    while index < len(tokens):
        token = tokens[index]
        raw_key, separator, _ = token[2:].partition("=")
        key = _normalise_key(raw_key)
        if key == "disable_radix_cache":
            if not separator and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                index += 1
        else:
            rendered.append(token)
            if not separator and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                rendered.append(tokens[index + 1])
                index += 1
        index += 1
    rendered.append("--disable-radix-cache")
    return shlex.join(rendered)


def _command_value(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


class CandidateSpec(StrictModel):
    """One candidate, with an optional legacy command constrained to an argv grammar."""

    id: Identifier
    params: dict[str, Any] = Field(default_factory=dict)
    cmd: str | None = None
    reasons: list[str] = Field(default_factory=list)

    _reserved_param_keys: ClassVar[frozenset[str]] = _RUNTIME_OWNED

    @field_validator("params")
    @classmethod
    def validate_and_normalise_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        normalised: dict[str, Any] = {}
        radix_values: list[object] = []
        for raw_key, value in params.items():
            key = _normalise_key(raw_key)
            if not _PARAMETER_NAME.fullmatch(key):
                raise ValueError(f"unsafe candidate parameter name {raw_key!r}")
            _json_scalar(value, location=f"candidate parameter {raw_key!r}")
            if key in normalised and normalised[key] != value:
                raise ValueError(f"conflicting candidate parameter spellings for {key!r}")
            if key == "disable_radix_cache":
                radix_values.append(value)
                continue
            if key in {"radix_cache", "prefix_cache", "enable_radix_cache", "enable_prefix_cache"}:
                raise ValueError("candidate must not enable Radix cache")
            if key in _RUNTIME_OWNED:
                # These values are executor-owned facts, not tuning decisions.
                continue
            if key == "is_baseline" and type(value) is not bool:
                raise ValueError("candidate parameter 'is_baseline' must be a boolean")
            normalised[key] = value
        if any(value is not True for value in radix_values):
            raise ValueError("candidate must not enable Radix cache; use bare disable-radix-cache")
        normalised["disable_radix_cache"] = True
        for field in ("tp_size", "ep_size", "dp_size"):
            if field in normalised and (
                type(normalised[field]) is not int or normalised[field] <= 0
            ):
                raise ValueError(f"candidate parameter {field!r} must be a positive integer")
        return normalised

    @field_validator("cmd")
    @classmethod
    def validate_command(cls, cmd: str | None) -> str | None:
        if cmd is None:
            return None
        flags = _parse_legacy_command(cmd)
        for key, value in flags.items():
            if key in {"radix_cache", "prefix_cache", "enable_radix_cache", "enable_prefix_cache"}:
                raise ValueError("candidate must not enable Radix cache")
            if key == "disable_radix_cache" and value is not True:
                raise ValueError("candidate must not enable Radix cache")
        return _canonical_cmd(cmd, flags)

    @model_validator(mode="after")
    def validate_command_agreement(self) -> CandidateSpec:
        if self.cmd is None:
            if not self.params:
                raise ValueError("candidate requires params or a legacy cmd")
            return self
        flags = _parse_legacy_command(self.cmd)
        command_params = {
            key: value
            for key, value in flags.items()
            if key not in _RUNTIME_OWNED and key != "disable_radix_cache"
        }
        params = {
            key: value
            for key, value in self.params.items()
            if key not in {"disable_radix_cache", "is_baseline", *_MAMBA_ALIASES}
        }
        if set(command_params) != set(params):
            raise ValueError("legacy cmd and params disagree on tuning flags")
        for key, value in params.items():
            if _command_value(command_params[key]) != _command_value(value):
                raise ValueError(f"legacy cmd and params disagrees for {key!r}")
        return self

    @property
    def is_baseline(self) -> bool:
        return self.params.get("is_baseline") is True

    @property
    def requested_mamba_radix_cache_strategy(self) -> str | None:
        for key in _MAMBA_ALIASES:
            value = self.params.get(_normalise_key(key))
            if value is not None:
                return str(value)
        return None

    @property
    def effective_mamba_radix_cache_strategy(self) -> str:
        return "inactive(radix_off)"

    @property
    def effective_params(self) -> dict[str, Any]:
        params = deepcopy(self.params)
        params.pop("is_baseline", None)
        for key in _MAMBA_ALIASES:
            params.pop(_normalise_key(key), None)
        return {key: value for key, value in params.items() if value not in (False, None)}


class CandidateSet(StrictModel):
    """Candidate collection tied to the job's exact expected search cardinality."""

    candidates: list[CandidateSpec] = Field(min_length=1)
    max_candidates: int = Field(gt=0)
    baseline_configured: bool = False

    @classmethod
    def from_candidates(
        cls, candidates: list[dict[str, Any]], *, search: SearchBudget
    ) -> CandidateSet:
        return cls.model_validate(
            {
                "candidates": candidates,
                "max_candidates": search.max_candidates,
                "baseline_configured": search.baseline is not None,
            }
        )

    @model_validator(mode="after")
    def validate_set(self) -> CandidateSet:
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        baselines = [candidate for candidate in self.candidates if candidate.is_baseline]
        if self.baseline_configured:
            if len(baselines) != 1 or baselines[0].id != "baseline":
                raise ValueError("configured baseline requires exactly one id='baseline' candidate")
        elif baselines:
            raise ValueError("candidate baseline is not configured by the job")
        expected = self.max_candidates + (1 if self.baseline_configured else 0)
        if len(self.candidates) != expected:
            raise ValueError(f"expected {expected} candidates, received {len(self.candidates)}")
        fingerprints = [
            json.dumps(candidate.effective_params, sort_keys=True, separators=(",", ":"))
            for candidate in self.candidates
        ]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("candidate effective configurations must be unique")
        return self


__all__ = ["CandidateSet", "CandidateSpec"]

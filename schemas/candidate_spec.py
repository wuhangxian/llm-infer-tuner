"""Strict, fail-closed contracts for generated server candidates."""

from __future__ import annotations

import json
import shlex
from copy import deepcopy
from typing import Any

from pydantic import Field, PrivateAttr, computed_field, model_validator

from schemas.job_spec import Identifier, SearchBudget, StrictModel
from schemas.parameter_contract import (
    MAMBA_STRATEGY_PARAMETERS,
    RUNTIME_PARAMETERS,
    CandidateParams,
    ParameterScalar,
    command_scalar,
    is_safe_parameter_name,
    normalise_parameter_name,
)


def _reject_shell_syntax(command: str) -> None:
    """Reject shell operators while retaining safely quoted scalar argument values."""
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote is None:
            if char == "#":
                raise ValueError("legacy cmd contains a shell comment")
            if char in "\r\n;|&<>`()":
                raise ValueError("legacy cmd contains shell control syntax")
            if char == "'":
                quote = char
            elif char == '"':
                quote = char
            elif char == "$":
                if not command.startswith("${MODEL_PATH}", index):
                    raise ValueError("legacy cmd contains shell substitution")
                index += len("${MODEL_PATH}") - 1
        elif quote == "'":
            if char == "'":
                quote = None
        else:
            if char == "`":
                raise ValueError("legacy cmd contains shell substitution")
            if char == '"':
                quote = None
            elif char == "$":
                if not command.startswith("${MODEL_PATH}", index):
                    raise ValueError("legacy cmd contains shell substitution")
                index += len("${MODEL_PATH}") - 1
        index += 1
    if quote is not None:
        raise ValueError("legacy cmd has an unterminated quote")


def _validate_runtime_flag(name: str, value: ParameterScalar) -> None:
    if name == "model_path" and value != "${MODEL_PATH}":
        raise ValueError("legacy cmd --model-path must be exactly ${MODEL_PATH}")
    if name == "host" and value != "0.0.0.0":
        raise ValueError("legacy cmd --host must be 0.0.0.0")
    if name == "port":
        if type(value) is not int or not 1 <= value <= 65535:
            raise ValueError("legacy cmd --port must be an integer in 1..65535")


def _parse_legacy_command(command: str) -> dict[str, ParameterScalar]:
    """Parse the approved legacy argv grammar without ever invoking a shell."""
    if not command.strip():
        return {}
    _reject_shell_syntax(command)
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"legacy cmd cannot be parsed: {exc}") from exc
    if tokens[:3] != ["python", "-m", "sglang.launch_server"]:
        raise ValueError("legacy cmd must start exactly with python -m sglang.launch_server")

    flags: dict[str, ParameterScalar] = {}
    index = 3
    while index < len(tokens):
        token = tokens[index]
        if token == "-p":
            raw_name, separator, raw_value = "port", "", ""
        elif token.startswith("-p="):
            raw_name, separator, raw_value = "port", "=", token[3:]
        elif token.startswith("--") and token != "--":
            raw_name, separator, raw_value = token[2:].partition("=")
        else:
            raise ValueError(f"legacy cmd has unsupported positional token {token!r}")
        name = normalise_parameter_name(raw_name)
        if not is_safe_parameter_name(name):
            raise ValueError(f"legacy cmd has unsafe flag --{raw_name}")
        if name in flags and name not in {"disable_radix_cache", "port"}:
            raise ValueError(f"legacy cmd repeats flag --{raw_name}")
        has_value = bool(separator)
        value: str | bool
        if separator:
            value = raw_value
        elif (
            index + 1 < len(tokens)
            and not tokens[index + 1].startswith("--")
            and tokens[index + 1] != "-p"
            and not tokens[index + 1].startswith("-p=")
        ):
            value = tokens[index + 1]
            has_value = True
            index += 1
        else:
            value = True
        try:
            parsed_value = command_scalar(name, value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if name == "disable_radix_cache":
            if has_value or parsed_value is not True:
                raise ValueError("candidate must use bare --disable-radix-cache")
        if name in RUNTIME_PARAMETERS:
            _validate_runtime_flag(name, parsed_value)
        flags[name] = parsed_value
        index += 1
    return flags


def _mamba_strategy(values: dict[str, ParameterScalar]) -> str | None:
    strategies = [str(values[name]) for name in MAMBA_STRATEGY_PARAMETERS if name in values]
    if len(set(strategies)) > 1:
        raise ValueError("Mamba radix/scheduler strategy aliases disagree")
    return strategies[0] if strategies else None


def _agreement_values(values: dict[str, ParameterScalar]) -> dict[str, ParameterScalar]:
    return {
        name: value
        for name, value in values.items()
        if name not in RUNTIME_PARAMETERS
        and name not in MAMBA_STRATEGY_PARAMETERS
        and name not in {"disable_radix_cache", "is_baseline"}
        and value is not False
        and value is not None
    }


def _equal_values(left: ParameterScalar, right: ParameterScalar) -> bool:
    if type(left) is float and type(right) is float:
        return left == right
    return type(left) is type(right) and left == right


def _same_flag_values(
    left: dict[str, ParameterScalar], right: dict[str, ParameterScalar]
) -> bool:
    return set(left) == set(right) and all(
        _equal_values(left[name], right[name]) for name in left
    )


def _reject_radix_enablement(values: dict[str, ParameterScalar]) -> None:
    """Keep every candidate on the required Radix-off execution path."""
    for name, value in values.items():
        if name == "disable_radix_cache":
            if value is not True:
                raise ValueError("candidate disable_radix_cache must be true when supplied")
            continue
        if name in MAMBA_STRATEGY_PARAMETERS:
            continue
        if (
            name in {"radix_cache", "prefix_cache", "enable_prefix_cache"}
            or (name.startswith("enable_") and "cache" in name)
            or (name.endswith("_radix_cache") and not name.startswith("disable_"))
        ):
            raise ValueError("candidate must not request Radix/prefix cache enablement")


def _canonical_cmd(flags: dict[str, ParameterScalar]) -> str:
    """Emit only effective tuning flags plus one canonical Radix-off switch."""
    parts = ["python", "-m", "sglang.launch_server"]
    for name, value in flags.items():
        if name in RUNTIME_PARAMETERS or name in MAMBA_STRATEGY_PARAMETERS:
            continue
        if name == "disable_radix_cache" or value is False or value is None:
            continue
        parts.append(f"--{name.replace('_', '-')}")
        if value is not True:
            parts.append(str(value))
    parts.append("--disable-radix-cache")
    return shlex.join(parts)


def _safe_execution_params(values: dict[str, ParameterScalar]) -> dict[str, ParameterScalar]:
    safe = {
        name: value
        for name, value in values.items()
        if name not in RUNTIME_PARAMETERS and name not in MAMBA_STRATEGY_PARAMETERS
    }
    safe["disable_radix_cache"] = True
    return safe


class CandidateSpec(StrictModel):
    """One candidate with typed params and a deliberately tiny legacy argv grammar."""

    id: Identifier
    params: CandidateParams = Field(default_factory=lambda: CandidateParams({}))
    cmd: str | None = None
    reasons: list[str] = Field(default_factory=list)
    _requested_params: CandidateParams = PrivateAttr(
        default_factory=lambda: CandidateParams({})
    )
    _requested_cmd: str | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def reject_externally_supplied_audit_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        reserved = {"requested_params", "requested_cmd"} & set(value)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"candidate audit fields are reserved: {names}")
        return value

    @model_validator(mode="after")
    def canonicalise_for_execution(self) -> CandidateSpec:
        requested = self.params.as_dict()
        self._requested_params = CandidateParams.model_validate(requested)
        self._requested_cmd = self.cmd
        command_flags = (
            _parse_legacy_command(self._requested_cmd)
            if self._requested_cmd is not None
            else {}
        )

        _reject_radix_enablement(requested)
        _reject_radix_enablement(command_flags)
        for runtime_name in RUNTIME_PARAMETERS:
            if runtime_name in requested:
                _validate_runtime_flag(runtime_name, requested[runtime_name])

        if self._requested_cmd is not None:
            expected = _agreement_values(requested)
            actual = _agreement_values(command_flags)
            if not _same_flag_values(expected, actual):
                missing = sorted(set(expected) ^ set(actual))
                if missing:
                    raise ValueError("legacy cmd disagrees with params on tuning flags")
                raise ValueError("legacy cmd disagrees with params on tuning flag values")
        requested_mamba = _mamba_strategy(requested)
        command_mamba = _mamba_strategy(command_flags)
        if (
            requested_mamba is not None
            and command_mamba is not None
            and requested_mamba != command_mamba
        ):
            raise ValueError("legacy cmd disagrees with params on Mamba strategy")
        if requested_mamba is None and command_mamba is not None:
            requested["mamba_radix_cache_strategy"] = command_mamba
            self._requested_params = CandidateParams.model_validate(requested)

        safe_requested = _safe_execution_params(requested)
        safe_supplied = _safe_execution_params(self.params.as_dict())
        if not _same_flag_values(safe_supplied, safe_requested):
            raise ValueError("params and requested_params disagree on effective tuning flags")
        self.params = CandidateParams.model_validate(safe_requested)
        self.cmd = _canonical_cmd(command_flags) if self._requested_cmd is not None else None
        return self

    @computed_field
    @property
    def requested_params(self) -> dict[str, ParameterScalar]:
        """Auditable requested values, never accepted as raw candidate input."""
        return self._requested_params.as_dict()

    @computed_field
    @property
    def requested_cmd(self) -> str | None:
        return self._requested_cmd

    @property
    def is_baseline(self) -> bool:
        return self._requested_params.get("is_baseline") is True

    @property
    def requested_mamba_radix_cache_strategy(self) -> str | None:
        return _mamba_strategy(self._requested_params.as_dict())

    @property
    def effective_mamba_radix_cache_strategy(self) -> str:
        return "inactive(radix_off)"

    @property
    def effective_params(self) -> dict[str, ParameterScalar]:
        params = deepcopy(self.params.as_dict())
        params.pop("is_baseline", None)
        return {
            name: value
            for name, value in params.items()
            if value is not False and value is not None
        }


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
        baseline_ids = [candidate for candidate in self.candidates if candidate.id == "baseline"]
        marked_baselines = [candidate for candidate in self.candidates if candidate.is_baseline]
        if self.baseline_configured:
            if len(baseline_ids) != 1 or len(marked_baselines) != 1:
                raise ValueError("configured baseline requires exactly one id='baseline' marker")
            if baseline_ids[0] is not marked_baselines[0]:
                raise ValueError("configured baseline id and is_baseline marker must match")
        elif baseline_ids or marked_baselines:
            raise ValueError("candidate baseline is not configured by the job")
        expected = self.max_candidates + (1 if self.baseline_configured else 0)
        if len(self.candidates) != expected:
            raise ValueError(f"expected {expected} candidates, received {len(self.candidates)}")
        fingerprints: dict[str, str] = {}
        for candidate in self.candidates:
            fingerprint = json.dumps(
                candidate.effective_params, sort_keys=True, separators=(",", ":")
            )
            original_id = fingerprints.get(fingerprint)
            if original_id is not None:
                raise ValueError(
                    f"{candidate.id} duplicates {original_id}: "
                    "candidate effective configurations must be unique"
                )
            fingerprints[fingerprint] = candidate.id
        return self


__all__ = ["CandidateParams", "CandidateSet", "CandidateSpec"]

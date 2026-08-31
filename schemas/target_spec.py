"""Input contract for one remote benchmark target."""

import re
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator

from schemas.job_spec import Identifier, StrictModel

ConnectionString = Annotated[str, Field(min_length=1, max_length=255)]
AbsolutePathString = Annotated[str, Field(min_length=1, max_length=4096)]
EnvironmentVariable = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
]
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_CONNECTION_METACHARACTERS = frozenset(";|&<>()$" + chr(96) + "\\'\"*?[]{}!")


def _safe_connection_value(value: str, *, field: str) -> str:
    if (
        value.startswith("-")
        or _CONTROL_CHARACTER.search(value)
        or any(character.isspace() for character in value)
        or any(character in _UNSAFE_CONNECTION_METACHARACTERS for character in value)
    ):
        raise ValueError(
            f"{field} must not start with '-', contain whitespace/control characters, "
            "or include shell metacharacters"
        )
    return value


def _absolute_posix_path(value: str, *, field: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return value
    if _CONTROL_CHARACTER.search(value) or not value.startswith("/"):
        raise ValueError(f"{field} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError(f"{field} must not contain '..'")
    return value


class TargetSpec(StrictModel):
    gpu_model: Identifier
    gpu_count: int = Field(gt=0)
    gpu_memory_gb: float = Field(gt=0)
    ssh_target: ConnectionString
    model_host_dir: AbsolutePathString
    model_container_path: AbsolutePathString
    image_ref: ConnectionString
    port: int = Field(ge=1, le=65535)
    ssh_password: SecretStr | None = Field(default=None, exclude=True, repr=False)
    ssh_password_env: EnvironmentVariable | None = None
    remote_outputs_dir: Annotated[str, Field(max_length=4096)] = ""
    exclusive_host: bool = False
    allow_cross_numa: bool = False

    @field_validator("ssh_target", "image_ref", mode="before")
    @classmethod
    def validate_connection_values(cls, value: object, info) -> object:
        if type(value) is not str:
            return value
        return _safe_connection_value(value, field=info.field_name)

    @field_validator("model_host_dir", "model_container_path")
    @classmethod
    def validate_model_paths(cls, value: str, info) -> str:
        return _absolute_posix_path(value, field=info.field_name)

    @field_validator("remote_outputs_dir")
    @classmethod
    def validate_remote_outputs_dir(cls, value: str) -> str:
        return _absolute_posix_path(value, field="remote_outputs_dir", allow_empty=True)

    @model_validator(mode="after")
    def validate_credential_source(self) -> "TargetSpec":
        if self.ssh_password is not None and len(self.ssh_password.get_secret_value()) > 4096:
            raise ValueError("ssh_password must not exceed 4096 characters")
        if self.ssh_password is not None and self.ssh_password_env:
            raise ValueError(
                "credential sources ssh_password and ssh_password_env are mutually exclusive"
            )
        return self

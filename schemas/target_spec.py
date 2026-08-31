"""Input contract for one remote benchmark target."""

import re
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator

from schemas.job_spec import Identifier, StrictModel

NonEmptyString = Annotated[str, Field(min_length=1)]
EnvironmentVariable = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
]
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


def _safe_connection_value(value: str, *, field: str) -> str:
    if value.startswith("-") or _CONTROL_CHARACTER.search(value):
        raise ValueError(f"{field} must not start with '-' or contain control characters")
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
    ssh_target: NonEmptyString
    model_host_dir: NonEmptyString
    model_container_path: NonEmptyString
    image_ref: NonEmptyString
    port: int = Field(ge=1, le=65535)
    ssh_password: SecretStr | None = Field(default=None, exclude=True, repr=False)
    ssh_password_env: EnvironmentVariable | None = None
    remote_outputs_dir: str = ""
    exclusive_host: bool = False

    @field_validator("ssh_target", "image_ref")
    @classmethod
    def validate_connection_values(cls, value: str, info) -> str:
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
        if self.ssh_password and self.ssh_password.get_secret_value() and self.ssh_password_env:
            raise ValueError(
                "credential sources ssh_password and ssh_password_env are mutually exclusive"
            )
        return self

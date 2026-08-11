"""Load hardware, model, and workload specs from the project registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

SpecKind = Literal["hardware", "model", "workload"]


class SpecLoaderError(RuntimeError):
    """Base error for spec discovery and loading failures."""


class SpecNotFoundError(SpecLoaderError):
    """Raised when a requested spec ID does not exist."""


class DuplicateSpecError(SpecLoaderError):
    """Raised when multiple files define the same spec ID."""


class SpecFormatError(SpecLoaderError):
    """Raised when a spec file cannot be parsed or has an invalid shape."""


class LoadedSpec(BaseModel):
    """A parsed spec with its identity and source path preserved."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: SpecKind
    spec_id: str = Field(min_length=1)
    path: Path
    data: dict[str, Any]


class SpecLoader:
    """Discover and load specs by their stable IDs."""

    _directories: ClassVar[dict[SpecKind, tuple[str, str]]] = {
        "hardware": ("hardware", "instance_type"),
        "model": ("models", "model_id"),
        "workload": ("workloads", "workload_id"),
    }

    def __init__(self, specs_root: Path) -> None:
        self.specs_root = Path(specs_root)
        if not self.specs_root.is_dir():
            raise SpecLoaderError(f"Specs directory does not exist: {self.specs_root}")

        self._index: dict[SpecKind, dict[str, Path]] = {
            "hardware": {},
            "model": {},
            "workload": {},
        }
        self._build_index()

    def load_hardware(self, spec_id: str) -> LoadedSpec:
        return self.load("hardware", spec_id)

    def load_model(self, spec_id: str) -> LoadedSpec:
        return self.load("model", spec_id)

    def load_workload(self, spec_id: str) -> LoadedSpec:
        return self.load("workload", spec_id)

    def load(self, kind: SpecKind, spec_id: str) -> LoadedSpec:
        try:
            path = self._index[kind][spec_id]
        except KeyError as exc:
            directory, id_field = self._directories[kind]
            raise SpecNotFoundError(
                f"No {kind} spec with {id_field}={spec_id!r} under {self.specs_root / directory}"
            ) from exc

        data = self._read_file(path)
        expected_field = self._directories[kind][1]
        if data.get(expected_field) != spec_id:
            raise SpecFormatError(
                f"Spec identity changed after indexing: {path} field {expected_field!r} "
                f"is {data.get(expected_field)!r}, expected {spec_id!r}"
            )
        return LoadedSpec(kind=kind, spec_id=spec_id, path=path, data=data)

    def _build_index(self) -> None:
        for kind, (directory, id_field) in self._directories.items():
            kind_dir = self.specs_root / directory
            if not kind_dir.is_dir():
                continue
            for path in sorted(kind_dir.iterdir()):
                if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                    continue
                data = self._read_file(path)
                spec_id = data.get(id_field)
                if not isinstance(spec_id, str) or not spec_id.strip():
                    raise SpecFormatError(
                        f"{path} must contain a non-empty string field {id_field!r}"
                    )
                if spec_id in self._index[kind]:
                    previous = self._index[kind][spec_id]
                    raise DuplicateSpecError(
                        f"Duplicate {kind} spec ID {spec_id!r}: {previous} and {path}"
                    )
                self._index[kind][spec_id] = path

    @staticmethod
    def _read_file(path: Path) -> dict[str, Any]:
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise SpecFormatError(f"Could not parse spec file {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise SpecFormatError(f"Spec file {path} must contain a JSON/YAML object")
        return data

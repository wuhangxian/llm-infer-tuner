from pathlib import Path

import pytest

from planner.spec_loader import DuplicateSpecError, SpecFormatError, SpecLoader, SpecNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loader_reads_hardware_spec_by_instance_type() -> None:
    loader = SpecLoader(PROJECT_ROOT / "specs")

    hardware = loader.load_hardware("GC50s.192XLARGE2304")

    assert hardware.spec_id == "GC50s.192XLARGE2304"
    assert hardware.data["gpu"]["model"] == "Pro5000"


def test_loader_reads_model_and_workload_specs() -> None:
    loader = SpecLoader(PROJECT_ROOT / "specs")

    model = loader.load_model("qwen36-27b-fp8")
    workload = loader.load_workload("random-32k-1k")

    assert model.data["quantization"]["method"] == "fp8"
    assert workload.data["input_tokens"]["value"] == 32768


def test_loader_reports_missing_spec() -> None:
    loader = SpecLoader(PROJECT_ROOT / "specs")

    with pytest.raises(SpecNotFoundError, match="does-not-exist"):
        loader.load_model("does-not-exist")


def test_loader_supports_yaml_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (tmp_path / "hardware").mkdir()
    (tmp_path / "workloads").mkdir()
    (models / "first.yaml").write_text("model_id: duplicate\nfamily: first\n", encoding="utf-8")
    (models / "second.json").write_text(
        '{"model_id": "duplicate", "family": "second"}', encoding="utf-8"
    )

    with pytest.raises(DuplicateSpecError, match="duplicate"):
        SpecLoader(tmp_path)


def test_loader_rejects_malformed_spec(tmp_path: Path) -> None:
    hardware = tmp_path / "hardware"
    hardware.mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "workloads").mkdir()
    (hardware / "broken.json").write_text('{"instance_type": ', encoding="utf-8")

    with pytest.raises(SpecFormatError, match="broken.json"):
        SpecLoader(tmp_path)

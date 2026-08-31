import pytest
from pydantic import ValidationError

from schemas.candidate_spec import CandidateSet, CandidateSpec
from schemas.job_spec import JobSpec
from schemas.target_spec import TargetSpec


def _valid_job() -> dict:
    return {
        "job_id": "job-1",
        "engine": "sglang",
        "gpu_model": "gpu-1",
        "gpu_count": 1,
        "gpu_memory_gb": 16,
        "model": "model-1",
        "image": "image-1",
        "workload": "workload-1",
        "benchmark_method": "method-1",
        "sla": {"max_avg_ttft_ms": 100, "max_avg_tpot_ms": 20},
        "search": {"max_candidates": 1},
    }


def _valid_target() -> dict:
    return {
        "gpu_model": "gpu-1",
        "gpu_count": 8,
        "gpu_memory_gb": 72,
        "ssh_target": "runner@example.test",
        "model_host_dir": "/models/example",
        "model_container_path": "/models/example",
        "image_ref": "registry.example/sglang:test",
        "port": 30000,
    }


def test_job_spec_accepts_minimal_valid_job() -> None:
    job = JobSpec.model_validate(
        {
            "job_id": "qwen36-27b-fp8_pro5000_8x72g_qa-chat-3.5k-1k",
            "engine": "sglang",
             "gpu_model": "pro5000",
             "gpu_count": 8,
             "gpu_memory_gb": 72,
            "model": "qwen36-27b-fp8",
            "image": "sglang-v0.5.10",
            "workload": "random-32k-1k",
            "benchmark_method": "sglang-bench-serving",
            "sla": {
                "max_avg_ttft_ms": 2000,
                "max_avg_tpot_ms": 80
            },
            "search": {
                "max_candidates": 30,
                "max_runtime_minutes": 180
            }
        }
    )

    assert job.engine == "sglang"
    assert job.search.max_candidates == 30


def test_target_spec_accepts_key_based_ssh_target() -> None:
    target = TargetSpec.model_validate(_valid_target())

    assert target.ssh_target == "runner@example.test"
    assert target.port == 30000


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_target_spec_rejects_out_of_range_ports(port: int) -> None:
    payload = _valid_target()
    payload["port"] = port

    with pytest.raises(ValidationError):
        TargetSpec.model_validate(payload)


def test_target_spec_accepts_password_environment_reference() -> None:
    payload = _valid_target()
    payload["ssh_password_env"] = "LLM_TUNER_TEST_SSH_PASSWORD"

    target = TargetSpec.model_validate(payload)

    assert target.ssh_password_env == "LLM_TUNER_TEST_SSH_PASSWORD"


def test_target_spec_accepts_current_optional_target_fields() -> None:
    payload = _valid_target()
    payload.update(
        {
            "ssh_password": "",
            "remote_outputs_dir": "",
            "exclusive_host": False,
        }
    )

    target = TargetSpec.model_validate(payload)

    assert target.ssh_password is not None
    assert target.ssh_password.get_secret_value() == ""
    assert target.remote_outputs_dir == ""
    assert target.exclusive_host is False


def test_target_spec_rejects_multiple_password_sources() -> None:
    payload = _valid_target()
    payload.update(
        {
            "ssh_password": "test-only-password",
            "ssh_password_env": "LLM_TUNER_TEST_SSH_PASSWORD",
        }
    )

    with pytest.raises(ValidationError, match="credential"):
        TargetSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ssh_target", "-oProxyCommand=bad"),
        ("ssh_target", "runner\nother"),
        ("image_ref", "registry.example/image\nother"),
        ("model_host_dir", "relative/models"),
        ("model_container_path", "relative/models"),
        ("remote_outputs_dir", "relative/outputs"),
    ],
)
def test_target_spec_rejects_unsafe_connection_and_path_values(field: str, value: str) -> None:
    payload = _valid_target()
    payload[field] = value

    with pytest.raises(ValidationError):
        TargetSpec.model_validate(payload)


def test_target_spec_excludes_plaintext_password_from_serialization() -> None:
    payload = _valid_target()
    payload["ssh_password"] = "test-only-password"

    target = TargetSpec.model_validate(payload)

    assert "ssh_password" not in target.model_dump()


def test_job_spec_rejects_unsupported_engine_and_unknown_fields() -> None:
    payload = {
        "job_id": "job-1",
        "engine": "vllm",
         "gpu_model": "gpu-1",
         "gpu_count": 1,
         "gpu_memory_gb": 16,
        "model": "model-1",
        "workload": "workload-1",
        "benchmark_method": "method-1",
        "sla": {"max_avg_ttft_ms": 100, "max_avg_tpot_ms": 20},
        "search": {"max_candidates": 1, "max_runtime_minutes": 1},
        "unexpected": True,
    }

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


def test_job_spec_exports_json_schema() -> None:
    assert "properties" in JobSpec.model_json_schema()


def test_job_has_no_required_overall_runtime_limit() -> None:
    payload = {
        "job_id": "long-job",
        "engine": "sglang",
        "gpu_model": "gpu-1",
        "gpu_count": 8,
        "gpu_memory_gb": 80,
        "model": "large-model",
        "image": "sglang",
        "workload": "workload",
        "benchmark_method": "method",
        "sla": {"max_avg_ttft_ms": 1000, "max_avg_tpot_ms": 100},
        "search": {"max_candidates": 100},
    }

    job = JobSpec.model_validate(payload)

    assert job.search.max_runtime_minutes is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("gpu_count",), "8"),
        (("gpu_count",), True),
        (("gpu_memory_gb",), "72"),
        (("sla", "max_avg_ttft_ms"), "2000"),
        (("search", "max_candidates"), "30"),
    ],
)
def test_job_spec_rejects_scalar_coercion(path: tuple[str, ...], value: object) -> None:
    payload = _valid_job()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_job_spec_rejects_nonfinite_numbers(value: float) -> None:
    payload = _valid_job()
    payload["gpu_memory_gb"] = value

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"nested": float("inf")}])
def test_job_spec_rejects_nonfinite_baseline_values(value: object) -> None:
    payload = _valid_job()
    payload["search"]["baseline"] = {"mem_fraction_static": value}

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


def test_job_spec_rejects_identifier_longer_than_128_characters() -> None:
    payload = _valid_job()
    payload["job_id"] = "j" * 129

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


def _candidate(candidate_id: str, **params: object) -> dict:
    tuning_flags = " ".join(
        f"--{key.replace('_', '-')} {value}"
        for key, value in params.items()
        if key
        not in {
            "is_baseline",
            "disable_radix_cache",
            "disable-radix-cache",
            "mamba_radix_cache_strategy",
        }
    )
    return {
        "id": candidate_id,
        "params": params,
        "cmd": (
            "python -m sglang.launch_server --model-path ${MODEL_PATH} "
            f"{tuning_flags}"
        ).rstrip(),
        "reasons": [],
    }


def _candidate_set(*candidates: dict, baseline_configured: bool = False) -> dict:
    return {
        "candidates": list(candidates),
        "max_candidates": 2,
        "baseline_configured": baseline_configured,
    }


@pytest.mark.parametrize(
    "payload",
    [
        _candidate_set(),
        _candidate_set(_candidate("c001"), _candidate("c001")),
        _candidate_set(_candidate("c001"), _candidate("c002"), _candidate("c003")),
        _candidate_set(
            _candidate("c001", tp_size=1), _candidate("c002", tp_size=1)
        ),
    ],
)
def test_candidate_set_rejects_empty_duplicate_or_wrong_count(payload: dict) -> None:
    with pytest.raises(ValidationError):
        CandidateSet.model_validate(payload)


def test_candidate_set_requires_one_baseline_only_when_job_configures_it() -> None:
    baseline = _candidate("baseline", is_baseline=True)
    candidates = _candidate_set(
        baseline, _candidate("c001", tp_size=1), _candidate("c002", tp_size=2),
        baseline_configured=True,
    )

    parsed = CandidateSet.model_validate(candidates)

    assert [candidate.id for candidate in parsed.candidates] == ["baseline", "c001", "c002"]

    candidates["candidates"].append(_candidate("second-baseline", is_baseline=True))
    with pytest.raises(ValidationError, match="baseline"):
        CandidateSet.model_validate(candidates)


def test_candidate_set_rejects_nonboolean_baseline_marker() -> None:
    payload = _candidate_set(_candidate("c001", is_baseline="true"), _candidate("c002"))

    with pytest.raises(ValidationError, match="is_baseline"):
        CandidateSet.model_validate(payload)


def test_candidate_spec_normalizes_radix_off_and_mamba_audit_metadata() -> None:
    candidate = CandidateSpec.model_validate(
        _candidate(
            "c001",
            tp_size=1,
            **{
                "disable-radix-cache": True,
                "disable_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
        )
    )

    assert candidate.params["disable_radix_cache"] is True
    assert "disable-radix-cache" not in candidate.params
    assert candidate.requested_mamba_radix_cache_strategy == "extra_buffer"
    assert candidate.effective_mamba_radix_cache_strategy == "inactive(radix_off)"


def test_candidate_spec_deduplicates_bare_radix_disable_flags() -> None:
    candidate = _candidate("c001")
    candidate["cmd"] += " --disable-radix-cache --disable-radix-cache"

    parsed = CandidateSpec.model_validate(candidate)

    assert parsed.cmd is not None
    assert parsed.cmd.split().count("--disable-radix-cache") == 1


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate("c001", disable_radix_cache=False),
        _candidate("c001", **{"disable-radix-cache": False}),
        _candidate("c001", radix_cache=True),
        _candidate("c001", enable_prefix_cache=True),
        {
            **_candidate("c001"),
            "cmd": (
                "python -m sglang.launch_server --model-path ${MODEL_PATH} "
                "--disable-radix-cache=0"
            ),
        },
        {**_candidate("c001"), "cmd": "python -m sglang.launch_server; id"},
        {**_candidate("c001"), "cmd": "python3 -m sglang.launch_server"},
    ],
)
def test_candidate_spec_rejects_radix_enablement_and_unsafe_legacy_command(candidate: dict) -> None:
    with pytest.raises(ValidationError):
        CandidateSpec.model_validate(candidate)


def test_candidate_spec_rejects_legacy_command_params_disagreement() -> None:
    candidate = _candidate("c001", tp_size=2)
    candidate["cmd"] = "python -m sglang.launch_server --model-path ${MODEL_PATH} --tp-size 1"

    with pytest.raises(ValidationError, match="disagrees"):
        CandidateSpec.model_validate(candidate)

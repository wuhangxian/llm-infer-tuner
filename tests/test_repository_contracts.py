from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_targets_have_no_plaintext_ssh_passwords() -> None:
    for path in sorted((ROOT / "input" / "targets").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not payload.get("ssh_password"), path
        env_name = payload.get("ssh_password_env")
        if env_name is not None:
            assert isinstance(env_name, str)
            assert env_name.isidentifier()


def test_secret_scanner_passes_tracked_tree() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_no_secrets.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_secret_scanner_rejects_plaintext_credential(tmp_path: Path) -> None:
    from scripts.check_no_secrets import scan_paths

    field = "ssh_" + "password"
    path = tmp_path / "target.json"
    path.write_text(json.dumps({field: "plain-text-secret"}), encoding="utf-8")
    findings = scan_paths([path])
    assert findings
    assert findings[0].path == path
    assert findings[0].line == 1
    assert "plain-text-secret" not in str(findings)


def test_jobs_are_jobspec_only() -> None:
    forbidden = {
        "ssh_target",
        "ssh_password",
        "ssh_password_env",
        "model_host_dir",
        "model_container_path",
        "port",
    }
    for path in sorted((ROOT / "input" / "jobs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert forbidden.isdisjoint(payload), path


def test_catalog_declared_totals_match_entries() -> None:
    expectations = {
        "gpu.yaml": "gpu_catalog",
        "models.yaml": "models",
        "workloads.yaml": "workloads",
        "sglang-images.yaml": "images",
    }
    for filename, section in expectations.items():
        payload = yaml.safe_load((ROOT / "catalogs" / filename).read_text(encoding="utf-8"))
        assert payload["total"] == len(payload[section]), filename


def test_ci_declares_reliability_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in (
        "pytest",
        "ruff check",
        "pyright",
        "bash -n",
        "check_no_secrets.py",
    ):
        assert command in workflow


def test_readme_documents_secret_migration_and_rotation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "ssh_password_env" in readme
    assert "LLM_INFER_TUNER_SSH_PASSWORD" in readme
    assert "rotate" in readme.lower()

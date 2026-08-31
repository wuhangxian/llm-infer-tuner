from __future__ import annotations

import math
from pathlib import Path

import pytest

from runners.reporting import (
    annotate_baseline_threshold,
    build_candidate_rows,
    render_candidate_preview,
    write_reports,
)


def test_baseline_threshold_is_annotation_not_filter() -> None:
    ranking = [
        {"candidate_id": "fast", "goodput_per_host": 130.0},
        {"candidate_id": "baseline", "goodput_per_host": 100.0},
        {"candidate_id": "slow", "goodput_per_host": 80.0},
    ]

    annotated = annotate_baseline_threshold(ranking, threshold_pct=20)

    assert [row["candidate_id"] for row in annotated] == ["fast", "baseline", "slow"]
    assert [row["rank"] for row in annotated] == [1, 2, 3]
    assert annotated[0]["beats_baseline_threshold"] is True
    assert annotated[1]["beats_baseline_threshold"] is False
    assert annotated[2]["beats_baseline_threshold"] is False
    assert annotated[0]["baseline_goodput_per_host"] == 100.0
    assert annotated[0]["threshold_goodput_per_host"] == 120.0
    assert annotated[0]["goodput_delta"] == 30.0
    assert annotated[0]["goodput_delta_pct"] == 30.0


def test_baseline_annotation_never_emits_nonfinite_derived_values() -> None:
    threshold_overflow = annotate_baseline_threshold(
        [
            {
                "candidate_id": "baseline",
                "goodput_per_host": float.fromhex("0x1.fffffffffffffp+1023"),
            },
            {"candidate_id": "c001", "goodput_per_host": 1.0},
        ],
        threshold_pct=20,
    )
    percentage_overflow = annotate_baseline_threshold(
        [
            {
                "candidate_id": "c001",
                "goodput_per_host": float.fromhex("0x1.fffffffffffffp+1023"),
            },
            {"candidate_id": "baseline", "goodput_per_host": 1e-308},
        ],
        threshold_pct=20,
    )

    assert threshold_overflow[0]["threshold_goodput_per_host"] is None
    assert threshold_overflow[0]["beats_baseline_threshold"] is False
    assert percentage_overflow[0]["goodput_delta_pct"] is None
    for rows in (threshold_overflow, percentage_overflow):
        for row in rows:
            for key in (
                "baseline_goodput_per_host",
                "threshold_goodput_per_host",
                "goodput_delta",
                "goodput_delta_pct",
            ):
                assert row[key] is None or math.isfinite(row[key])


def test_write_reports_rejects_nonfinite_json_without_replacing_old_files(
    tmp_path: Path,
) -> None:
    old_contents = {
        "ranking.json": "[]\n",
        "candidate_results.jsonl": "{}\n",
        "task_status.json": '{"status":"OLD"}\n',
    }
    for name, text in old_contents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_reports(
            tmp_path,
            ranking=[{"candidate_id": "c001", "goodput_per_host": 1.0}],
            candidate_rows=[{"candidate_id": "c001", "metric": math.inf}],
            task_status={"status": "PROVISIONAL"},
        )

    for name, text in old_contents.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == text


def test_report_preserves_requested_cache_params_and_shows_forced_effective_state() -> None:
    candidates = [{
        "id": "c001",
        "params": {
            "disable-radix-cache": False,
            "mamba-radix-cache-strategy": "extra_buffer",
        },
    }]
    rows = build_candidate_rows(
        candidates,
        {"c001": {"attempts": 1}},
        {},
        [],
        output_len=1024,
    )

    assert rows[0]["requested_params"]["mamba-radix-cache-strategy"] == "extra_buffer"
    assert rows[0]["effective_params"]["disable_radix_cache"] is True
    assert "disable-radix-cache" not in rows[0]["effective_params"]
    assert rows[0]["effective_params"]["mamba_cache_strategy"] == "inactive(radix_off)"
    preview = render_candidate_preview(rows)[0]
    assert 'requested_params={"disable-radix-cache":false' in preview
    assert (
        'effective_params={"disable_radix_cache":true,'
        '"mamba_cache_strategy":"inactive(radix_off)"}' in preview
    )


def test_report_uses_audit_requested_params_after_loader_strips_mamba() -> None:
    rows = build_candidate_rows(
        [
            {
                "id": "c001",
                "params": {"disable_radix_cache": True},
                "requested_params": {
                    "mamba_radix_cache_strategy": "no_buffer",
                    "mem_fraction_static": 0.82,
                },
                "cmd": "python -m sglang.launch_server --disable-radix-cache",
                "requested_cmd": (
                    "python -m sglang.launch_server "
                    "--mamba-radix-cache-strategy no_buffer"
                ),
            }
        ],
        {"c001": {"attempts": 1}},
        {},
        [],
        output_len=1024,
    )

    assert rows[0]["requested_params"]["mamba_radix_cache_strategy"] == "no_buffer"
    assert rows[0]["effective_params"]["mamba_cache_strategy"] == "inactive(radix_off)"
    assert "--mamba-radix-cache-strategy" in rows[0]["requested_command"]

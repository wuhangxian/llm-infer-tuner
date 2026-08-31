from __future__ import annotations

from runners.reporting import (
    annotate_baseline_threshold,
    build_candidate_rows,
    render_candidate_preview,
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

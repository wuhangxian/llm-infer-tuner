#!/usr/bin/env python3
"""Generate a human-readable best-config report from an executor output directory.

The executor deliberately keeps machine-readable reports in JSON/JSONL.  This
script is a separate, deterministic presentation layer so a report can be
regenerated from an existing benchmark without rerunning the remote job.

Typical usage::

    uv run python scripts/generate_best_config_md.py \
        outputs/<job_id>

The argument may be either ``outputs/<job_id>`` (containing ``results/``) or
``outputs/<job_id>/results``.  ``best_config.md`` is written to the directory
passed by the user.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the project declares PyYAML
    yaml = None  # type: ignore[assignment]


class ReportError(RuntimeError):
    """An actionable input/output error for the command-line tool."""


_REPORT_ONLY_PARAMS = {"is_baseline", "mamba_cache_strategy"}
_UNSPECIFIED = "未显式指定（使用 SGLang/镜像默认值）"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"JSON 格式错误: {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # One malformed evidence line should not prevent a useful summary
            # from being generated from the remaining rows.
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            raise ReportError(f"JSONL 第 {line_number} 行不是对象: {path}")
    return rows


def _load_ranking(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("ranking"), list):
        return [row for row in value["ranking"] if isinstance(row, dict)]
    raise ReportError(f"ranking.json 应为数组或包含 ranking 数组的对象: {path}")


def _load_candidates(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load both generated JSON and single-file JSONL candidate formats."""
    if path is None or not path.exists():
        return [], {}

    text = path.read_text(encoding="utf-8")
    candidates: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for row in _read_jsonl(path):
            if isinstance(row.get("_meta"), dict):
                metadata = dict(row["_meta"])
            elif row.get("id") is not None:
                candidates.append(row)
        return candidates, metadata

    if isinstance(value, dict):
        if isinstance(value.get("_meta"), dict):
            metadata = dict(value["_meta"])
        if isinstance(value.get("candidates"), list):
            candidates = [row for row in value["candidates"] if isinstance(row, dict)]
        elif value.get("id") is not None:
            candidates = [value]
    elif isinstance(value, list):
        candidates = [row for row in value if isinstance(row, dict) and row.get("id")]
    else:
        raise ReportError(f"候选配置应为 JSON 对象、数组或 JSONL: {path}")
    return candidates, metadata


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 2) -> str:
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{number:,.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    number = _to_float(value)
    return "—" if number is None else f"{number:.{digits}f}%"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _inline_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text.replace("|", "\\|").replace("\n", " ")


def _safe_text(value: Any) -> str:
    return str(value).replace("```", "'''\n").strip()


def _qualified(row: dict[str, Any]) -> bool:
    """Mirror the ranker's notion of a usable winner without rerunning ranking."""
    concurrency = row.get("best_concurrency")
    goodput = _to_float(row.get("goodput_per_host"))
    return concurrency not in (None, "", 0) and goodput is not None and goodput > 0


def _find_job(project_root: Path, job_id: str) -> dict[str, Any]:
    jobs_dir = project_root / "input" / "jobs"
    if not jobs_dir.exists():
        return {}
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("job_id") == job_id:
            return value
    return {}


def _load_workload(project_root: Path, workload_id: str | None) -> dict[str, Any]:
    if not workload_id or yaml is None:
        return {}
    path = project_root / "catalogs" / "workloads.yaml"
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    workloads = value.get("workloads", {}) if isinstance(value, dict) else {}
    result = workloads.get(workload_id, {}) if isinstance(workloads, dict) else {}
    return result if isinstance(result, dict) else {}


def _resolve_layout(output_dir: Path) -> tuple[Path, Path]:
    """Return (results_dir, report_dir), accepting job or results directory."""
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise ReportError(f"output 目录不存在: {output_dir}")
    if (output_dir / "ranking.json").is_file():
        return output_dir, output_dir
    nested_results = output_dir / "results"
    if (nested_results / "ranking.json").is_file():
        return nested_results, output_dir
    raise ReportError(
        f"在 {output_dir} 或 {nested_results} 中都没有找到 ranking.json；"
        "请传入具体的 outputs/<job_id> 或其 results 子目录"
    )


def _candidate_config_path(
    report_dir: Path, results_dir: Path, explicit: Path | None
) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for path in (report_dir / "configs.jsonl", results_dir.parent / "configs.jsonl"):
        if path.is_file():
            return path
    return None


def _safe_metadata(
    job: dict[str, Any],
    config_meta: dict[str, Any],
    task_status: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Keep report metadata useful while never copying SSH credentials."""
    merged = dict(config_meta)
    merged.update(job)
    merged.setdefault("job_id", task_status.get("job_id") or job_id)
    # Single-file configs use ``image_ref`` and ``model_host_dir`` while JobSpec
    # uses ``image`` and ``model``.  Normalize those names so reports generated
    # directly from a ``_meta`` config do not lose the most useful identity data.
    if not merged.get("image") and merged.get("image_ref"):
        merged["image"] = merged["image_ref"]
    if not merged.get("model") and merged.get("model_host_dir"):
        model_path = str(merged["model_host_dir"]).rstrip("/")
        if model_path:
            merged["model"] = Path(model_path).name
    allowed = {
        "job_id",
        "engine",
        "gpu_model",
        "gpu_count",
        "gpu_memory_gb",
        "model",
        "image",
        "workload",
        "benchmark_method",
        "sla",
        "search",
    }
    return {key: merged[key] for key in allowed if key in merged}


def _fallback_command(params: dict[str, Any]) -> str:
    parts = ["python -m sglang.launch_server", "--model-path ${MODEL_PATH}"]
    for key, value in params.items():
        if key in {"is_baseline", "model_path", "host", "port", "mamba_cache_strategy"}:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            parts.append(flag)
        elif value not in (False, None):
            parts.extend([flag, str(value)])
    parts.extend(["--host", "0.0.0.0", "--port", "30000"])
    return " ".join(parts)


def _param_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    differences: list[str] = []
    for key in sorted((set(left) | set(right)) - _REPORT_ONLY_PARAMS):
        if left.get(key) != right.get(key):
            left_value = _UNSPECIFIED if key not in left else _inline_json(left[key])
            right_value = _UNSPECIFIED if key not in right else _inline_json(right[key])
            differences.append(f"`{key}`：最佳={left_value}；对比={right_value}")
    return differences


def _check(value: Any, limit: Any, operator: str) -> str:
    actual = _to_float(value)
    target = _to_float(limit)
    if actual is None or target is None:
        return "—"
    passed = actual <= target if operator == "le" else actual >= target
    return "通过" if passed else "不通过"


def render_markdown(
    *,
    report_dir: Path,
    results_dir: Path,
    ranking: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    task_status: dict[str, Any],
    metadata: dict[str, Any],
    workload: dict[str, Any],
    top_k: int,
) -> str:
    rows_by_id = {str(row.get("candidate_id")): row for row in candidate_rows}
    configs_by_id = {str(row.get("id")): row for row in candidates if row.get("id") is not None}
    winner = next((row for row in ranking if _qualified(row)), None)
    qualified = [row for row in ranking if _qualified(row)]
    job_id = str(metadata.get("job_id") or task_status.get("job_id") or report_dir.name)
    sla = metadata.get("sla") if isinstance(metadata.get("sla"), dict) else {}

    lines = [
        "# 最佳启动参数报告",
        "",
        f"- 任务：`{job_id}`",
        f"- 模型：`{metadata.get('model', '未知')}`",
        f"- GPU：`{metadata.get('gpu_model', '未知')}` × "
        f"`{metadata.get('gpu_count', '未知')}`（单卡 "
        f"`{metadata.get('gpu_memory_gb', '未知')} GB`）",
        f"- 镜像：`{metadata.get('image', '未知')}`",
        f"- 负载：`{metadata.get('workload', '未知')}`",
        f"- 排名状态：`{task_status.get('ranking_status', '未知')}`",
        "",
        "## 结论",
        "",
    ]

    if winner is None:
        lines.append("没有找到同时满足 SLA 和输出健康检查的候选配置，不能推荐可上线参数。")
        if sla:
            lines.extend(
                [
                    "",
                    "## 本次 SLA",
                    "",
                    f"- 平均 TTFT：≤ `{_fmt(sla.get('max_avg_ttft_ms'))}` ms",
                    f"- 平均 TPOT：≤ `{_fmt(sla.get('max_avg_tpot_ms'))}` ms",
                    f"- 成功率：≥ `{_fmt_pct((_to_float(sla.get('min_success_rate')) or 0) * 100)}`",
                ]
            )
        failed_points: list[tuple[str, dict[str, Any]]] = []
        for row in ranking:
            candidate_id = str(row.get("candidate_id", "unknown"))
            detail = rows_by_id.get(candidate_id, {})
            points = detail.get("concurrency_points", [])
            if isinstance(points, list):
                failed_points.extend(
                    (candidate_id, point)
                    for point in points
                    if isinstance(point, dict)
                )
        if failed_points:
            lines.extend(
                [
                    "",
                    "## 未通过候选的实测点",
                    "",
                    "| 候选 | 并发 | 平均 TTFT | 平均 TPOT | 成功率 | 平均输出 tokens | 状态 |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for candidate_id, point in failed_points[: max(1, top_k)]:
                lines.append(
                    f"| `{candidate_id}` | {point.get('concurrency', '—')} | "
                    f"{_fmt(point.get('mean_ttft_ms'))} ms | "
                    f"{_fmt(point.get('mean_tpot_ms'))} ms | "
                    f"{_fmt_pct((_to_float(point.get('success_rate')) or 0) * 100)} | "
                    f"{_fmt(point.get('avg_output_tokens'), 0)} | "
                    f"{point.get('status', '—')} |"
                )
    else:
        winner_id = str(winner.get("candidate_id"))
        winner_row = rows_by_id.get(winner_id, {})
        lines.append(
            f"推荐候选 **`{winner_id}`**（排名第 `{winner.get('rank', 1)}`）："
            f"在满足 SLA 的候选中，它的整机 goodput 最高，"
            f"为 **{_fmt(winner.get('goodput_per_host'))}**。"
        )
        lines.extend(["", "## 最终启动参数", ""])
        command = configs_by_id.get(winner_id, {}).get("cmd") or winner_row.get("requested_command")
        if not command:
            command = _fallback_command(winner_row.get("effective_params", {}))
        lines.extend(["本次候选启动命令：", "", "```bash", _safe_text(command), "```", ""])
        lines.append(
            "> 说明：执行器会替换 `${MODEL_PATH}` 和端口，并强制加入 `--disable-radix-cache`。"
        )
        lines.append(
            "> 参数差异中的“未显式指定”表示该参数没有传给 `launch_server`，"
            "实际测试采用 SGLang/镜像默认行为；报告不会擅自猜测默认值。"
        )
        lines.extend(
            [
                "",
                "生效参数：",
                "",
                "```json",
                _json_text(winner_row.get("effective_params", {})),
                "```",
                "",
            ]
        )
        output_target_value = (
            (_to_float(workload.get("output_tokens")) or 0) * 0.9 if workload else None
        )
        output_target = _fmt(output_target_value, 0) if output_target_value is not None else "—"
        output_actual = _to_float(winner.get("avg_output_tokens"))
        output_result = (
            "通过"
            if output_target_value is None
            or (output_actual is not None and output_actual >= output_target_value)
            else "不通过"
        )
        lines.extend(
            [
                f"- TP 大小：`{winner.get('tp_size', '—')}`",
                f"- 每台机器实例数：`{_fmt(winner.get('instances_per_host'), 0)}`",
                f"- 最佳并发：`{winner.get('best_concurrency', '—')}`",
                "",
                "## 实测指标与 SLA",
                "",
                "| 指标 | 最佳实测 | 目标 | 结果 |",
                "|---|---:|---:|---|",
                f"| goodput / host | {_fmt(winner.get('goodput_per_host'))} | "
                "越高越好 | 排名依据 |",
                f"| 平均 TTFT | {_fmt(winner.get('mean_ttft_ms'))} ms | "
                f"≤ {_fmt(sla.get('max_avg_ttft_ms'))} ms | "
                f"{_check(winner.get('mean_ttft_ms'), sla.get('max_avg_ttft_ms'), 'le')} |",
                f"| 平均 TPOT | {_fmt(winner.get('mean_tpot_ms'))} ms | "
                f"≤ {_fmt(sla.get('max_avg_tpot_ms'))} ms | "
                f"{_check(winner.get('mean_tpot_ms'), sla.get('max_avg_tpot_ms'), 'le')} |",
                f"| 成功率 | {_fmt_pct((_to_float(winner.get('success_rate')) or 0) * 100)} | "
                f"≥ {_fmt_pct((_to_float(sla.get('min_success_rate')) or 0) * 100)} | "
                f"{_check(winner.get('success_rate'), sla.get('min_success_rate'), 'ge')} |",
                f"| 平均输出 tokens | {_fmt(winner.get('avg_output_tokens'), 0)} | "
                f"≥ {output_target} | {output_result} |",
            ]
        )

        lines.extend(["", "## 为什么它比其他候选好", ""])
        runner_up = next(
            (row for row in qualified if str(row.get("candidate_id")) != winner_id),
            None,
        )
        winner_goodput = _to_float(winner.get("goodput_per_host")) or 0.0
        if runner_up is not None:
            runner_goodput = _to_float(runner_up.get("goodput_per_host")) or 0.0
            margin = (winner_goodput / runner_goodput - 1) * 100 if runner_goodput else None
            lines.append(
                f"- 相比第二名 `{runner_up.get('candidate_id')}`，goodput / host 高 "
                f"`{_fmt(winner_goodput - runner_goodput)}`（`{_fmt_pct(margin)}`）。"
            )
            diffs = _param_differences(
                winner_row.get("effective_params", {}),
                rows_by_id.get(str(runner_up.get("candidate_id")), {}).get("effective_params", {}),
            )
            if diffs:
                lines.append("- 与第二名的有效参数差异：")
                lines.extend(f"  - {item}" for item in diffs)
        else:
            lines.append("- 没有第二个满足 SLA 的候选可作直接对比。")

        baseline = next((row for row in ranking if row.get("candidate_id") == "baseline"), None)
        if baseline is not None and str(baseline.get("candidate_id")) != winner_id:
            baseline_goodput = _to_float(baseline.get("goodput_per_host")) or 0.0
            margin = (winner_goodput / baseline_goodput - 1) * 100 if baseline_goodput else None
            lines.append(
                f"- 相比 baseline，goodput / host 高 `{_fmt(winner_goodput - baseline_goodput)}` "
                f"（`{_fmt_pct(margin)}`）。"
            )
        lines.append(
            f"- 排名使用的是 `goodput_per_host`，即候选在 SLA 约束下的单实例吞吐，"
            f"再结合 TP 大小折算整机可放置的实例数（当前最佳为 "
            f"`{winner.get('instances_per_host', '—')}` 个）。"
        )

        reasons = configs_by_id.get(winner_id, {}).get("reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons]
        if reasons:
            lines.append("- 该候选生成时记录的配置依据：")
            lines.extend(f"  - {_safe_text(reason)}" for reason in reasons)

    lines.extend(
        [
            "",
            "## 候选排名对比",
            "",
            "| 排名 | 候选 | TP | 实例/机 | 最佳并发 | "
            "goodput/host | TTFT | TPOT | 成功率 | 状态 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for index, row in enumerate(ranking[: max(1, top_k)], 1):
        candidate_id = str(row.get("candidate_id", "unknown"))
        status = rows_by_id.get(candidate_id, {}).get("status", "未知")
        display_id = (
            f"**{candidate_id}**"
            if winner and candidate_id == winner.get("candidate_id")
            else candidate_id
        )
        success = _to_float(row.get("success_rate"))
        success_text = "—" if success is None else f"{success * 100:.2f}%"
        lines.append(
            f"| {row.get('rank', index)} | {display_id} | {row.get('tp_size', '—')} | "
            f"{_fmt(row.get('instances_per_host'), 0)} | {row.get('best_concurrency', '—')} | "
            f"{_fmt(row.get('goodput_per_host'))} | {_fmt(row.get('mean_ttft_ms'))} ms | "
            f"{_fmt(row.get('mean_tpot_ms'))} ms | {success_text} | {status} |"
        )

    if winner is None:
        failed = [row for row in ranking if not _qualified(row)][: max(1, top_k)]
        lines.extend(["", "## 未通过候选", ""])
        for row in failed:
            candidate_id = str(row.get("candidate_id", "unknown"))
            detail = rows_by_id.get(candidate_id, {})
            reason = detail.get("failure_reason") or "没有找到满足 SLA 的并发点"
            lines.append(f"- `{candidate_id}`：{_safe_text(reason)}")

    lines.extend(
        [
            "",
            "## 报告来源",
            "",
            f"- 排名：`{results_dir / 'ranking.json'}`",
            f"- 候选结果：`{results_dir / 'candidate_results.jsonl'}`",
            "- 本报告由确定性脚本生成，没有调用 AI，也不会修改原始排名文件。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 executor 输出目录生成最佳启动参数 Markdown 报告"
    )
    parser.add_argument(
        "output_dir", type=Path, help="outputs/<job_id> 或 outputs/<job_id>/results"
    )
    parser.add_argument(
        "--job", type=Path, help="可选，显式指定 job.json；不指定时按 job_id 自动查找"
    )
    parser.add_argument("--configs", type=Path, help="可选，显式指定 configs.jsonl")
    parser.add_argument("--output", type=Path, help="可选，覆盖默认输出路径")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Markdown 中展示的候选数量（默认 10）"
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    try:
        results_dir, report_dir = _resolve_layout(args.output_dir)
        ranking = _load_ranking(results_dir / "ranking.json")
        if not ranking:
            raise ReportError("ranking.json 为空，没有可生成的排名")
        candidate_rows = _read_jsonl(results_dir / "candidate_results.jsonl")
        task_status = (
            _read_json(results_dir / "task_status.json")
            if (results_dir / "task_status.json").exists()
            else {}
        )
        if not isinstance(task_status, dict):
            task_status = {}
        config_path = _candidate_config_path(report_dir, results_dir, args.configs)
        candidates, config_meta = _load_candidates(config_path)
        job_id = str(task_status.get("job_id") or report_dir.name)
        job = (
            _read_json(args.job.expanduser().resolve())
            if args.job
            else _find_job(args.project_root.resolve(), job_id)
        )
        if not isinstance(job, dict):
            job = {}
        metadata = _safe_metadata(job, config_meta, task_status, job_id)
        workload = _load_workload(args.project_root.resolve(), metadata.get("workload"))
        output_path = (args.output or report_dir / "best_config.md").expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_markdown(
                report_dir=report_dir,
                results_dir=results_dir,
                ranking=ranking,
                candidate_rows=candidate_rows,
                candidates=candidates,
                task_status=task_status,
                metadata=metadata,
                workload=workload,
                top_k=max(1, args.top_k),
            ),
            encoding="utf-8",
        )
        print(f"已生成: {output_path}")
        return 0
    except ReportError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""The llmopt command-line entry point."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from planner.candidate_generator import CandidateGenerator
from planner.claude_code_client import ClaudeCodeClient
from planner.claude_env import default_env_file, load_env_file
from planner.command_renderer import (
    BenchmarkMethod,
    CommandRenderContext,
    CommandRenderer,
)
from planner.help_snapshot import load_help_flags
from planner.plan_validator import PlanValidator
from planner.reference_loader import ReferenceLoader, SGLangReferences
from planner.rule_checker import RuleChecker
from planner.search_planner import SearchPlanner
from planner.spec_loader import LoadedSpec, SpecLoader
from schemas.candidate import Candidate
from schemas.job_spec import JobSpec

app = typer.Typer(
    name="llmopt",
    help="Generate bounded LLM serving parameter search plans.",
    invoke_without_command=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the LLMOptAgent version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Entry point for the LLMOptAgent command-line interface."""
    if version:
        typer.echo("llmopt-agent 0.1.0")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc


def _load_job(path: Path) -> JobSpec:
    try:
        return JobSpec.model_validate(_read_json(path))
    except (ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid JobSpec: {exc}") from exc


def _load_method(project_root: Path, method_id: str) -> BenchmarkMethod:
    directory = project_root / "references" / "benchmark_methods"
    for path in sorted(directory.glob("*.json")):
        try:
            method = BenchmarkMethod.model_validate(_read_json(path))
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"Invalid benchmark method file {path}: {exc}") from exc
        if method.method_id == method_id:
            return method
    raise FileNotFoundError(
        f"Benchmark method {method_id!r} was not found under {directory}"
    )


def _validate_references(
    job: JobSpec,
    project_root: Path,
) -> tuple[LoadedSpec, LoadedSpec, LoadedSpec, SGLangReferences, BenchmarkMethod]:
    specs = SpecLoader(project_root / "specs")
    hardware = specs.load_hardware(job.instance_type)
    model = specs.load_model(job.model)
    workload = specs.load_workload(job.workload)
    references = ReferenceLoader(project_root / "references" / "sglang").load_sglang()
    method = _load_method(project_root, job.benchmark_method)
    return hardware, model, workload, references, method


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


@app.command("validate-job")
def validate_job(
    job_path: Annotated[Path, typer.Argument(help="Path to a fixed JSON JobSpec.")],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="LLMOptAgent project root."),
    ] = Path("."),
) -> None:
    """Validate a JobSpec and all referenced local specs and references."""
    try:
        job = _load_job(job_path)
        _validate_references(job, project_root)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(f"JobSpec valid: {job.job_id}")


@app.command()
def plan(
    job_path: Annotated[Path, typer.Argument(help="Path to a fixed JSON JobSpec.")],
    output: Annotated[Path, typer.Option("--output", help="Output artifact directory.")],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="LLMOptAgent project root."),
    ] = Path("."),
    search_plan: Annotated[
        Path | None,
        typer.Option("--search-plan", help="Offline fixture SearchPlan JSON."),
    ] = None,
    allow_dangerous_permissions: Annotated[
        bool,
        typer.Option("--dangerously-skip-permissions"),
    ] = False,
    claude_env_file: Annotated[
        Path | None,
        typer.Option(
            "--claude-env-file",
            help=(
                "Private Claude Code env file; defaults to .env or "
                "~/.config/llmopt-agent/claude.env."
            ),
        ),
    ] = None,
) -> None:
    """Generate a SearchPlan and candidates without executing any command."""
    try:
        job = _load_job(job_path)
        hardware, model, workload, references, _ = _validate_references(job, project_root)
        if search_plan is None:
            env_file = claude_env_file or default_env_file(project_root)
            if env_file is not None:
                load_env_file(env_file)
            planner = SearchPlanner(ClaudeCodeClient())
            plan_model = planner.generate(
                job,
                [project_root, project_root / "specs", project_root / "references"],
                allow_dangerous_permissions=allow_dangerous_permissions,
            )
        else:
            plan_model = PlanValidator().validate(_read_json(search_plan), job)
        rule_result = RuleChecker().check(job, hardware, model, workload, references, plan_model)
        if not rule_result.valid:
            details = "\n".join(f"- {issue.code}: {issue.message}" for issue in rule_result.errors)
            _fail(f"SearchPlan rejected by local rules:\n{details}")
        candidates = CandidateGenerator().generate(plan_model)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _fail(str(exc))

    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / "job_spec.json").write_text(
            json.dumps(job.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "search_plan.json").write_text(
            json.dumps(plan_model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
            for candidate in candidates:
                stream.write(
                    json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )
    except OSError as exc:
        _fail(
            f"Could not write plan output {output}: {exc}. "
            "Check directory ownership and permissions."
        )
    typer.echo(f"Plan written to {output}")


@app.command()
def render(
    search_plan_path: Annotated[Path, typer.Argument(help="Path to search_plan.json.")],
    output: Annotated[Path, typer.Option("--output", help="Output artifact directory.")],
    job: Annotated[Path, typer.Option("--job", help="Path to the original JobSpec JSON.")],
    model_path: Annotated[str, typer.Option("--model-path", help="Model weights path.")],
    dataset_path: Annotated[str, typer.Option("--dataset-path", help="Benchmark dataset path.")],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="LLMOptAgent project root."),
    ] = Path("."),
    host: Annotated[str, typer.Option("--host", help="Benchmark target host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Benchmark target port.")] = 30000,
    server_help: Annotated[
        Path | None,
        typer.Option(
            "--server-help",
            help="Path to launch_server --help snapshot for flag validation.",
        ),
    ] = None,
    benchmark_help: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-help",
            help="Path to bench_serving --help snapshot for flag validation.",
        ),
    ] = None,
) -> None:
    """Render candidates into server and benchmark commands without executing them."""
    try:
        job_model = _load_job(job)
        _validate_references(job_model, project_root)
        PlanValidator().validate(_read_json(search_plan_path), job_model)
        method = _load_method(project_root, job_model.benchmark_method)
        specs = SpecLoader(project_root / "specs")
        workload = specs.load_workload(job_model.workload)
        candidates_path = search_plan_path.parent / "candidates.jsonl"
        candidates = [
            Candidate.model_validate(json.loads(line))
            for line in candidates_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        context = CommandRenderContext(
            model_path=model_path,
            dataset_path=dataset_path,
            output_file=str(output / f"result_{job_model.job_id}_{{concurrency}}.jsonl"),
            benchmark_host=host,
            benchmark_port=port,
            available_server_flags=load_help_flags(server_help) if server_help else None,
            available_benchmark_flags=(
                load_help_flags(benchmark_help) if benchmark_help else None
            ),
        )
        renderer = CommandRenderer()
        rendered_candidates = [renderer.render(c, workload, method, context) for c in candidates]
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _fail(str(exc))

    try:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "rendered_candidates.jsonl").open("w", encoding="utf-8") as stream:
            for candidate in rendered_candidates:
                stream.write(
                    json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )

        shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for candidate in rendered_candidates:
            shell_lines.append(f"# {candidate.candidate_id}")
            shell_lines.append(shlex.join(candidate.server_command))
            shell_lines.extend(shlex.join(command) for command in candidate.benchmark_commands)
            shell_lines.append("")
        (output / "commands.sh").write_text("\n".join(shell_lines), encoding="utf-8")
        (output / "plan_report.md").write_text(
            "# LLMOptAgent Plan Report\n\n"
            f"- Job: `{job_model.job_id}`\n"
            f"- Candidates: {len(rendered_candidates)}\n"
            f"- Benchmark method: `{method.method_id}`\n\n"
            "Commands were rendered only; no server or benchmark was executed.\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _fail(
            f"Could not write rendered output {output}: {exc}. "
            "Check directory ownership and permissions."
        )
    typer.echo(f"Commands written to {output}")


if __name__ == "__main__":
    app()

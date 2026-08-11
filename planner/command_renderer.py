"""Render validated candidates into SGLang server and benchmark argv."""

from __future__ import annotations

import shlex
from typing import Any

from pydantic import Field

from planner.spec_loader import LoadedSpec
from schemas.candidate import Candidate
from schemas.job_spec import StrictModel


class BenchmarkMethod(StrictModel):
    method_id: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    entrypoint: str = Field(min_length=1)
    endpoint_protocol: str = Field(min_length=1)
    fixed_args: dict[str, Any] = Field(default_factory=dict)
    server_argument_mapping: dict[str, str] = Field(default_factory=dict)
    runtime_args: dict[str, str] = Field(default_factory=dict)
    argument_mapping: dict[str, str] = Field(default_factory=dict)
    traffic: dict[str, Any] = Field(default_factory=dict)
    derived_args: dict[str, str] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CommandRenderContext(StrictModel):
    model_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    output_file: str = Field(min_length=1)
    benchmark_host: str = "127.0.0.1"
    benchmark_port: int = Field(default=30000, gt=0, le=65535)
    available_server_flags: set[str] | None = None
    available_benchmark_flags: set[str] | None = None


class CommandRenderError(ValueError):
    """Raised when a candidate cannot be translated into a command."""


class CommandRenderer:
    """Translate normalized candidate parameters using a benchmark method mapping."""

    def render(
        self,
        candidate: Candidate,
        workload: LoadedSpec,
        method: BenchmarkMethod,
        context: CommandRenderContext,
    ) -> Candidate:
        server_command = self._render_server(candidate, method, context)
        benchmark_commands = self._render_benchmarks(workload, method, context)
        return candidate.model_copy(
            update={
                "server_command": server_command,
                "benchmark_command": benchmark_commands[0],
                "benchmark_commands": benchmark_commands,
            }
        )

    def _render_server(
        self,
        candidate: Candidate,
        method: BenchmarkMethod,
        context: CommandRenderContext,
    ) -> list[str]:
        params = dict(candidate.params)
        params.setdefault("model_path", context.model_path)
        command = ["python", "-m", "sglang.launch_server"]
        for parameter, value in params.items():
            flag = method.server_argument_mapping.get(parameter)
            if flag is None:
                raise CommandRenderError(
                    f"No server CLI mapping for candidate parameter {parameter!r}"
                )
            if (
                context.available_server_flags is not None
                and flag not in context.available_server_flags
            ):
                raise CommandRenderError(
                    f"Server flag {flag!r} is not present in the target launch_server --help"
                )
            command.extend([flag, self._stringify(value)])
        return command

    def _render_benchmarks(
        self,
        workload: LoadedSpec,
        method: BenchmarkMethod,
        context: CommandRenderContext,
    ) -> list[list[str]]:
        data = workload.data
        traffic = data.get("traffic", {})
        concurrency_values = traffic.get("values")
        if not isinstance(concurrency_values, list) or not concurrency_values:
            raise CommandRenderError("WorkloadSpec traffic.values must be a non-empty list")
        multiplier = traffic.get(
            "num_prompts_multiplier",
            method.traffic.get("num_prompts_multiplier", 1),
        )
        if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier <= 0:
            raise CommandRenderError("num_prompts_multiplier must be a positive integer")

        command_list: list[list[str]] = []
        for concurrency in concurrency_values:
            if (
                not isinstance(concurrency, int)
                or isinstance(concurrency, bool)
                or concurrency <= 0
            ):
                raise CommandRenderError(
                    "WorkloadSpec traffic.values must contain positive integers"
                )
            values = {
                "fixed_args.backend": method.fixed_args.get("backend"),
                "fixed_args.dataset_name": method.fixed_args.get("dataset_name"),
                "fixed_args.random_range_ratio": method.fixed_args.get("random_range_ratio"),
                "runtime_args.host": context.benchmark_host,
                "runtime_args.port": context.benchmark_port,
                "runtime_args.model": context.model_path,
                "runtime_args.dataset_path": context.dataset_path,
                "traffic.concurrency": concurrency,
                "workload.input_tokens.value": data.get("input_tokens", {}).get("value"),
                "workload.output_tokens.value": data.get("output_tokens", {}).get("value"),
                "result.output_file": context.output_file.replace(
                    "{concurrency}", str(concurrency)
                ),
                "derived.num_prompts": concurrency * multiplier,
            }
            command = ["python", "-m", "sglang.bench_serving"]
            for key, value in values.items():
                if value is None:
                    raise CommandRenderError(f"No value available for benchmark argument {key!r}")
                flag = method.argument_mapping.get(key)
                if flag is None:
                    raise CommandRenderError(f"No benchmark CLI mapping for {key!r}")
                if (
                    context.available_benchmark_flags is not None
                    and flag not in context.available_benchmark_flags
                ):
                    raise CommandRenderError(
                        f"Benchmark flag {flag!r} is not present in the target bench_serving --help"
                    )
                command.extend([flag, self._stringify(value)])
            command_list.append(command)
        return command_list

    @staticmethod
    def render_shell(candidate: Candidate) -> str:
        """Render a completed candidate as copyable shell commands."""
        if not candidate.server_command or not candidate.benchmark_commands:
            raise CommandRenderError("Candidate must be rendered before creating shell output")
        lines = [shlex.join(candidate.server_command)]
        lines.extend(shlex.join(command) for command in candidate.benchmark_commands)
        return "\n".join(lines)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

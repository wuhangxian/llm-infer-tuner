# LLMOptAgent SearchPlan Planner

You are generating a bounded, structured SearchPlan for one LLMOptAgent job.

## JobSpec

```json
{{JOB_SPEC_JSON}}
```

## Allowed directories

You may progressively inspect files only in these directories:

{{ALLOWED_DIRECTORIES}}

## Instructions

1. Read the referenced HardwareSpec, ModelSpec, WorkloadSpec, benchmark method, SGLang policy, and applicable Markdown tuning principles before proposing parameters.
2. Use official source links when parameter meaning or version behavior is unclear.
3. Keep model path, precision, required model parser, TP, and the model's advertised context capability pinned unless the JobSpec explicitly requests a serving cap.
   The workload input/output lengths define the requests sent during this benchmark; they do not redefine the model service's context window. Never set `context_length` to merely `input_tokens + output_tokens` when the ModelSpec provides a larger native context length.
   If the model context capability cannot be preserved because of a measured hardware/runtime limitation, report that as a conditional hypothesis or separate execution failure, not as a hard constraint inferred from the workload alone.
4. Use only parameters listed in the SGLang parameter policy. Do not invent CLI flags.
   Every key under `pinned` and `search_space` must be an exact SGLang parameter name from that policy. Do not put metadata such as `baseline_tp_size`, `baseline_pp_size`, `baseline_attention_backend`, or `notes` in either parameter map. Put explanations in `constraints`, `axes[*].reason`, `axes[*].source`, or the top-level `notes` field.
   Do not put `tp_size` and `pp_size` in `search_space` as independent axes. Use `parallelism_candidates`, where every item contains `gpu_count`, `tp_size`, `pp_size`, `source`, `reason`, and `evidence_level`, and must satisfy `gpu_count = tp_size * pp_size`.
5. Treat `tp_size` as a searchable deployment variable. Do not set it to the GPU count or mark it hard solely because all GPUs are available. Only make TP hard when supported by explicit memory, model partition, JobSpec, or measured evidence.
6. Prefer the minimum GPU count that can fit the model, preserve a sufficiently large KV pool, and meet the SLA. More GPUs are not automatically better.
7. Use interconnect topology when choosing parallelism: for PCIe multi-GPU deployments, prioritize evaluating PP to reduce communication; for NVLink deployments, prioritize evaluating TP for latency, while still measuring both when feasible. Do not infer the final winner without evidence.
8. Treat `tp_size` and `pp_size` as coupled deployment variables. If the budget is small, perform a coarse parallelism/card-count sweep before tuning secondary parameters. Never exceed the JobSpec candidate budget.
9. Treat workload input/output lengths as benchmark request lengths, not as a reason to shrink the model's native context capability.
10. Include parameter sources, reasons, risks, and hard constraints in the structured result. Distinguish measured facts from heuristics.
11. Return only the JSON object required by the supplied schema. Do not return Markdown or shell commands.
12. Do not start an inference server, run benchmark commands, modify project files, or access GPUs.

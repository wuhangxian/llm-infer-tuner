"""Shared strict scalar contract for SGLang candidate and baseline parameters."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import TypeAlias

from pydantic import ConfigDict, RootModel, model_validator

ParameterScalar: TypeAlias = str | bool | int | float | None

_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# The image catalogue calls ``mamba_scheduler_strategy`` an alias for the
# newer radix-cache spelling.  Normalize it at the boundary so it cannot
# create a second, semantically identical parameter or leak to the executor.
PARAMETER_ALIASES = {
    "mamba_scheduler_strategy": "mamba_radix_cache_strategy",
}

# These are the scalar types explicitly represented by the current benchmark
# mapping, README contract, and SGLang image catalogue.  Unknown future flags
# remain supported as finite scalar values, but cannot evade validation through
# an alternate spelling of one of these known flags.
INTEGER_PARAMETERS = frozenset(
    {
        "asr_max_buffer_seconds",
        "asr_max_concurrent_sessions",
        "attn_cp_size",
        "base_gpu_id",
        "batch_notify_size",
        "chunked_prefill_size",
        "context_length",
        "cpu_offload_gb",
        "cuda_graph_bs",
        "cuda_graph_bs_decode",
        "cuda_graph_bs_prefill",
        "cuda_graph_max_bs",
        "cuda_graph_max_bs_decode",
        "cuda_graph_max_bs_prefill",
        "dcp_size",
        "decoupled_spec_rank",
        "decode_log_interval",
        "default_priority_value",
        "detokenizer_worker_num",
        "disaggregation_bootstrap_port",
        "disaggregation_decode_extra_slots",
        "disaggregation_decode_polling_interval",
        "dp_size",
        "dist_timeout",
        "dynamic_batch_tokenizer_batch_size",
        "dwdp_size",
        "elastic_ep_initial_size",
        "encoder_bootstrap_port",
        "engine_info_bootstrap_port",
        "ep_size",
        "ep_num_redundant_experts",
        "eplb_rebalance_layers_per_chunk",
        "eplb_rebalance_num_iterations",
        "expert_distribution_recorder_buffer_size",
        "gpu_id_step",
        "grpc_port",
        "hicache_size",
        "int8_mamba_ckpt_size",
        "kt_cpuinfer",
        "kt_max_deferred_experts_per_token",
        "kt_num_gpu_experts",
        "kt_threadpool_count",
        "kv_canary_sweep_interval",
        "linear_replayssm_cache_len",
        "load_snapshot_publish_interval",
        "log_requests_level",
        "mamba_cache_philox_rounds",
        "mamba_track_interval",
        "max_ep_size",
        "max_lora_chunk_size",
        "max_loaded_loras",
        "max_lora_rank",
        "max_loras_per_batch",
        "max_mamba_cache_size",
        "max_prefill_tokens",
        "max_queued_requests",
        "max_running_requests",
        "max_total_tokens",
        "min_free_slots_delay",
        "moe_dense_tp_size",
        "moe_dp_size",
        "nccl_port",
        "nnodes",
        "node_rank",
        "num_continuous_decode_steps",
        "num_draft_tokens",
        "num_reserved_decode_tokens",
        "offload_group_size",
        "offload_num_in_group",
        "offload_prefetch_step",
        "optimistic_prefill_attempts",
        "page_size",
        "port",
        "pp_max_micro_batch_size",
        "pp_async_batch_depth",
        "pp_size",
        "prefill_max_requests",
        "prefill_delayer_max_delay_passes",
        "priority_scheduling_preemption_threshold",
        "random_seed",
        "remote_instance_weight_loader_seed_instance_service_port",
        "scheduler_recv_interval",
        "sm_group_num",
        "smg_http_sidecar_port",
        "speculative_dflash_block_size",
        "speculative_dflash_draft_window_size",
        "speculative_draft_window_size",
        "speculative_dspark_block_size",
        "speculative_eagle_topk",
        "speculative_ngram_capacity",
        "speculative_ngram_external_sam_budget",
        "speculative_ngram_external_corpus_max_tokens",
        "speculative_ngram_max_bfs_breadth",
        "speculative_ngram_max_trie_depth",
        "speculative_ngram_min_bfs_breadth",
        "speculative_num_draft_tokens",
        "speculative_num_steps",
        "stream_interval",
        "tp_size",
        "tokenizer_worker_num",
        "torch_compile_max_bs",
        "triton_attention_num_kv_splits",
        "triton_attention_split_tile_size",
        "weight_loader_prefetch_num_threads",
    }
)
FLOAT_PARAMETERS = frozenset(
    {
        "dynamic_batch_tokenizer_batch_timeout",
        "eplb_min_rebalancing_utilization_threshold",
        "gc_warning_threshold_secs",
        "hicache_ratio",
        "lora_drain_wait_threshold",
        "mamba_full_memory_ratio",
        "mem_fraction_static",
        "prefill_delayer_max_delay_ms",
        "prefill_delayer_queue_min_ratio",
        "prefill_delayer_token_usage_low_watermark",
        "random_range_ratio",
        "schedule_conservativeness",
        "soft_watchdog_timeout",
        "speculative_accept_threshold_acc",
        "speculative_accept_threshold_single",
        "swa_full_tokens_ratio",
        "tbo_token_distribution_threshold",
        "watchdog_timeout",
    }
)
BOOLEAN_PARAMETERS = frozenset(
    {
        "abort_on_priority_when_disabled",
        "allow_auto_truncate",
        "checkpoint_engine_wait_weights_before_ready",
        "constrained_json_disable_any_whitespace",
        "debug_cuda_graph",
        "delete_ckpt_after_loading",
        "disaggregation_decode_enable_offload_kvcache",
        "disaggregation_decode_enable_radix_cache",
        "disable_attn_tp_gather",
        "disable_chunked_prefix_cache",
        "disable_cuda_graph_padding",
        "disable_cuda_graph",
        "disable_custom_all_reduce",
        "disable_decode_cuda_graph",
        "disable_fast_image_processor",
        "disable_flashinfer_autotune",
        "disable_flashinfer_cutlass_moe_fp4_allgather",
        "disable_hybrid_swa_memory",
        "disable_outlines_disk_cache",
        "disable_overlap_schedule",
        "disable_piecewise_cuda_graph",
        "disable_prefill_cuda_graph",
        "disable_priority_preemption",
        "disable_radix_cache",
        "disable_shared_experts_fusion",
        "disable_tokenizer_batch_decode",
        "dllm_fdfo",
        "elastic_ep_rejoin",
        "enable_adaptive_dispatch_to_encoder",
        "enable_aiter_allreduce_fusion",
        "enable_attn_tp_input_scattered",
        "enable_breakable_cuda_graph",
        "enable_broadcast_mm_inputs_process",
        "enable_cache_report",
        "enable_cudagraph_gc",
        "enable_custom_logit_processor",
        "enable_deepseek_v4_fp4_indexer",
        "enable_deterministic_inference",
        "enable_dp_attention",
        "enable_dp_attention_local_control_broadcast",
        "enable_dp_lm_head",
        "enable_draft_weights_cpu_backup",
        "enable_dsa_cache_layer_split",
        "enable_dsa_prefill_context_parallel",
        "enable_dynamic_batch_tokenizer",
        "enable_dynamic_chunking",
        "enable_elastic_expert_backup",
        "enable_eplb",
        "enable_expert_distribution_metrics",
        "enable_flashinfer_allreduce_fusion",
        "enable_flexkv",
        "enable_forward_pass_metrics",
        "enable_fp32_lm_head",
        "enable_fused_moe_sum_all_reduce",
        "enable_fused_qk_norm_rope",
        "enable_hisparse",
        "enable_hierarchical_cache",
        "enable_http2",
        "enable_int8_mamba_checkpoint",
        "enable_layerwise_nvtx_marker",
        "enable_linear_replayssm",
        "enable_lmcache",
        "enable_lora_overlap_loading",
        "enable_mamba_cache_stochastic_rounding",
        "enable_memory_saver",
        "enable_metrics",
        "enable_metrics_for_all_schedulers",
        "enable_mfu_metrics",
        "enable_mm_global_cache",
        "enable_mscclpp",
        "enable_mis",
        "enable_mixed_chunk",
        "enable_multi_layer_eagle",
        "enable_nccl_nvls",
        "enable_page_major_kv_layout",
        "enable_p2p_check",
        "enable_pdmux",
        "enable_prefill_context_parallel",
        "enable_prefill_cp",
        "enable_prefill_delayer",
        "enable_precise_embedding_interpolation",
        "enable_priority_scheduling",
        "enable_prefix_mm_cache",
        "enable_profile_cuda_graph",
        "enable_request_time_stats_logging",
        "enable_return_hidden_states",
        "enable_return_indexer_topk",
        "enable_return_routed_experts",
        "enable_session_radix_cache",
        "enable_single_batch_overlap",
        "enable_ssl_refresh",
        "enable_streaming_session",
        "enable_strict_thinking",
        "enable_symm_mem",
        "enable_tf32_matmul",
        "enable_tokenizer_batch_encode",
        "enable_torch_compile",
        "enable_torch_compile_debug_mode",
        "enable_torch_symm_mem",
        "enable_trace",
        "enable_two_batch_overlap",
        "enable_unified_memory",
        "enable_waterfill",
        "enable_weights_cpu_backup",
        "encoder_only",
        "enforce_piecewise_cuda_graph",
        "enforce_disable_flashinfer_allreduce_fusion",
        "enforce_shared_experts_fusion",
        "export_metrics_to_file",
        "flashinfer_mla_disable_ragged",
        "is_baseline",
        "is_embedding",
        "incremental_streaming_output",
        "keep_mm_feature_on_device",
        "language_only",
        "log_requests",
        "lora_strict_loading",
        "lora_use_virtual_experts",
        "mm_enable_dp_encoder",
        "prefill_only_disable_kv_cache",
        "pre_warm_nccl",
        "quantize_and_serve",
        "remote_instance_weight_loader_start_seed_via_transfer_engine",
        "schedule_low_priority_values_first",
        "show_time_cost",
        "skip_server_warmup",
        "skip_tokenizer_init",
        "sleep_on_idle",
        "smg_grpc_mode",
        "grpc_mode",
        "speculative_adaptive",
        "speculative_dspark_align_verify_tokens_to_graph_tier",
        "speculative_skip_dp_mlp_sync",
        "speculative_use_rejection_sampling",
        "stream_response_default_include_usage",
        "strip_thinking_cache",
        "triton_attention_reduce_in_fp32",
        "trust_remote_code",
        "use_ray",
        "uses_mamba_radix_cache",
        "weight_loader_disable_mmap",
        "weight_loader_drop_cache_after_load",
        "weight_loader_prefetch_checkpoints",
    }
)
STRING_PARAMETERS = frozenset(
    {
        "attention_backend",
        "kv_cache_dtype",
        "mamba_radix_cache_strategy",
        "mamba_scheduler_strategy",
        "mamba_ssm_dtype",
        "reasoning_parser",
        "speculative_algorithm",
        "tool_call_parser",
    }
)
RUNTIME_PARAMETERS = frozenset({"host", "model_path", "port"})
MAMBA_STRATEGY_PARAMETERS = frozenset({"mamba_radix_cache_strategy"})


def normalise_parameter_name(name: str) -> str:
    normalised = name.replace("-", "_")
    return PARAMETER_ALIASES.get(normalised, normalised)


def is_safe_parameter_name(name: str) -> bool:
    return bool(_PARAMETER_NAME.fullmatch(name))


def _finite_number(value: object, *, location: str) -> float:
    if type(value) is int:
        return float(value)
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{location} must be a finite number")


def validate_parameter_value(name: str, value: object, *, location: str) -> ParameterScalar:
    """Validate a scalar and canonicalise known floating-point values."""
    if name in INTEGER_PARAMETERS:
        if type(value) is not int:
            raise ValueError(f"{location} must be an integer")
        return value
    if name in FLOAT_PARAMETERS:
        return _finite_number(value, location=location)
    if name in BOOLEAN_PARAMETERS:
        if type(value) is not bool:
            raise ValueError(f"{location} must be a boolean")
        return value
    if name in STRING_PARAMETERS:
        if type(value) is not str:
            raise ValueError(f"{location} must be a string")
        return value
    if name in {"host", "model_path"}:
        if type(value) is not str:
            raise ValueError(f"{location} must be a string")
        return value
    if name == "port":
        if type(value) is not int:
            raise ValueError(f"{location} must be an integer")
        return value
    if value is None:
        return None
    if type(value) is str:
        return value
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{location} must be a finite scalar JSON value")


def normalise_parameter_mapping(values: object, *, location: str) -> dict[str, ParameterScalar]:
    """Normalise flag aliases and reject every duplicate or non-scalar value."""
    if not isinstance(values, dict):
        raise ValueError(f"{location} must be a JSON object")
    normalised: dict[str, ParameterScalar] = {}
    for raw_name, raw_value in values.items():
        if type(raw_name) is not str:
            raise ValueError(f"{location} has a non-string parameter name")
        name = normalise_parameter_name(raw_name)
        if not is_safe_parameter_name(name):
            raise ValueError(f"{location} has unsafe parameter name {raw_name!r}")
        if name in normalised:
            if name == "disable_radix_cache" and raw_value is True and normalised[name] is True:
                continue
            raise ValueError(f"{location} repeats normalized parameter {name!r}")
        normalised[name] = validate_parameter_value(
            name, raw_value, location=f"{location}.{raw_name}"
        )
    return normalised


def command_scalar(name: str, value: str | bool) -> ParameterScalar:
    """Parse a shell-token value using the same known-flag type contract."""
    if value is True:
        return validate_parameter_value(name, value, location=f"legacy cmd --{name}")
    if not isinstance(value, str):
        raise ValueError(f"legacy cmd --{name} must be a string value")
    if name in INTEGER_PARAMETERS or name == "port":
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError(f"legacy cmd --{name} must be an integer")
        return validate_parameter_value(name, int(value), location=f"legacy cmd --{name}")
    if name in FLOAT_PARAMETERS:
        try:
            numeric = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"legacy cmd --{name} must be a finite number") from exc
        if not numeric.is_finite():
            raise ValueError(f"legacy cmd --{name} must be a finite number")
        return validate_parameter_value(name, float(numeric), location=f"legacy cmd --{name}")
    if name in BOOLEAN_PARAMETERS:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        raise ValueError(f"legacy cmd --{name} must be a boolean")
    return validate_parameter_value(name, value, location=f"legacy cmd --{name}")


class CandidateParams(RootModel[dict[str, ParameterScalar]]):
    """Typed parameter map used at the candidate boundary."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def normalise(cls, value: object) -> dict[str, ParameterScalar]:
        if isinstance(value, CandidateParams):
            return value.root
        return normalise_parameter_mapping(value, location="candidate params")

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def __getitem__(self, key: str) -> ParameterScalar:
        return self.root[key]

    def get(self, key: str, default: ParameterScalar | None = None) -> ParameterScalar | None:
        return self.root.get(key, default)

    def items(self):
        return self.root.items()

    def as_dict(self) -> dict[str, ParameterScalar]:
        return dict(self.root)


__all__ = [
    "BOOLEAN_PARAMETERS",
    "CandidateParams",
    "FLOAT_PARAMETERS",
    "INTEGER_PARAMETERS",
    "MAMBA_STRATEGY_PARAMETERS",
    "PARAMETER_ALIASES",
    "ParameterScalar",
    "RUNTIME_PARAMETERS",
    "command_scalar",
    "is_safe_parameter_name",
    "normalise_parameter_mapping",
    "normalise_parameter_name",
]

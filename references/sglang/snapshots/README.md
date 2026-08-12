# Target SGLang help snapshots

Save help output from the exact SGLang image used for a real run under a
versioned directory:

```bash
snapshot_dir="references/sglang/snapshots/<sglang-version>"
mkdir -p "$snapshot_dir"
python -m sglang.launch_server --help \
  > "$snapshot_dir/launch_server_help.txt"
python -m sglang.bench_serving --help \
  > "$snapshot_dir/bench_serving_help.txt"
```

Then pass both files to `llmopt render` with `--server-help` and
`--benchmark-help`. These files must come from the target image; do not create
them from the host Python environment or infer them from web documentation.

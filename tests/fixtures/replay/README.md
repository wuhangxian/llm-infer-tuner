# Offline report replay fixtures

These fixtures are intentionally small, synthetic extracts of historical
benchmark evidence.  They contain only the fields needed to exercise the
report writer and finality rules; no logs, hostnames, credentials, model
paths, or raw request payloads are checked in.

`valid_metrics.json` represents a mechanically valid Round-2 boundary and is
used to verify that finite throughput/latency values survive a schema-v2
rewrite.  `runtime_failures.json` represents the six infrastructure failure
classes seen in older runs.  Those attempts remain visible, but the candidate
must stay `incomplete` and the task must remain `PROVISIONAL`.

The replay tests deliberately run through `runners.reporting.write_reports`
and then reload the immutable generation.  They are an offline compatibility
check, not a claim that the historical runs were re-executed.

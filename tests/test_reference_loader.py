from pathlib import Path

import pytest

from planner.reference_loader import ReferenceLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loader_reads_sglang_sources_and_parameter_policy() -> None:
    references = ReferenceLoader(PROJECT_ROOT / "references" / "sglang").load_sglang()

    assert references.sources.engine == "sglang"
    assert references.sources.sources[0].authority == "official"
    assert "chunked_prefill_size" in references.policy.policy.searchable_first_pass
    assert "tp_size" in references.policy.policy.searchable_first_pass
    assert "pp_size" in references.policy.policy.searchable_first_pass
    assert "tp_size" not in references.policy.policy.usually_pinned
    assert references.policy.parameters["chunked_prefill_size"].risk == "high"
    assert references.policy.parameters["tp_size"].risk == "high"


def test_loader_reports_missing_reference_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sources.json"):
        ReferenceLoader(tmp_path).load_sglang()

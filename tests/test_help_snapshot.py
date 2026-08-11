from pathlib import Path

import pytest

from planner.help_snapshot import HelpSnapshotError, load_help_flags


def test_load_help_flags_extracts_long_options_and_aliases(tmp_path: Path) -> None:
    snapshot = tmp_path / "launch_server_help.txt"
    snapshot.write_text(
        """Usage: launch_server [OPTIONS]
  --model-path TEXT
  --tp-size INTEGER
  --attention-backend TEXT, --attention-backend-v2 TEXT
  -h, --help  Show this message and exit.
""",
        encoding="utf-8",
    )

    assert load_help_flags(snapshot) == {
        "--model-path",
        "--tp-size",
        "--attention-backend",
        "--attention-backend-v2",
        "--help",
    }


def test_load_help_flags_rejects_empty_or_unreadable_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "empty.txt"
    snapshot.write_text("usage: launch_server\n", encoding="utf-8")

    with pytest.raises(HelpSnapshotError, match="no long options"):
        load_help_flags(snapshot)

    with pytest.raises(HelpSnapshotError, match="does not exist"):
        load_help_flags(tmp_path / "missing.txt")

from typer.testing import CliRunner

from cli.main import app


def test_cli_help_is_available() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Generate bounded LLM serving parameter search plans." in result.stdout


def test_cli_version_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "llmopt-agent 0.1.0"

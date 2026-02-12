from pathlib import Path


def test_makefile_primary_targets_use_victor_cli():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    expected_commands = [
        "python3 -m victor_invest.cli analyze $(SYMBOL) --mode standard",
        "python3 -m victor_invest.cli analyze $(SYMBOL) --mode standard --force-refresh",
        "python3 -m victor_invest.cli batch $(SYMBOLS) --mode standard",
        "python3 -m victor_invest.cli status",
        "python3 -m victor_invest.cli inspect-cache --symbol $(SYMBOL) --verbose",
        "python3 -m victor_invest.cli clean-cache --symbol $(SYMBOL)",
    ]

    for expected in expected_commands:
        assert expected in makefile


def test_pyproject_pins_supported_victor_version_range():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "victor-ai>=0.5.0,<0.6.0" in pyproject


def test_legacy_cli_declares_deprecated_forwarding_mode():
    legacy_cli = Path("cli_orchestrator.py").read_text(encoding="utf-8")

    assert "DEPRECATED: This CLI is maintained for backwards compatibility only." in legacy_cli
    assert "python -m victor_invest.cli analyze AAPL --mode standard" in legacy_cli
    assert '[sys.executable, "-m", "victor_invest.cli"] + args' in legacy_cli

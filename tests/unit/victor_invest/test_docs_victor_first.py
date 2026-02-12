from pathlib import Path


ACTIVE_DOCS = [
    Path("README.md"),
    Path("README.adoc"),
    Path("docs/README.adoc"),
    Path("docs/DEVELOPER_GUIDE.adoc"),
    Path("docs/OPERATIONS_RUNBOOK.md"),
    Path("docs/ARCHITECTURE.md"),
]


def test_active_docs_do_not_use_legacy_cli_commands():
    legacy_patterns = [
        "python3 cli_orchestrator.py",
        "python cli_orchestrator.py",
    ]

    violations = {}
    for doc in ACTIVE_DOCS:
        content = doc.read_text(encoding="utf-8")
        found = [pattern for pattern in legacy_patterns if pattern in content]
        if found:
            violations[str(doc)] = found

    assert not violations, f"Legacy CLI command references found in active docs: {violations}"

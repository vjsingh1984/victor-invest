from pathlib import Path
import re


def test_tool_implementations_use_victor_result_factory_names():
    disallowed_patterns = [
        "ToolResult.success_result(",
        "ToolResult.error_result(",
        "ToolResult.create_success(data=",
        "ToolResult.create_failure(data=",
    ]

    violations = {}
    for tool_file in Path("victor_invest/tools").glob("*.py"):
        if tool_file.name in {"base.py", "__init__.py"}:
            continue
        content = tool_file.read_text(encoding="utf-8")
        found = [pattern for pattern in disallowed_patterns if pattern in content]
        if re.search(r"ToolResult\.create_(success|failure)\([^)]*warnings\s*=", content, re.DOTALL):
            found.append("ToolResult.create_*(..., warnings=...)")
        if found:
            violations[str(tool_file)] = found

    assert not violations, f"Legacy ToolResult helper usage found: {violations}"


def test_no_legacy_data_alias_usage_in_tools_or_workflows():
    target_dirs = [Path("victor_invest/tools"), Path("victor_invest/workflows")]
    violations = {}

    pattern = re.compile(r"\b\w+\.data\b")
    for target_dir in target_dirs:
        for py_file in target_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            matches = pattern.findall(content)
            if matches:
                violations[str(py_file)] = sorted(set(matches))

    assert not violations, f"Legacy `.data` alias usage found: {violations}"

from pathlib import Path


def test_ci_pipeline_has_performance_benchmark_gate():
    ci_yaml = Path(".github/workflows/ci-cd.yml").read_text(encoding="utf-8")

    assert "performance-benchmark:" in ci_yaml
    assert "--runner stub" in ci_yaml
    assert "--budget-profile ci_stub" in ci_yaml
    assert "--fail-on-budget-breach" in ci_yaml


def test_build_job_depends_on_performance_benchmark_gate():
    ci_yaml = Path(".github/workflows/ci-cd.yml").read_text(encoding="utf-8")
    assert "needs: [test, security, victor-compat, performance-benchmark]" in ci_yaml

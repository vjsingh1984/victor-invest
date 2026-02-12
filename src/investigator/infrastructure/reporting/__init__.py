"""Reporting infrastructure package with lazy optional dependency loading."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from investigator.infrastructure.reporting.report_generator import PDFReportGenerator, ReportConfig
    from investigator.infrastructure.reporting.report_payload_builder import ReportPayloadBuilder
    from investigator.infrastructure.reporting.weekly_report_generator import WeeklyReportGenerator


def __getattr__(name: str) -> Any:
    if name in {"PDFReportGenerator", "ReportConfig"}:
        from investigator.infrastructure.reporting.report_generator import (
            PDFReportGenerator,
            ReportConfig,
        )

        return {"PDFReportGenerator": PDFReportGenerator, "ReportConfig": ReportConfig}[name]
    if name == "ReportPayloadBuilder":
        from investigator.infrastructure.reporting.report_payload_builder import ReportPayloadBuilder

        return ReportPayloadBuilder
    if name == "WeeklyReportGenerator":
        from investigator.infrastructure.reporting.weekly_report_generator import WeeklyReportGenerator

        return WeeklyReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PDFReportGenerator",
    "ReportConfig",
    "ReportPayloadBuilder",
    "WeeklyReportGenerator",
]

# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow state definitions for investment analysis.

This module defines the state structures used in analysis workflows,
including the AnalysisMode enum and AnalysisWorkflowState dataclass.

Example:
    from victor_invest.workflows.state import AnalysisMode, AnalysisWorkflowState

    # Create state for a standard analysis
    state = AnalysisWorkflowState(
        symbol="AAPL",
        mode=AnalysisMode.STANDARD
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AnalysisMode(str, Enum):
    """Analysis mode determining which agents participate in the workflow.

    Attributes:
        QUICK: Technical analysis only - fastest, market data focused
        STANDARD: Technical + Fundamental - balanced analysis
        COMPREHENSIVE: All agents - full institutional-grade research
        CUSTOM: User-defined agent selection
    """

    QUICK = "quick"  # Technical only
    STANDARD = "standard"  # Technical + Fundamental
    COMPREHENSIVE = "comprehensive"  # All agents
    CUSTOM = "custom"  # User-defined


@dataclass
class AnalysisWorkflowState:
    """State container for investment analysis workflows.

    This dataclass holds all data flowing through the analysis workflow,
    from initial data collection through final synthesis. It follows
    the StateGraph pattern from victor-core.

    Attributes:
        symbol: Stock ticker symbol being analyzed
        mode: Analysis mode determining workflow path
        sec_data: SEC filings and fundamental data
        market_data: Price, volume, and market indicators
        fundamental_analysis: Results from fundamental analysis agent
        technical_analysis: Results from technical analysis agent
        market_context: Market regime and sector context analysis
        synthesis: Combined analysis synthesis
        recommendation: Final investment recommendation
        errors: List of errors encountered during workflow
        completed_steps: List of successfully completed workflow steps
    """

    # Required fields
    symbol: str
    mode: AnalysisMode

    # Data collection results
    sec_data: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None

    # Analysis results
    fundamental_analysis: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None

    # Synthesis
    synthesis: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None

    # Tracking
    errors: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of the workflow state.
        """
        return {
            "symbol": self.symbol,
            "mode": self.mode.value,
            "sec_data": self.sec_data,
            "market_data": self.market_data,
            "fundamental_analysis": self.fundamental_analysis,
            "technical_analysis": self.technical_analysis,
            "market_context": self.market_context,
            "synthesis": self.synthesis,
            "recommendation": self.recommendation,
            "errors": self.errors,
            "completed_steps": self.completed_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisWorkflowState":
        """Create state from dictionary.

        Args:
            data: Dictionary with state fields.

        Returns:
            AnalysisWorkflowState instance.
        """
        mode = data.get("mode", AnalysisMode.STANDARD)
        if isinstance(mode, str):
            mode = AnalysisMode(mode)

        return cls(
            symbol=data["symbol"],
            mode=mode,
            sec_data=data.get("sec_data"),
            market_data=data.get("market_data"),
            fundamental_analysis=data.get("fundamental_analysis"),
            technical_analysis=data.get("technical_analysis"),
            market_context=data.get("market_context"),
            synthesis=data.get("synthesis"),
            recommendation=data.get("recommendation"),
            errors=data.get("errors", []),
            completed_steps=data.get("completed_steps", []),
        )

    def add_error(self, error: str) -> None:
        """Add an error to the tracking list.

        Args:
            error: Error message to record.
        """
        self.errors.append(error)

    def mark_step_completed(self, step: str) -> None:
        """Mark a workflow step as completed.

        Args:
            step: Name of the completed step.
        """
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def is_step_completed(self, step: str) -> bool:
        """Check if a workflow step has been completed.

        Args:
            step: Name of the step to check.

        Returns:
            True if step is completed, False otherwise.
        """
        return step in self.completed_steps

    def has_errors(self) -> bool:
        """Check if any errors occurred during workflow.

        Returns:
            True if errors exist, False otherwise.
        """
        return len(self.errors) > 0


__all__ = [
    "AnalysisMode",
    "AnalysisWorkflowState",
]

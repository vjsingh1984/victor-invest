"""
Signal Generation Module

Provides entry/exit signal generation for stock analysis.
"""

from investigator.domain.services.signals.entry_exit_engine import (
    EntrySignal,
    ExitSignal,
    OptimalEntryZone,
    SignalType,
    SignalConfidence,
    SignalTiming,
    ScalingStrategy,
    EntryExitEngine,
    get_entry_exit_engine,
)

from investigator.domain.services.signals.signal_integrator import (
    IntegratedSignals,
    SignalIntegrator,
    get_signal_integrator,
)

__all__ = [
    # Entry/Exit Engine
    "EntrySignal",
    "ExitSignal",
    "OptimalEntryZone",
    "SignalType",
    "SignalConfidence",
    "SignalTiming",
    "ScalingStrategy",
    "EntryExitEngine",
    "get_entry_exit_engine",
    # Signal Integrator
    "IntegratedSignals",
    "SignalIntegrator",
    "get_signal_integrator",
]

# Victor-Invest Alignment Review and Roadmap (2026-02-11)

## Scope
- Reviewed `victor-invest` architecture, implementation quality, and migration status.
- Reviewed Victor framework updates in `../codingagent` (version `0.5.0`, build `1964e9bf`).
- Identified what should remain business-domain logic in `victor-invest` versus what should be reused from Victor framework.

## Executive Summary
- `victor-invest` is in a **hybrid state**: strong domain depth, but still carrying legacy orchestration and duplicated integration logic.
- Victor framework capabilities now cover most agentic infrastructure (`VerticalBase`, workflow provider patterns, handler registry, tool registry contracts).
- Immediate compatibility gaps with latest Victor were present and have now been patched in this iteration.
- Primary next objective: make `victor-invest` a **thin investment vertical + domain tools/handlers** and remove duplicated orchestration/agent framework code.

## Current Snapshot (Evidence)
- Python files scanned under `src/`, `victor_invest/`, `tests/`: `407`
- Approx LOC under `src/` + `victor_invest/`: `163,549`
- `TODO|FIXME|HACK|DEPRECATED` markers: `51`
- Large files:
- `src/investigator/domain/agents/fundamental/agent.py` (~6340 lines)
- `src/investigator/application/synthesizer.py` (~6018 lines)
- `victor_invest/handlers.py` (~1936 lines)
- `victor_invest/cli.py` (~1534 lines)

## Strengths and Weaknesses

### Vision
| Area | Strengths | Weaknesses |
|---|---|---|
| Product direction | Strong vertical focus: institutional-grade investment workflows, SEC + valuation + technical + synthesis | Migration narrative is split across legacy and Victor-first language, causing architectural ambiguity |
| Strategic positioning | Correct long-term direction: reuse Victor platform for agentic capabilities | Legacy-first docs and commands reduce confidence in the new operating model |

### Features
| Area | Strengths | Weaknesses |
|---|---|---|
| Domain capabilities | Broad and differentiated toolset (valuation, macro, regime, insider, short interest, peer, RL backtest) | Feature surface is large relative to test depth; many capabilities are weakly validated end-to-end |
| Workflow coverage | YAML workflows for quick/standard/comprehensive/backtest/peer modes | Duplicate execution paths (legacy orchestrator, Victor CLI, graph helpers) increase behavior drift risk |

### Design
| Area | Strengths | Weaknesses |
|---|---|---|
| Architectural pattern | Adopts Victor vertical model and YAML-first workflow provider | Integration bootstrap logic is duplicated between CLI and workflow provider |
| Extensibility | Handler registry and lazy registration pattern are in place | Global tool singletons in graph module create cross-run shared state and test contamination risk |
| Separation of concerns | Business-domain tools/handlers generally separated from framework APIs | Monolithic modules remain in legacy stack, making refactor and performance tuning expensive |

### Implementation
| Area | Strengths | Weaknesses |
|---|---|---|
| Migration quality | Legacy CLI forwarding to Victor exists and eases transition | Make and docs still route users to legacy entrypoints |
| Test posture | Victor unit tests now exist for loading, validation, execution wiring | No strong contract/integration tests proving real tool calls through latest Victor registry APIs |
| Framework compatibility | Updated in this iteration to align imports and tool registration contracts | Compatibility drift can reappear without a pinned version matrix + CI gate |

## Victor Framework Updates Relevant to Victor-Invest
- `VerticalBase` now emphasizes YAML-configurable verticals and SRP-composed internals (`../codingagent/victor/core/verticals/base.py`).
- Explicit handler registry and executor sync are first-class (`../codingagent/victor/framework/handler_registry.py`).
- Base YAML workflow provider reduces provider duplication (`../codingagent/victor/framework/workflows/base_yaml_provider.py`).
- Tool registry now enforces Victor `BaseTool` contracts (`../codingagent/victor/tools/registry.py`).

## Alignment Findings (Victor-Invest vs Victor Framework)
| Capability | Should Live In | Current Status | Action |
|---|---|---|---|
| Agent orchestration lifecycle | Victor framework | Partially duplicated in `victor_invest/cli.py` and workflow provider | Centralize bootstrap in one shared integration module |
| Workflow execution engine | Victor framework | Correctly reused | Keep reuse, remove parallel custom pathways |
| Handler registration + sync | Victor framework APIs | Reused with `ensure_handlers_registered()` | Keep, add CI assertions |
| Tool protocol + execution contract | Victor framework | Was mismatched, now adapted | Keep adapter short-term; migrate tools to native Victor `BaseTool` long-term |
| Investment valuation/analysis logic | Victor-Invest domain | Strong and differentiated | Keep here; optimize and test deeply |

## Before / After
| State | CLI Entry | Tool Registration | Handler Wiring | Outcome |
|---|---|---|---|---|
| Before this iteration | Spec imports pointed at old `victor.agents.spec`; tool registration failed against current registry | Incompatible with Victor `BaseTool` contract | Existing but unverified by passing tests | Victor test collection/execution failures |
| After this iteration | Agent spec imports aligned to `victor.agent.specs.models` | Added protocol adapter in registration path | Existing handler wiring preserved | `tests/unit/victor_invest` passing (`14 passed`) |
| Target architecture | Victor CLI primary, legacy CLI optional/isolated | Native Victor tool classes or stable adapter layer | Single registration bootstrap path | Clean vertical model, lower drift, stronger CI guarantees |

## Progress Update (Continuation)
- Added shared bootstrap module: `victor_invest/framework_bootstrap.py`.
- Consolidated duplicated CLI/provider setup paths to use shared bootstrap:
- `victor_invest/cli.py` now uses `create_investment_orchestrator(...)`.
- `victor_invest/workflows/__init__.py` now uses `create_investment_orchestrator(...)`.
- Updated developer defaults to Victor-first entrypoints:
- `Makefile` analysis/status/cache targets use `python3 -m victor_invest.cli`.
- `docs/ARCHITECTURE.md` and `docs/DEVELOPER_GUIDE.adoc` now point to Victor CLI as primary.
- Added bootstrap-focused unit tests: `tests/unit/victor_invest/test_framework_bootstrap.py`.
- Replaced module-global graph tool singletons with task-scoped cache in `victor_invest/workflows/graphs.py`.
- Added tool-cache behavior test: `tests/unit/victor_invest/test_graph_tool_cache.py`.
- Updated active docs/runbook to Victor-first command paths (`docs/README.adoc`, `docs/DEVELOPER_GUIDE.adoc`, `docs/ARCHITECTURE.md`, `docs/OPERATIONS_RUNBOOK.md`).
- Moved `InvestmentVertical` toward YAML-first configuration using `victor_invest/vertical/config/vertical.yaml` and `victor_invest/vertical/config/investment_system_prompt.txt`.
- Simplified `InvestmentVertical.create_orchestrator()` to reuse shared bootstrap (`victor_invest/framework_bootstrap.py`).
- Added YAML-config loading test: `tests/unit/victor_invest/test_vertical_yaml_config.py`.
- Updated `InvestmentVertical.get_tools()` to use YAML-backed tool lists (with default fallback), reducing duplicated tool configuration.
- Added docs conformance guard: `tests/unit/victor_invest/test_docs_victor_first.py` to prevent legacy CLI command examples in active docs.
- Added CI Victor compatibility matrix in `.github/workflows/ci-cd.yml` (`victor-compat` job) and gated package build on it.
- Pinned supported Victor framework dependency range in `pyproject.toml` to `victor-ai>=0.5.0,<0.6.0`.
- Added handler sync/idempotency contract tests for lazy registration:
  - `tests/unit/victor_invest/test_handler_registration_sync.py`
- Added Victor-first entrypoint conformance tests:
  - `tests/unit/victor_invest/test_victor_entrypoint_conformance.py`
- Fixed bootstrap test coverage gap in `tests/unit/victor_invest/test_framework_bootstrap.py` and added a lifecycle contract for "vertical already registered" behavior.
- Added deterministic workflow golden-output checks for canonical symbols:
  - `tests/unit/victor_invest/test_workflow_golden_outputs.py`
  - `tests/fixtures/victor_invest/golden/standard_aapl.json`
  - `tests/fixtures/victor_invest/golden/standard_msft.json`
- Added explicit workflow latency budget module and benchmark runner:
  - `victor_invest/latency_budgets.py`
  - `scripts/benchmark_victor_workflows.py`
  - `tests/unit/victor_invest/test_latency_budgets.py`
- Added CI workflow performance gate using deterministic stub execution:
  - `.github/workflows/ci-cd.yml` (`performance-benchmark` job)
  - `tests/unit/victor_invest/test_ci_quality_gates.py`
- Started native Victor `BaseTool` migration:
  - Updated `register_investment_tools(...)` to register Victor-native tool instances (removed inline adapter path in `victor_invest/tools/__init__.py`)
  - Promoted local tool base abstraction to inherit Victor `BaseTool` directly (`victor_invest/tools/base.py`), enabling direct registry registration for local tools
  - Converged local `ToolResult` onto Victor `ToolResult` via subclassing in `victor_invest/tools/base.py`
  - Removed residual tool wrapper indirection; registry validation now enforces Victor-native tools directly inside `register_investment_tools(...)`
  - Standardized all tool `execute(...)` signatures to Victor-compatible optional `_exec_ctx` defaults for safe direct and registry invocation
  - Migrated tool implementations to Victor-native factory names (`ToolResult.create_success/create_failure`) and added conformance guard `tests/unit/victor_invest/test_tool_result_factory_conformance.py`
  - Removed legacy `ToolResult.success_result/error_result` aliases after codebase migration (Victor-native factory names only)
  - Tightened `ToolResult.create_success/create_failure` to strict Victor-style signatures (`output` + `metadata`) and removed migration-only `data`/`warnings` factory kwargs
  - Removed local `warnings` field extension from `ToolResult` and migrated warning payloads to `metadata["warnings"]`
  - Removed `ToolResult.data` compatibility alias and completed `.output`-only migration for tools/workflows
  - Added signature contract guard in `tests/unit/victor_invest/test_tool_signature_contract.py`
  - Added result-contract tests in `tests/unit/victor_invest/test_tool_result_contract.py`
  - Extended tool registration contract tests in `tests/unit/victor_invest/test_tool_registration.py`
  - Added optional-dependency import contract tests for synthesis/reporting boundaries in
    `tests/unit/application/test_optional_dependency_import_contract.py`
  - Hardened reporting package imports with lazy loading and reportlab-fallback symbols to
    keep `investigator.application.synthesizer` import-safe in minimal environments
  - Added reporting package-level import contract coverage under blocked `reportlab` in
    `tests/unit/application/test_optional_dependency_import_contract.py`
  - Started monolith decomposition in legacy stack by extracting prompt-safe fundamental
    formatters into `src/investigator/domain/agents/fundamental/formatters.py` with unit tests
  - Continued synthesizer decomposition by extracting text insight/risk parsing into
    `src/investigator/application/synthesizer_text_insights.py` with delegating wrappers in
    `src/investigator/application/synthesizer.py`
  - Extracted deterministic response/cache payload builders into
    `src/investigator/domain/agents/fundamental/deterministic_payloads.py` and reused them in both
    `FundamentalAnalysisAgent` and `DeterministicAnalyzer` to remove duplicated response-shaping logic
  - Continued `FundamentalAnalysisAgent` decomposition by extracting trend/company summary + latest-financial
    extraction helpers into `src/investigator/domain/agents/fundamental/summaries.py` with delegating wrappers
    in `src/investigator/domain/agents/fundamental/agent.py`
  - Replaced in-class trend-analysis implementations in `FundamentalAnalysisAgent` with delegating wrappers to
    `src/investigator/domain/agents/fundamental/trend_analyzer.py` using lazy resolver caching
  - Replaced in-class data-quality implementations in `FundamentalAnalysisAgent` with delegating wrappers to
    `src/investigator/domain/agents/fundamental/data_quality_assessor.py` using lazy resolver caching
  - Replaced in-class deterministic health/growth/profitability implementations in `FundamentalAnalysisAgent`
    with delegating async wrappers to `src/investigator/domain/agents/fundamental/deterministic_analyzer.py`
    using per-agent lazy analyzer caching
  - Extracted valuation rendering/prompt helpers from `FundamentalAnalysisAgent._perform_valuation(...)` into
    `src/investigator/domain/agents/fundamental/valuation_synthesis.py` and switched agent flow to these helpers
  - Extracted multi-model blending preparation helpers (model collection/filtering, applicability-field hydration,
    weight propagation) from `FundamentalAnalysisAgent._perform_valuation(...)` into
    `src/investigator/domain/agents/fundamental/valuation_blending.py`
  - Extracted dynamic/static fallback weighting logic from `FundamentalAnalysisAgent._resolve_fallback_weights(...)`
    into `src/investigator/domain/agents/fundamental/valuation_weighting.py` and switched agent method to delegating wrapper
  - Extracted relative-model computation (P/E, EV/EBITDA, P/S, P/B with insurance P/BV override) from
    `FundamentalAnalysisAgent._perform_valuation(...)` into
    `src/investigator/domain/agents/fundamental/valuation_models.py`
  - Extracted GGM + valuation extension model block (Damodaran DCF, Rule of 40, SaaS) from
    `FundamentalAnalysisAgent._perform_valuation(...)` into
    `src/investigator/domain/agents/fundamental/valuation_extensions.py`
  - Extracted multi-model blend orchestration + valuation summary logging from
    `FundamentalAnalysisAgent._perform_valuation(...)` into
    `src/investigator/domain/agents/fundamental/valuation_orchestrator.py`
  - Extracted deterministic-vs-LLM valuation synthesis dispatch from
    `FundamentalAnalysisAgent._perform_valuation(...)` into
    `src/investigator/domain/agents/fundamental/valuation_orchestrator.py`
  - Added delegation contract tests:
    `tests/unit/domain/agents/fundamental/test_agent_trend_delegation.py`,
    `tests/unit/domain/agents/fundamental/test_agent_data_quality_delegation.py`,
    `tests/unit/domain/agents/fundamental/test_agent_deterministic_delegation.py`
  - Added valuation helper unit coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_synthesis.py`
  - Added blending helper unit coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_blending.py`
  - Added weighting helper/delegation coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_weighting.py`
  - Added relative-model helper coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_models.py`
  - Added valuation-extension helper coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_extensions.py`
  - Added valuation-orchestrator helper coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_orchestrator.py`
  - Added valuation synthesis dispatch coverage in
    `tests/unit/domain/agents/fundamental/test_valuation_synthesis_dispatch.py`
  - Extracted quarterly-ingestion helpers from `FundamentalAnalysisAgent` into
    `src/investigator/domain/agents/fundamental/quarterly_fetch.py`:
    processed-period window query/mapping, cached-quarter normalization, processed-data flattening,
    fallback canonical extraction, and processed-quarter payload mapping
  - Replaced monolithic implementations with delegating wrappers in
    `src/investigator/domain/agents/fundamental/agent.py` for:
    `_fetch_historical_quarters(...)` and `_fetch_from_processed_table(...)`
  - Added quarterly-fetch helper coverage in
    `tests/unit/domain/agents/fundamental/test_quarterly_fetch.py`
  - Extracted company-level processed-table fetch/mapping from
    `FundamentalAnalysisAgent._fetch_company_data_from_processed_table(...)` into
    `src/investigator/domain/agents/fundamental/company_fetch.py`
  - Switched `FundamentalAnalysisAgent` to delegating wrapper for company-level processed snapshot fetch
  - Added company-fetch helper coverage in
    `tests/unit/domain/agents/fundamental/test_company_fetch.py`

## Concrete Issues to Address Next
| Priority | Issue | Evidence | Impact |
|---|---|---|---|
| P1 | Residual legacy command references in historical docs | Multiple historical analysis/session docs under `docs/` still use `cli_orchestrator.py` | Onboarding noise and mixed guidance outside active runbooks |
| P1 | Oversized modules in legacy stack | `src/investigator/domain/agents/fundamental/agent.py`, `src/investigator/application/synthesizer.py` | Slower changes, higher regression risk |
| P2 | Nightly live benchmark depends on runtime secrets and trend reporting | `.github/workflows/nightly-live-benchmark.yml` now runs CLI/live path, but does not yet enforce secret presence or publish long-term trend dashboards | Benchmark data may exist without actionable alerting until trend/alert plumbing is added |

## Roadmap (Robust, Scalable, Performant)

### Phase 0 (Completed in this iteration)
- Fix Victor API drift in agent spec imports.
- Enforce native Victor `BaseTool` registration contract in `register_investment_tools(...)`.
- Validate Victor-focused unit suite.

### Phase 1 (Platform Consolidation, 1-2 weeks)
- Move all Victor bootstrap concerns into one module (vertical registration, role provider, tool enabling, handler sync).
- Refactor `victor_invest/cli.py` and workflow provider to call that single bootstrap utility.
- Update `Makefile` and docs to Victor-first commands.

### Phase 2 (Vertical Purification, 2-3 weeks)
- Convert `InvestmentVertical` to YAML-based vertical config where possible to reduce programmatic boilerplate.
- Keep only investment domain policies/prompts in vertical code.
- Remove duplicate or deprecated graph-execution pathways once YAML workflows are complete.

### Phase 3 (Performance and Reliability, 2-4 weeks)
- Replace module-global tool singletons with scoped factories (per execution context).
- Add latency budgets and benchmark targets for quick/standard/comprehensive modes.
- Add robust cache invalidation and deterministic replay mode for backtests.

### Phase 4 (Quality Gates and Release Readiness, 2 weeks)
- Completed:
- CI matrix against pinned Victor versions.
- tool registration and execution through real Victor registry contract tests.
- handler registration sync assertions.
- workflow golden-output checks for canonical symbols.
- architecture conformance checks (Victor-first active docs + entrypoint assertions).
- latency budget definitions and benchmark runner.
- CI gate for latency budget enforcement on quick/standard/comprehensive benchmark runs.
- nightly live-provider benchmark workflow for end-to-end latency snapshots.
- Pending:
- benchmark trend dashboarding/alert thresholds from nightly artifacts.

## Suggested Operating Model
- `victor-invest` owns:
- domain data modeling, valuation math, domain prompts, investment-specific handlers/tools
- Victor framework owns:
- orchestration, subagent mechanics, workflow runtime, tool protocol, event/observability infrastructure

## Success Metrics
- 100% analysis commands default to Victor path.
- No duplicated bootstrap code across CLI/provider paths.
- Stable CI contract tests against supported Victor versions.
- Reduced large-file risk (progressive decomposition of >3000-line modules).
- Measurable mode latency targets met with reproducible benchmarks.

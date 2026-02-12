import asyncio

from victor_invest import framework_bootstrap


class _FakeToolSelector:
    def __init__(self):
        self.invalidated = False

    def invalidate_tool_cache(self):
        self.invalidated = True


class _FakeOrchestrator:
    def __init__(self):
        self.tools = object()
        self.enabled_tools = None
        self.tool_selector = _FakeToolSelector()

    def set_enabled_tools(self, tools):
        self.enabled_tools = tools


def test_resolve_model_explicit():
    assert framework_bootstrap.resolve_investment_model("ollama", "custom-model") == "custom-model"


def test_resolve_model_non_ollama_defaults_to_none():
    assert framework_bootstrap.resolve_investment_model("anthropic", None) is None


def test_prepare_orchestrator_registers_tools_and_enables_vertical_tools(monkeypatch):
    calls = {}

    def _register(_registry):
        calls["registered"] = True
        return {"registered": 3, "errors": []}

    monkeypatch.setattr(framework_bootstrap, "register_investment_tools", _register)
    monkeypatch.setattr(
        framework_bootstrap.InvestmentVertical,
        "get_tools",
        classmethod(lambda cls: ["sec_filing", "valuation"]),
    )

    orch = _FakeOrchestrator()
    framework_bootstrap.prepare_orchestrator_for_investment(orch)

    assert calls.get("registered") is True
    assert orch.enabled_tools == {"sec_filing", "valuation"}
    assert orch.tool_selector.invalidated is True


def test_prepare_orchestrator_reports_registration_failure(monkeypatch):
    def _boom(_registry):
        raise RuntimeError("registration failed")

    warnings = []
    monkeypatch.setattr(framework_bootstrap, "register_investment_tools", _boom)

    orch = _FakeOrchestrator()
    framework_bootstrap.prepare_orchestrator_for_investment(
        orch,
        warning_callback=warnings.append,
    )

    assert warnings
    assert "registration failed" in warnings[0]


def test_create_investment_orchestrator_bootstrap_flow(monkeypatch):
    class _FakeAgent:
        def __init__(self, orchestrator):
            self._orchestrator = orchestrator

        def get_orchestrator(self):
            return self._orchestrator

    fake_orchestrator = object()
    calls = {
        "ensure_handlers": 0,
        "register_vertical": 0,
        "register_role_provider": 0,
        "prepare": 0,
        "agent_create": 0,
    }

    def _ensure_handlers():
        calls["ensure_handlers"] += 1

    def _vr_get(_name):
        return None

    def _vr_register(_vertical):
        calls["register_vertical"] += 1

    def _register_role_provider():
        calls["register_role_provider"] += 1

    def _prepare(orchestrator, warning_callback=None):
        assert orchestrator is fake_orchestrator
        assert warning_callback is not None
        calls["prepare"] += 1

    async def _agent_create(**kwargs):
        assert kwargs["provider"] == "ollama"
        calls["agent_create"] += 1
        return _FakeAgent(fake_orchestrator)

    monkeypatch.setattr(framework_bootstrap.VerticalRegistry, "get", _vr_get)
    monkeypatch.setattr(framework_bootstrap.VerticalRegistry, "register", _vr_register)
    monkeypatch.setattr(
        framework_bootstrap,
        "register_investment_role_provider",
        _register_role_provider,
    )
    monkeypatch.setattr(
        framework_bootstrap,
        "prepare_orchestrator_for_investment",
        _prepare,
    )
    monkeypatch.setattr(framework_bootstrap.Agent, "create", _agent_create)

    result = asyncio.run(
        framework_bootstrap.create_investment_orchestrator(
            provider="ollama",
            model="test-model",
            ensure_handlers=_ensure_handlers,
            warning_callback=lambda msg: None,
        )
    )

    assert result is fake_orchestrator
    assert calls["ensure_handlers"] == 1
    assert calls["register_vertical"] == 1
    assert calls["register_role_provider"] == 1
    assert calls["prepare"] == 1
    assert calls["agent_create"] == 1


def test_create_investment_orchestrator_skips_vertical_register_when_present(monkeypatch):
    class _FakeAgent:
        def __init__(self, orchestrator):
            self._orchestrator = orchestrator

        def get_orchestrator(self):
            return self._orchestrator

    fake_orchestrator = object()
    calls = {"register_vertical": 0}

    def _vr_get(_name):
        return object()

    def _vr_register(_vertical):
        calls["register_vertical"] += 1

    async def _agent_create(**kwargs):
        return _FakeAgent(fake_orchestrator)

    monkeypatch.setattr(framework_bootstrap.VerticalRegistry, "get", _vr_get)
    monkeypatch.setattr(framework_bootstrap.VerticalRegistry, "register", _vr_register)
    monkeypatch.setattr(framework_bootstrap.Agent, "create", _agent_create)
    monkeypatch.setattr(
        framework_bootstrap,
        "prepare_orchestrator_for_investment",
        lambda orchestrator, warning_callback=None: None,
    )
    monkeypatch.setattr(
        framework_bootstrap,
        "register_investment_role_provider",
        lambda: None,
    )

    result = asyncio.run(
        framework_bootstrap.create_investment_orchestrator(
            provider="ollama",
            model="test-model",
        )
    )

    assert result is fake_orchestrator
    assert calls["register_vertical"] == 0

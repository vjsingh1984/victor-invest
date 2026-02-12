import victor_invest.workflows as workflows


def test_ensure_handlers_registered_is_idempotent(monkeypatch):
    import victor.framework.handler_registry as handler_registry_module
    import victor_invest.handlers as handlers_module

    calls = {"register_handlers": 0, "sync_handlers": 0}

    monkeypatch.setattr(workflows, "_handlers_registered", False)

    def _register_handlers():
        calls["register_handlers"] += 1

    def _sync_handlers_with_executor(*, direction):
        calls["sync_handlers"] += 1
        assert direction == "to_executor"

    monkeypatch.setattr(handlers_module, "register_handlers", _register_handlers)
    monkeypatch.setattr(handler_registry_module, "sync_handlers_with_executor", _sync_handlers_with_executor)

    workflows.ensure_handlers_registered()
    workflows.ensure_handlers_registered()

    assert calls["register_handlers"] == 1
    assert calls["sync_handlers"] == 1


def test_ensure_handlers_registered_short_circuits_when_marked_done(monkeypatch):
    monkeypatch.setattr(workflows, "_handlers_registered", True)
    workflows.ensure_handlers_registered()
    assert workflows._handlers_registered is True

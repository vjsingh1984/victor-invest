from victor.framework.handler_registry import get_handler_registry
from victor.workflows.executor import get_compute_handler
from victor.workflows.definition import ComputeNode

from victor_invest.workflows import InvestmentWorkflowProvider, ensure_handlers_registered


def test_workflows_validate_and_handlers_resolve():
    ensure_handlers_registered()
    registry = get_handler_registry()

    provider = InvestmentWorkflowProvider()
    workflows = provider.get_workflows()
    assert workflows

    workflow_errors = {}
    missing_handlers = {}

    for name, workflow in workflows.items():
        errors = workflow.validate()
        if errors:
            workflow_errors[name] = errors

        for node in workflow.nodes.values():
            if isinstance(node, ComputeNode) and node.handler:
                handler = registry.get(node.handler)
                if handler is None:
                    missing_handlers.setdefault(name, []).append(node.handler)
                if get_compute_handler(node.handler) is None:
                    missing_handlers.setdefault(name, []).append(node.handler)

    assert not workflow_errors, f"Workflow validation errors: {workflow_errors}"
    assert not missing_handlers, f"Missing handlers: {missing_handlers}"

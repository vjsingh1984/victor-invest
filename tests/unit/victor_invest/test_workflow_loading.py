from victor_invest.workflows import InvestmentWorkflowProvider


def test_yaml_workflows_load():
    provider = InvestmentWorkflowProvider()
    names = provider.get_workflow_names()

    for expected in [
        "quick",
        "standard",
        "comprehensive",
        "rl_backtest",
        "peer_comparison",
    ]:
        assert expected in names

from victor_invest.vertical import InvestmentVertical


def test_investment_vertical_loads_yaml_config():
    InvestmentVertical.clear_config_cache()
    config = InvestmentVertical.get_config(use_cache=False, use_yaml=True)

    assert config.name == "investment"
    assert "sec_filing" in config.tools.tools
    assert "entry_exit_signals" in config.tools.tools
    assert "DATA_GATHERING" in config.stages
    assert "preferred_providers" in config.provider_hints
    assert any("Data accuracy and completeness" == item for item in config.evaluation_criteria)


def test_investment_vertical_get_tools_uses_yaml_backed_toolset():
    InvestmentVertical.clear_config_cache()
    tools = InvestmentVertical.get_tools()

    assert "sec_filing" in tools
    assert "entry_exit_signals" in tools

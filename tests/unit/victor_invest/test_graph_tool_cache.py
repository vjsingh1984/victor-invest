import asyncio

from victor_invest.workflows import graphs


def test_tool_cache_reuses_within_task_and_isolates_across_tasks(monkeypatch):
    created = {"count": 0}

    class DummySECTool:
        def __init__(self):
            created["count"] += 1

    monkeypatch.setattr(graphs, "SECFilingTool", DummySECTool)
    graphs._task_tool_cache.clear()

    async def _same_task():
        first = await graphs._get_sec_tool()
        second = await graphs._get_sec_tool()
        return first, second

    async def _two_tasks():
        async def _one():
            return await graphs._get_sec_tool()

        return await asyncio.gather(_one(), _one())

    first, second = asyncio.run(_same_task())
    assert first is second

    a, b = asyncio.run(_two_tasks())
    assert a is not b
    assert created["count"] == 3

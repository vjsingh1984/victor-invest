from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from investigator.infrastructure.cache.cache_types import CacheType
from investigator.infrastructure.cache.parquet_cache_handler import ParquetCacheStorageHandler


def _make_handler(tmp_path: Path) -> ParquetCacheStorageHandler:
    return ParquetCacheStorageHandler(
        cache_type=CacheType.TECHNICAL_DATA,
        base_path=tmp_path / "parquet-cache",
        priority=5,
    )


def _make_dataframe(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    # ensure deterministic column order
    return df[[column for column in sorted(df.columns)]]


def test_parquet_handler_round_trip(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="D"),
            "close": [100.0, 101.5, 102.0],
            "volume": [1_000_000, 1_050_000, 990_000],
        }
    )
    payload = {"dataframe": df, "metadata": {"symbol": "ROUND", "timeframe": "3d"}}
    key = ("ROUND", "technical_data", "3d")

    assert handler.set(key, payload)
    assert handler.exists(key)

    stored = handler.get(key)
    assert stored is not None
    pd.testing.assert_frame_equal(stored["dataframe"].reset_index(drop=True), df.reset_index(drop=True))
    assert stored["cache_info"]["records"] == len(df)
    assert stored["metadata"]["original_metadata"]["symbol"] == "ROUND"

    info = handler.get_cache_info(key)
    assert info and info["cache_key"]["symbol"] == "ROUND"

    assert handler.delete(key)
    assert handler.get(key) is None


def test_parquet_handler_accepts_record_payload(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    records = [
        {"timestamp": "2025-01-01", "close": 100.0},
        {"timestamp": "2025-01-02", "close": 101.0},
    ]
    payload = {"data": records, "metadata": {"symbol": "REC", "timeframe": "2d"}}
    key = {"symbol": "REC", "data_type": "technical_data", "timeframe": "2d"}

    assert handler.set(key, payload)
    fetched = handler.get(key)
    assert fetched is not None
    assert set(fetched["dataframe"].columns) == {"timestamp", "close"}


def test_parquet_handler_symbol_cleanup(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    df = _make_dataframe(
        [
            {"timestamp": "2025-01-01", "close": 100.0},
            {"timestamp": "2025-01-02", "close": 101.5},
        ]
    )
    payload = {"dataframe": df, "metadata": {"symbol": "DEL", "timeframe": "2d"}}

    key = ("DEL", "technical_data", "2d")
    handler.set(key, payload)
    assert handler.exists(key)

    deleted_files = handler.delete_by_symbol("DEL")
    assert deleted_files >= 1
    assert handler.exists(key) is False

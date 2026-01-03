#!/usr/bin/env python

import json

# Load files
with open("data/sector_mapping.json") as f:
    sector_mapping = json.load(f)

with open("data/russell_1000_peer_groups.json") as f:
    peer_groups = json.load(f)["peer_groups"]

# Create reverse index from peer groups
peer_group_data = {}

for sector, groups in peer_groups.items():
    for group_name, details in groups.items():
        for cap_class in ["large_cap", "mid_cap"]:
            for symbol in details.get(cap_class, []):
                peer_group_data[symbol] = {
                    "peer_group_sector": sector,
                    "peer_group": group_name,
                    "market_cap_class": cap_class,
                }

# Merge with sector mapping
combined_data = []
symbols = set(sector_mapping) | set(peer_group_data)

for symbol in symbols:
    data = {
        "symbol": symbol,
        "sector": sector_mapping.get(symbol),
        **peer_group_data.get(symbol, {"peer_group_sector": None, "peer_group": None, "market_cap_class": None}),
    }
    combined_data.append(data)

# Optionally: convert to a DataFrame
import pandas as pd

df = pd.DataFrame(combined_data)
df.set_index("symbol", inplace=True)  # symbol as primary key

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.types import Numeric
import pandas as pd
from decimal import Decimal


def getConnectionEngine(user, password, database, host, port):
    return create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{database}".format(
            user=user, password=password, database=database, host=host, port=port
        )
    )


connEngine = getConnectionEngine(
    user="stockuser", password=os.environ.get("STOCK_DB_PASSWORD", ""), database="stock", host="${DB_HOST:-localhost}", port=5432
)

df.to_sql(name="peermapping", con=connEngine, index=True, if_exists="replace")

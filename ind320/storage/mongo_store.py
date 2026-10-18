from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
from pymongo import MongoClient, ReplaceOne

from ind320.config import get_setting
from ind320.data.transfers import (
    SOURCE,
    annual_totals,
    clip_intervals,
    month_bounds,
    prepare_intervals,
)


@contextmanager
def mongo_database() -> Iterator:
    """Open MongoDB with private settings and close the connection after use."""
    with MongoClient(
        get_setting("MONGODB_URI"),
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        tz_aware=True,
    ) as client:
        yield client[get_setting("MONGODB_DATABASE", "ind320")]


def publish(
    data: pd.DataFrame, years: tuple[int, ...] = (2024, 2025)
) -> pd.DataFrame:
    """Upsert intervals and summaries derived exclusively from Cassandra extraction."""
    data = prepare_intervals(data)
    summaries = annual_totals(data, years)
    with mongo_database() as db:
        db.transfer_intervals.create_index([("connection", 1), ("start", 1)])
        # Read existing curated records once. Unchanged notebook reruns then avoid
        # hundreds of thousands of needless writes to a small Atlas cluster.
        existing = {
            doc["_id"]: doc for doc in db.transfer_intervals.find({}, batch_size=5000)
        }
        operations = []
        for row in data.itertuples():
            key = f"{row.connection}|{row.direction}|{row.start.isoformat()}"
            document = {
                "_id": key,
                "connection": row.connection,
                "direction": row.direction,
                "start": row.start.to_pydatetime(),
                "end": row.end.to_pydatetime(),
                "power_mw": float(row.power_mw),
                "energy_mwh": float(row.energy_mwh),
                "source": SOURCE,
                "power_unit": "MW",
                "energy_unit": "MWh",
            }
            if existing.get(key) == document:
                continue
            operations.append(ReplaceOne({"_id": key}, document, upsert=True))
            if len(operations) == 2000:
                db.transfer_intervals.bulk_write(operations, ordered=False)
                operations.clear()
        if operations:
            db.transfer_intervals.bulk_write(operations, ordered=False)
        annual_operations = []
        for row in summaries.to_dict("records"):
            # BSON has no NaN-as-missing convention; publish null for no energy data.
            if pd.isna(row["energy_mwh"]):
                row["energy_mwh"] = None
            row.update(source=SOURCE, energy_unit="MWh")
            key = f"{row['connection']}|{row['year']}|{row['direction']}"
            annual_operations.append(
                ReplaceOne({"_id": key}, {"_id": key, **row}, upsert=True)
            )
        if annual_operations:
            db.transfer_annual.bulk_write(annual_operations, ordered=False)
    return summaries


def available_transfers() -> pd.DataFrame:
    """Discover published connection/year choices, including incomplete coverage."""
    with mongo_database() as db:
        return pd.DataFrame(list(db.transfer_annual.find({}, {"_id": 0})))


def query_annual(connection: str, year: int) -> pd.DataFrame:
    """Read the same annual summaries used by notebook charts."""
    with mongo_database() as db:
        return pd.DataFrame(
            list(
                db.transfer_annual.find(
                    {"connection": connection, "year": year}, {"_id": 0}
                )
            )
        )


def query_months(connection: str, year: int, months: list[int]) -> pd.DataFrame:
    """Read selected local calendar months only, using indexed UTC ranges."""
    frames = []
    with mongo_database() as db:
        for month in sorted(set(months)):
            start, end = month_bounds(year, month)
            documents = list(
                db.transfer_intervals.find(
                    {
                        "connection": connection,
                        "start": {"$lt": end.to_pydatetime()},
                        "end": {"$gt": start.to_pydatetime()},
                    },
                    {"_id": 0},
                )
            )
            if documents:
                frames.append(
                    clip_intervals(
                        prepare_intervals(pd.DataFrame(documents)),
                        start,
                        end,
                    )
                )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

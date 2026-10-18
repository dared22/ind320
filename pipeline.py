"""Local notebook workflow: Spark → Cassandra → Spark → curated MongoDB.

Only this module imports local processing dependencies; Streamlit never does.
"""

import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from pyspark.sql import SparkSession

from config import ConfigurationError, get_setting
from database_checks import SPARK_CASSANDRA_PACKAGE
from transfers import COLUMNS, validate_intervals


@contextmanager
def local_databases() -> Iterator[tuple[SparkSession, str]]:
    """Open the configured local services and release them even if a cell fails."""
    keyspace = get_setting("CASSANDRA_KEYSPACE", "ind320")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", keyspace):
        raise ConfigurationError(
            "Use a lowercase CQL identifier for CASSANDRA_KEYSPACE."
        )
    host = get_setting("CASSANDRA_HOST", "127.0.0.1")
    port = int(get_setting("CASSANDRA_PORT", "9042"))
    username, password = (
        get_setting("CASSANDRA_USERNAME"),
        get_setting("CASSANDRA_PASSWORD"),
    )
    if bool(username) != bool(password):
        raise ConfigurationError("Set both Cassandra authentication settings.")
    auth = PlainTextAuthProvider(username, password) if username else None
    if get_setting("JAVA_HOME"):
        os.environ["JAVA_HOME"] = get_setting("JAVA_HOME")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    cluster = Cluster([host], port=port, auth_provider=auth, connect_timeout=10)
    spark = None
    try:
        session = cluster.connect()
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = "
            "{'class': 'SimpleStrategy', 'replication_factor': 1}"
        )
        session.execute(
            f"CREATE TABLE IF NOT EXISTS {keyspace}.physical_flows ("
            "connection text, direction text, year int, start timestamp, "
            "end timestamp, power_mw double, energy_mwh double, "
            "PRIMARY KEY ((connection, direction, year), start))"
        )
        builder = (
            SparkSession.builder.master("local[2]")
            .appName("IND320 transfers")
            .config("spark.jars.packages", SPARK_CASSANDRA_PACKAGE)
            .config("spark.jars.ivy", str(Path(__file__).parent / ".local/ivy"))
            .config("spark.cassandra.connection.host", host)
            .config("spark.cassandra.connection.port", str(port))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .config("spark.redaction.regex", "(?i)secret|password|token|username")
        )
        if username:
            builder = builder.config("spark.cassandra.auth.username", username).config(
                "spark.cassandra.auth.password", password
            )
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        yield spark, keyspace
    finally:
        if spark is not None:
            spark.stop()
        cluster.shutdown()


def write_cassandra(spark: SparkSession, keyspace: str, data: pd.DataFrame) -> None:
    """Upsert stable interval keys with Spark; reruns overwrite the same records."""
    data = validate_intervals(data)
    if data.empty:
        return
    # UTC epoch milliseconds avoid host-timezone conversion of naive datetimes.
    rows = [
        (
            r.connection,
            r.direction,
            int(r.start.tz_convert("Europe/Oslo").year),
            int(r.start.timestamp() * 1000),
            int(r.end.timestamp() * 1000),
            float(r.power_mw),
            float(r.energy_mwh),
        )
        for r in data.itertuples()
    ]
    from pyspark.sql import functions as F

    frame = spark.createDataFrame(
        rows,
        "connection string, direction string, year int, "
        "start_ms long, end_ms long, power_mw double, energy_mwh double",
    )
    frame = (
        frame.withColumn("start", F.timestamp_millis("start_ms"))
        .withColumn("end", F.timestamp_millis("end_ms"))
        .drop("start_ms", "end_ms")
    )
    frame.write.format("org.apache.spark.sql.cassandra").options(
        keyspace=keyspace, table="physical_flows"
    ).mode("append").save()


def read_cassandra(
    spark: SparkSession, keyspace: str, years: tuple[int, ...] = (2024, 2025)
) -> pd.DataFrame:
    """Extract only relevant fields through Spark, then validate before publishing."""
    from pyspark.sql import functions as F

    frame = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(keyspace=keyspace, table="physical_flows")
        .load()
        .filter(F.col("year").isin(list(years)))
    )
    # Explicit epoch conversion also makes reads independent of the host timezone.
    rows = frame.select(
        "connection",
        "direction",
        F.unix_millis("start").alias("start_ms"),
        F.unix_millis("end").alias("end_ms"),
        "power_mw",
        "energy_mwh",
    ).collect()
    result = pd.DataFrame(
        [r.asDict() for r in rows],
        columns=[
            "connection",
            "direction",
            "start_ms",
            "end_ms",
            "power_mw",
            "energy_mwh",
        ],
    )
    for col in ("start", "end"):
        result[col] = pd.to_datetime(result.pop(f"{col}_ms"), unit="ms", utc=True)
    return validate_intervals(result[COLUMNS])

"""Small real-service checks for the local notebook environment.

These checks touch only a uniquely identified temporary row/document. They do
not ingest transfers or publish project data (those belong to later plan parts).
"""

import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from config import ConfigurationError, get_setting, require_setting

# Match the exact Spark/Scala combination in the official course Spark example.
SPARK_CASSANDRA_PACKAGE = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1"


def mongo_round_trip() -> str:
    """Create, read, update, and delete our temporary MongoDB document."""
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    uri = require_setting("MONGODB_URI")
    database = get_setting("MONGODB_DATABASE", "ind320")
    marker = f"ind320-check-{uuid4()}"
    try:
        with MongoClient(
            uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000
        ) as client:
            collection = client[database]["configuration_checks"]
            try:
                collection.insert_one({"_id": marker, "value": 1})
                if collection.find_one({"_id": marker})["value"] != 1:
                    raise RuntimeError("MongoDB create/read check failed.")
                collection.update_one({"_id": marker}, {"$set": {"value": 2}})
                if collection.find_one({"_id": marker})["value"] != 2:
                    raise RuntimeError("MongoDB update/read check failed.")
            finally:
                collection.delete_one({"_id": marker})
            if collection.find_one({"_id": marker}) is not None:
                raise RuntimeError("MongoDB delete check failed.")
    except (PyMongoError, ValueError):
        # Database driver errors can contain URIs, addresses, and user names.
        raise ConfigurationError(
            "MongoDB check failed. Check private settings, network access, and database permissions."
        ) from None
    return "MongoDB create/read/update/delete passed; temporary document removed."


def _cassandra_round_trip() -> str:
    """Write and read a unique Cassandra row through Spark, then remove it."""
    # Imports stay inside the local check: the deployed app never imports Spark.
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster
    from pyspark.sql import SparkSession

    java_home = get_setting("JAVA_HOME")
    if java_home:
        os.environ["JAVA_HOME"] = java_home
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    host = get_setting("CASSANDRA_HOST", "127.0.0.1")
    port = int(get_setting("CASSANDRA_PORT", "9042"))
    keyspace = get_setting("CASSANDRA_KEYSPACE", "ind320")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", keyspace):
        raise ConfigurationError(
            "CASSANDRA_KEYSPACE must be a lowercase CQL identifier."
        )
    username = get_setting("CASSANDRA_USERNAME")
    password = get_setting("CASSANDRA_PASSWORD")
    if bool(username) != bool(password):
        raise ConfigurationError(
            "Set both Cassandra authentication settings, or leave both empty."
        )
    auth = PlainTextAuthProvider(username, password) if username else None
    marker = f"ind320-check-{uuid4()}"
    spark = None
    cluster = Cluster([host], port=port, auth_provider=auth, connect_timeout=10)
    try:
        session = cluster.connect()
        # SimpleStrategy with one replica is for this local single-node check.
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = "
            "{'class': 'SimpleStrategy', 'replication_factor': 1}"
        )
        session.execute(
            f"CREATE TABLE IF NOT EXISTS {keyspace}.configuration_checks "
            "(id text PRIMARY KEY, value int)"
        )
        builder = (
            SparkSession.builder.master("local[2]")
            .appName("IND320 configuration check")
            .config("spark.jars.packages", SPARK_CASSANDRA_PACKAGE)
            .config(
                "spark.jars.ivy",
                str(Path(__file__).parent / ".local" / "ivy"),
            )
            .config("spark.cassandra.connection.host", host)
            .config("spark.cassandra.connection.port", str(port))
            .config(
                "spark.sql.extensions",
                "com.datastax.spark.connector.CassandraSparkExtensions",
            )
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .config("spark.redaction.regex", "(?i)secret|password|token|username")
        )
        if username:
            builder = builder.config("spark.cassandra.auth.username", username).config(
                "spark.cassandra.auth.password", password
            )
        try:
            spark = builder.getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            spark.createDataFrame([(marker, 42)], "id string, value int").write.format(
                "org.apache.spark.sql.cassandra"
            ).options(keyspace=keyspace, table="configuration_checks").mode(
                "append"
            ).save()
            rows = (
                spark.read.format("org.apache.spark.sql.cassandra")
                .options(keyspace=keyspace, table="configuration_checks")
                .load()
                .filter(f"id = '{marker}'")
                .collect()
            )
            if len(rows) != 1 or rows[0]["value"] != 42:
                raise RuntimeError("Spark/Cassandra read-back check failed.")
        finally:
            session.execute(
                f"DELETE FROM {keyspace}.configuration_checks WHERE id = %s", (marker,)
            )
    finally:
        if spark is not None:
            spark.stop()
        cluster.shutdown()
    return "Spark wrote/read a Cassandra record; temporary row removed."


def cassandra_round_trip() -> str:
    """Run the local Spark check with safe, actionable connection errors."""
    from cassandra import DriverException
    from py4j.protocol import Py4JError
    from pyspark.errors import PySparkException

    try:
        return _cassandra_round_trip()
    except (DriverException, Py4JError, PySparkException, OSError, ValueError):
        raise ConfigurationError(
            "Spark/Cassandra check failed. Check Java 17, the notebook dependency "
            "group, the published Cassandra port, and private connection settings."
        ) from None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "service", choices=("mongo", "cassandra", "all"), default="all", nargs="?"
    )
    args = parser.parse_args()
    # Only safe, fixed status messages are printed; no connection settings.
    if args.service in ("mongo", "all"):
        print(mongo_round_trip())
    if args.service in ("cassandra", "all"):
        print(cassandra_round_trip())

import argparse
import hashlib
import json
import os
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from transforms import SOURCE_COLUMNS, build_gold, build_silver, read_csv

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "data" / "sensors.csv"
SCHEMA_VERSION = 3


def ensure_schema(cursor):
    # Coordinate schema initialization and publication across CLI runs.
    cursor.execute("SELECT pg_advisory_xact_lock(2017, 1)")
    cursor.execute((PROJECT_ROOT / "schema.sql").read_text(encoding="utf-8"))


def seed_bronze(database_url, source):
    payload = source.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    with psycopg.connect(database_url) as conn, conn.cursor() as cursor:
        ensure_schema(cursor)
        cursor.execute(
            """
            INSERT INTO bronze.csv_imports (source_sha256, filename, payload)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_sha256) DO NOTHING
            RETURNING source_sha256
            """,
            (checksum, source.name, payload),
        )
        if cursor.fetchone() is None:
            return checksum, False

        with cursor.copy(
            """
            COPY bronze.events ("Timestamp", "car-id", "car-type", "gate-name",
                                source_sha256, source_file_line) FROM STDIN
            """
        ) as copy:
            for line, row in read_csv(payload):
                if len(row) != len(SOURCE_COLUMNS):
                    raise ValueError(f"CSV line {line}: expected {len(SOURCE_COLUMNS)} fields")
                copy.write_row((*row, checksum, line))
    return checksum, True


def copy_rows(cursor, relation, snapshot_id, rows):
    if not rows:
        return
    columns = tuple(rows[0])
    query = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(*relation),
        sql.SQL(", ").join(map(sql.Identifier, ("snapshot_id", *columns))),
    )
    with cursor.copy(query) as copy:
        for row in rows:
            copy.write_row((snapshot_id, *(row[column] for column in columns)))


def publish(database_url):
    with psycopg.connect(database_url) as conn, conn.cursor() as cursor:
        ensure_schema(cursor)
        cursor.execute("""
            SELECT event_id, ARRAY["Timestamp", "car-id", "car-type", "gate-name"]
            FROM bronze.events
            ORDER BY event_id
        """)
        raw_events = cursor.fetchall()
        if not raw_events:
            raise ValueError("bronze.events is empty. Run seed or insert raw events into Bronze.")

        payload = json.dumps(raw_events, ensure_ascii=False, separators=(",", ":"))
        checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        snapshot_id = f"v{SCHEMA_VERSION}-{checksum}"
        cursor.execute(
            "SELECT 1 FROM public.vast_snapshots WHERE snapshot_id = %s", (snapshot_id,)
        )
        if cursor.fetchone() is not None:
            return snapshot_id, False

        events, rejected = build_silver(raw_events)
        trips, traffic = build_gold(events)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bronze_sha256": checksum,
            "rows": {
                "bronze": len(raw_events),
                "silver": len(events),
                "rejected": len(rejected),
                "trips": len(trips),
                "hourly_traffic": len(traffic),
            },
        }
        cursor.execute(
            "INSERT INTO public.vast_snapshots (snapshot_id, manifest) VALUES (%s, %s)",
            (snapshot_id, Jsonb(manifest)),
        )
        copy_rows(cursor, ("silver", "events"), snapshot_id, events)
        copy_rows(cursor, ("silver", "rejected_events"), snapshot_id, rejected)
        copy_rows(cursor, ("gold", "trips"), snapshot_id, trips)
        copy_rows(cursor, ("gold", "hourly_traffic"), snapshot_id, traffic)

    return snapshot_id, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Bronze or publish Silver and Gold.")
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed", help="Import the initial CSV into Bronze.")
    seed.add_argument("--source", type=Path, default=SOURCE)
    commands.add_parser("process", help="Read Bronze and publish Silver and Gold.")
    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("Set DATABASE_URL to the PostgreSQL connection string.")

    if args.command == "seed":
        try:
            checksum, added = seed_bronze(database_url, args.source)
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        status = "Imported" if added else "Already imported"
        print(f"{status} {checksum}")
    else:
        try:
            snapshot_id, published = publish(database_url)
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        status = "Published" if published else "Already published"
        print(f"{status} {snapshot_id}")

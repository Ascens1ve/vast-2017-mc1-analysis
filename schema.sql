CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS public.vast_snapshots (
    snapshot_id text PRIMARY KEY,
    manifest jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS bronze.csv_imports (
    source_sha256 text PRIMARY KEY,
    filename text NOT NULL,
    payload bytea NOT NULL
);

CREATE TABLE IF NOT EXISTS bronze.events (
    event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    "Timestamp" text NOT NULL,
    "car-id" text NOT NULL,
    "car-type" text NOT NULL,
    "gate-name" text NOT NULL,
    source_sha256 text REFERENCES bronze.csv_imports (source_sha256),
    source_file_line bigint,
    CHECK ((source_sha256 IS NULL) = (source_file_line IS NULL)),
    UNIQUE (source_sha256, source_file_line)
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'bronze' AND table_name = 'events' AND column_name = 'raw_fields'
    ) THEN
        IF EXISTS (SELECT 1 FROM bronze.events WHERE cardinality(raw_fields) <> 4) THEN
            RAISE EXCEPTION 'Bronze contains an existing row with a field count other than four';
        END IF;

        ALTER TABLE bronze.events
            ADD COLUMN "Timestamp" text,
            ADD COLUMN "car-id" text,
            ADD COLUMN "car-type" text,
            ADD COLUMN "gate-name" text;

        UPDATE bronze.events
        SET "Timestamp" = raw_fields[1],
            "car-id" = raw_fields[2],
            "car-type" = raw_fields[3],
            "gate-name" = raw_fields[4];

        ALTER TABLE bronze.events
            ALTER COLUMN "Timestamp" SET NOT NULL,
            ALTER COLUMN "car-id" SET NOT NULL,
            ALTER COLUMN "car-type" SET NOT NULL,
            ALTER COLUMN "gate-name" SET NOT NULL;

        IF to_regclass('bronze.csv_rows') IS NOT NULL THEN
            CREATE OR REPLACE VIEW bronze.csv_rows AS
            SELECT "Timestamp", "car-id", "car-type", "gate-name" FROM bronze.events;
        END IF;

        ALTER TABLE bronze.events DROP COLUMN raw_fields;
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS silver.events (
    snapshot_id text REFERENCES public.vast_snapshots (snapshot_id),
    "source_line" bigint NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    "car_id" text NOT NULL,
    "car_type" text NOT NULL,
    "gate_name" text NOT NULL,
    "sensor_type" text NOT NULL,
    "date" date NOT NULL,
    "hour" smallint NOT NULL,
    "dow" smallint NOT NULL,
    "month" smallint NOT NULL,
    "visit_order" integer NOT NULL,
    "prev_gate" text,
    "next_gate" text,
    "elapsed_seconds" bigint,
    PRIMARY KEY (snapshot_id, car_id, visit_order)
);

CREATE TABLE IF NOT EXISTS silver.rejected_events (
    snapshot_id text REFERENCES public.vast_snapshots (snapshot_id),
    "source_line" bigint NOT NULL,
    "reason" text NOT NULL,
    "raw_fields" text[] NOT NULL,
    PRIMARY KEY (snapshot_id, source_line)
);

CREATE TABLE IF NOT EXISTS gold.trips (
    snapshot_id text REFERENCES public.vast_snapshots (snapshot_id),
    "car_id" text NOT NULL,
    "car_type" text NOT NULL,
    "first_gate" text NOT NULL,
    "last_gate" text NOT NULL,
    "first_time" timestamp without time zone NOT NULL,
    "last_time" timestamp without time zone NOT NULL,
    "n_events" bigint NOT NULL,
    "n_unique_gates" bigint NOT NULL,
    "n_unique_sensors" bigint NOT NULL,
    "visited_camping" boolean NOT NULL,
    "route" text NOT NULL,
    "visited_ranger" boolean NOT NULL,
    "visited_gate_restr" boolean NOT NULL,
    "entry_hour" smallint NOT NULL,
    "entry_dow" smallint NOT NULL,
    "exit_hour" smallint NOT NULL,
    "exit_dow" smallint NOT NULL,
    "duration_h" double precision NOT NULL,
    "is_weekend" boolean NOT NULL,
    "is_thrutraffic" boolean NOT NULL,
    "is_ranger" boolean NOT NULL,
    "is_overnight" boolean NOT NULL,
    PRIMARY KEY (snapshot_id, car_id)
);

CREATE TABLE IF NOT EXISTS gold.hourly_traffic (
    snapshot_id text REFERENCES public.vast_snapshots (snapshot_id),
    "date" date NOT NULL,
    "hour" smallint NOT NULL,
    "gate_name" text NOT NULL,
    "car_type" text NOT NULL,
    "event_count" bigint NOT NULL,
    "car_id_count" bigint NOT NULL,
    PRIMARY KEY (snapshot_id, date, hour, gate_name, car_type)
);

-- Ordered event sequences for the ML consumer; preserve earlier snapshots as well.
CREATE TABLE IF NOT EXISTS gold.route_events (
    snapshot_id text NOT NULL REFERENCES public.vast_snapshots (snapshot_id),
    car_id text NOT NULL, visit_order integer NOT NULL,
    car_type text NOT NULL, timestamp timestamp without time zone NOT NULL,
    gate_name text NOT NULL,
    PRIMARY KEY (snapshot_id, car_id, visit_order)
);
INSERT INTO gold.route_events (snapshot_id, car_id, visit_order, car_type, timestamp, gate_name)
SELECT s.snapshot_id, s.car_id, s.visit_order, s.car_type, s.timestamp, s.gate_name
FROM silver.events s
WHERE NOT EXISTS (SELECT 1 FROM gold.route_events g WHERE g.snapshot_id = s.snapshot_id)
ON CONFLICT DO NOTHING;

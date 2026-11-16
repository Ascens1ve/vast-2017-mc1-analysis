"""Score ordered routes from a published Gold snapshot using the V3 service."""
import argparse
from itertools import groupby
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg

MAX_BODY = 49_000_000


def routes_from_rows(rows):
    routes = []
    for car_id, group in groupby(rows, key=lambda row: row[0]):
        events = list(group)
        kinds = {row[1] for row in events}
        if len(kinds) != 1:
            raise ValueError(f"У маршрута {car_id!r} несколько типов автомобиля: {sorted(kinds)}")
        routes.append({"car_id": car_id, "car_type": events[0][1], "events": [
            {"timestamp": row[2].strftime("%Y-%m-%d %H:%M:%S"), "gate_name": row[3]}
            for row in events
        ]})
    return routes


def read_routes(database_url, snapshot):
    with psycopg.connect(database_url) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM public.vast_snapshots WHERE snapshot_id = %s", (snapshot,))
        if cursor.fetchone() is None:
            raise ValueError(f"Снимок не найден: {snapshot}")
        cursor.execute("""SELECT car_id, car_type, timestamp, gate_name FROM gold.route_events
                          WHERE snapshot_id = %s ORDER BY car_id, visit_order""", (snapshot,))
        routes = routes_from_rows(cursor.fetchall())
    if not routes:
        raise ValueError("В Gold нет маршрутов для снимка. Выполните process для миграции и построения Gold.")
    return routes


def encode(routes):
    return json.dumps({"routes": routes}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def post(base_url, endpoint, body):
    request = Request(base_url.rstrip("/") + endpoint, data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=3600) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Ошибка ML API {error.code}: {error.read().decode('utf-8', errors='replace')}") from error
    except URLError as error:
        raise RuntimeError(f"ML API недоступен: {error.reason}") from error


def batches(routes, count):
    batch, size = [], len(encode([]))
    for route in routes:
        route_size = len(json.dumps(route, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
        if route_size + len(encode([])) > MAX_BODY:
            raise ValueError(f"Маршрут {route['car_id']!r} превышает лимит запроса ML API")
        if batch and (len(batch) >= count or size + route_size > MAX_BODY):
            yield batch
            batch, size = [], len(encode([]))
        batch.append(route)
        size += route_size
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "train"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--ml-url", default=os.getenv("ML_URL", "http://127.0.0.1:8767"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if not args.database_url or args.batch_size <= 0:
        parser.error("Укажите DATABASE_URL и положительный --batch-size")
    try:
        routes = read_routes(args.database_url, args.snapshot)
        if args.command == "train":
            body = encode(routes)
            if len(body) > MAX_BODY:
                raise ValueError("Снимок превышает лимит обучения ML API; разделять обучение на запросы нельзя")
            result = post(args.ml_url, "/train", body)
        else:
            scores = []
            for batch in batches(routes, args.batch_size):
                response = post(args.ml_url, "/score/batch", encode(batch))
                received = response["scores"]
                if {r["car_id"] for r in received} != {r["car_id"] for r in batch} or len(received) != len(batch):
                    raise ValueError("ML API вернул неверный состав маршрутов")
                scores.extend(received)
            result = {"snapshot_id": args.snapshot, "scores": scores}
        print(json.dumps(result, ensure_ascii=False))
    except (ValueError, RuntimeError, psycopg.Error, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

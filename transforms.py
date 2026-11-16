import csv
from collections import Counter
from datetime import datetime
from io import BytesIO, TextIOWrapper
from itertools import groupby

SOURCE_COLUMNS = ["Timestamp", "car-id", "car-type", "gate-name"]
CAR_TYPES = {"1", "2", "2P", "3", "4", "5", "6"}
SENSOR_TYPES = {
    "Entrance": [f"entrance{i}" for i in range(5)],
    "General-Gate": [f"general-gate{i}" for i in range(8)],
    "Gate (restricted)": [f"gate{i}" for i in range(9)],
    "Ranger-Stop": [f"ranger-stop{i}" for i in range(8)],
    "Camping": [f"camping{i}" for i in range(9)],
    "Unknown": ["ranger-base"],
}
SENSORS = {gate: category for category, gates in SENSOR_TYPES.items() for gate in gates}


def normalize(row):
    if len(row) != len(SOURCE_COLUMNS):
        raise ValueError("invalid_field_count")
    timestamp, car_id, car_type, gate_name = (value.strip() for value in row)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        raise ValueError("invalid_timestamp") from None
    if parsed.tzinfo is not None or timestamp != parsed.isoformat(sep=" ", timespec="seconds"):
        raise ValueError("invalid_timestamp")
    if not car_id:
        raise ValueError("empty_car_id")
    car_type = car_type.upper()
    if car_type not in CAR_TYPES:
        raise ValueError("unknown_car_type")
    gate_name = gate_name.lower()
    if gate_name not in SENSORS:
        raise ValueError("unknown_gate")
    return parsed, car_id, car_type, gate_name


def read_csv(payload):
    with TextIOWrapper(BytesIO(payload), encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != SOURCE_COLUMNS:
            raise ValueError(f"expected CSV header {','.join(SOURCE_COLUMNS)}")
        while True:
            source_line = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            yield source_line, row


def build_silver(raw_events):
    accepted = []
    rejected = []
    seen = set()
    for event_id, row in raw_events:
        try:
            event = normalize(row)
        except ValueError as error:
            reason = str(error)
        else:
            if event not in seen:
                seen.add(event)
                accepted.append((*event, event_id))
                continue
            reason = "duplicate_event"
        rejected.append({"source_line": event_id, "reason": reason, "raw_fields": row})

    return list(enrich_events(accepted)), rejected


def enrich_events(accepted):
    accepted.sort(key=lambda event: (event[1], event[0], event[4]))
    for car_id, group in groupby(accepted, key=lambda event: event[1]):
        route = list(group)
        for index, (timestamp, _, car_type, gate, source_line) in enumerate(route):
            previous = route[index - 1] if index else None
            following = route[index + 1] if index + 1 < len(route) else None
            yield {
                "source_line": source_line,
                "timestamp": timestamp,
                "car_id": car_id,
                "car_type": car_type,
                "gate_name": gate,
                "sensor_type": SENSORS[gate],
                "date": timestamp.date(),
                "hour": timestamp.hour,
                "dow": timestamp.weekday(),
                "month": timestamp.month,
                "visit_order": index + 1,
                "prev_gate": previous[3] if previous else None,
                "next_gate": following[3] if following else None,
                "elapsed_seconds": int((timestamp - previous[0]).total_seconds()) if previous else None,
            }


def summarize_trip(car_id, route):
    first, last = route[0], route[-1]
    start, end = first["timestamp"], last["timestamp"]
    car_type = first["car_type"]
    gates = [event["gate_name"] for event in route]
    sensors = {event["sensor_type"] for event in route}
    camping = "Camping" in sensors
    ranger = car_type == "2P"
    hours = (end - start).total_seconds() / 3600
    return {
        "car_id": car_id,
        "car_type": car_type,
        "first_gate": gates[0],
        "last_gate": gates[-1],
        "first_time": start,
        "last_time": end,
        "n_events": len(route),
        "n_unique_gates": len(set(gates)),
        "n_unique_sensors": len(sensors),
        "visited_camping": camping,
        "route": " ".join(gates),
        "visited_ranger": "Ranger-Stop" in sensors,
        "visited_gate_restr": "Gate (restricted)" in sensors,
        "entry_hour": start.hour,
        "entry_dow": start.weekday(),
        "exit_hour": end.hour,
        "exit_dow": end.weekday(),
        "duration_h": hours,
        "is_weekend": start.weekday() >= 5,
        "is_thrutraffic": not camping and not ranger and hours < 2,
        "is_ranger": ranger,
        "is_overnight": camping and not ranger and hours > 12,
    }


def build_gold(events):
    trips = []
    event_counts = Counter()
    car_counts = Counter()
    for car_id, group in groupby(events, key=lambda event: event["car_id"]):
        route = list(group)
        trips.append(summarize_trip(car_id, route))
        buckets = Counter(
            (event["date"], event["hour"], event["gate_name"], event["car_type"]) for event in route
        )
        event_counts.update(buckets)
        car_counts.update(buckets.keys())

    traffic = [
        {
            "date": date,
            "hour": hour,
            "gate_name": gate,
            "car_type": car_type,
            "event_count": count,
            "car_id_count": car_counts[date, hour, gate, car_type],
        }
        for (date, hour, gate, car_type), count in sorted(event_counts.items())
    ]
    return trips, traffic

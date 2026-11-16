"""Thread-safe JSON and TSV HTTP interface to the V3 model."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
from threading import RLock

import pandas as pd

from .features import RouteFeatureExtractor
from .model import RouteAnomalyModel

COLUMNS = ['car-id', 'car-type', 'Timestamp', 'gate-name']
MAX_REQUEST_BYTES = 50_000_000


def route_events(routes):
    if not isinstance(routes, list) or not routes:
        raise ValueError('routes must be a nonempty list')
    rows, ids = [], set()
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError('route must be an object')
        car, kind, events = route.get('car_id'), route.get('car_type'), route.get('events')
        if not isinstance(car, str) or not car.strip() or not isinstance(kind, str) or not kind.strip():
            raise ValueError('car_id and car_type must be nonempty strings')
        if car in ids:
            raise ValueError('Duplicate car_id')
        ids.add(car)
        if not isinstance(events, list) or not events:
            raise ValueError('events must be a nonempty list')
        for event in events:
            if not isinstance(event, dict) or not all(isinstance(event.get(k), str) and event[k].strip() for k in ('timestamp', 'gate_name')):
                raise ValueError('timestamp and gate_name must be nonempty strings')
            rows.append((car, kind, event['timestamp'], event['gate_name']))
    return pd.DataFrame(rows, columns=COLUMNS)


class ModelService:
    def __init__(self, distances_path, artifact_path):
        self.distances_path = Path(distances_path) if distances_path else None
        self.artifact_path = Path(artifact_path)
        self._model = None
        self._lock = RLock()

    def _distances(self):
        distances = pd.read_csv(self.distances_path) if self.distances_path else RouteFeatureExtractor().distances
        if not {'sensor_from', 'sensor_to', 'distance_km'}.issubset(distances.columns):
            raise ValueError('Invalid distances CSV')
        return distances

    @staticmethod
    def _prepare(events, distances):
        events = events.copy()
        if list(events.columns) != COLUMNS or events.empty:
            raise ValueError('Expected nonempty events with the protocol columns')
        if events.isna().any().any() or events[COLUMNS].astype(str).apply(lambda x: x.str.strip().eq('')).any().any():
            raise ValueError('Empty event field')
        if events.groupby('car-id')['car-type'].nunique().gt(1).any():
            raise ValueError('Inconsistent car type for car-id')
        sensors = set(distances.sensor_from) | set(distances.sensor_to)
        if not set(events['gate-name']).issubset(sensors):
            raise ValueError('Unknown sensor')
        events['Timestamp'] = pd.to_datetime(events['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        return events.sort_values(['car-id', 'Timestamp'], kind='stable')

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.artifact_path.is_file():
            raise FileNotFoundError('Model is not trained')
        try:
            model = RouteAnomalyModel.load(self.artifact_path)
            sensor = next(iter(model.positions))
            probe = pd.DataFrame([['health-check', '__unknown__', pd.Timestamp('2015-01-01'), sensor]], columns=COLUMNS)
            _, scores = model.score(probe)
            if len(scores) != 1 or not 0 <= float(scores[0]) <= 1:
                raise ValueError('Invalid probe result')
        except Exception as exc:
            raise FileNotFoundError('Model could not be loaded or failed its readiness check') from exc
        self._model = model
        return model

    def ready(self):
        with self._lock:
            try:
                self._load()
                return True
            except FileNotFoundError:
                return False

    def train(self, events):
        with self._lock:
            distances = self._distances()
            events = self._prepare(events, distances)
            model = RouteAnomalyModel().fit(events, distances)
            model.save(self.artifact_path)
            self._model = model
            return events['car-id'].nunique()

    def score(self, events):
        with self._lock:
            model = self._load()
            events = self._prepare(events, model.distances)
            ids, scores = model.score(events)
            return [{'car_id': str(car), 'score': float(score)} for car, score in zip(ids, scores)]


class AnomalyServer(ThreadingHTTPServer):
    def __init__(self, address, service):
        super().__init__(address, Handler)
        self.service = service


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, content, mime='application/json'):
        data = content.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', mime + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, status, data):
        self.reply(status, json.dumps(data, ensure_ascii=False, allow_nan=False))

    def do_GET(self):
        if self.path == '/v1/live':
            return self.json_reply(200, {'status': 'ok'})
        if self.path != '/v1/health':
            return self.json_reply(404, {'detail': 'Not found'})
        self.json_reply(200, {'protocol': 'anomaly-route-tsv/1', 'provider': 'Isolation Forest + SOM v3', 'ready': self.server.service.ready()})

    def do_POST(self):
        try:
            if self.path not in ('/train', '/score', '/score/batch', '/v1/score'):
                return self.json_reply(404, {'detail': 'Not found'})
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError('Invalid request size')
            body = self.rfile.read(length).decode('utf-8')
            service = self.server.service
            if self.path == '/v1/score':
                events = pd.read_csv(StringIO(body), sep='\t', dtype=str, keep_default_na=False)
                scores = service.score(events)
                text = 'car-id\tscore\n' + ''.join(f"{row['car_id']}\t{row['score']:.10f}\n" for row in scores)
                return self.reply(200, text, 'text/plain')
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError('Expected JSON object')
            if self.path == '/score':
                return self.json_reply(200, service.score(route_events([payload]))[0])
            events = route_events(payload.get('routes'))
            if self.path == '/train':
                return self.json_reply(200, {'routes': service.train(events), 'saved': True})
            return self.json_reply(200, {'scores': service.score(events)})
        except FileNotFoundError as exc:
            self.json_reply(503, {'detail': str(exc)})
        except (ValueError, KeyError, AssertionError, UnicodeError, pd.errors.ParserError) as exc:
            self.json_reply(400, {'detail': str(exc)})
        except Exception:
            self.json_reply(500, {'detail': 'Internal server error'})


def main():
    parser = argparse.ArgumentParser(description='VAST V3 anomaly server')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--model', default=os.getenv('VAST_MODEL_PATH', 'model_v3.joblib'))
    parser.add_argument('--distances', default=os.getenv('VAST_DISTANCES_PATH'))
    args = parser.parse_args()
    with AnomalyServer((args.host, args.port), ModelService(args.distances, args.model)) as server:
        print(f'Listening on http://{args.host}:{server.server_port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()

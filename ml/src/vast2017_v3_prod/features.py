"""Fixed-length route representation compatible with the original V3 model.

Static map data is bundled with the package; no repository or GUI is required.
"""
from importlib.resources import files
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import csr_matrix

Representation = tuple[pd.DataFrame, csr_matrix, pd.Series, pd.Series]

def _segment_statistics(events: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consecutive sensor transitions per car."""
    g = events.groupby('car-id', sort=False)
    seg = events[['car-id', 'gate-name', 'Timestamp']].copy()
    seg['sensor_to'] = g['gate-name'].shift(-1)
    seg['time_to'] = g['Timestamp'].shift(-1)
    seg = seg.dropna(subset=['sensor_to']).rename(columns={'gate-name': 'sensor_from', 'Timestamp': 'time_from'})
    seg['hours'] = (seg.time_to - seg.time_from).dt.total_seconds() / 3600
    seg = seg.merge(dist, on=['sensor_from', 'sensor_to'], how='left', validate='many_to_one')
    seg['same_sensor'] = seg.sensor_from.eq(seg.sensor_to)
    seg['camping_stay'] = seg.same_sensor & seg.sensor_from.str.match('^camping\\d+$') & seg.hours.gt(0)
    seg['valid'] = ~seg.same_sensor & seg.hours.gt(0) & seg.distance_km.notna()
    seg['speed'] = seg.distance_km / seg.hours.where(seg.valid)
    seg['invalid'] = ~seg.valid
    seg['missing_path'] = seg.distance_km.isna()
    seg['nonpositive'] = seg.hours.le(0)
    seg['camp_hours'] = seg.hours.where(seg.camping_stay, 0)
    agg = seg.groupby('car-id').agg(
        valid_segments=('valid', 'sum'),
        invalid_segments=('invalid', 'sum'),
        repeats=('same_sensor', 'sum'),
        missing_paths=('missing_path', 'sum'),
        nonpositive=('nonpositive', 'sum'),
        camping_hours=('camp_hours', 'sum'),
        speed_mean=('speed', 'mean'),
        speed_min=('speed', 'min'),
        speed_median=('speed', 'median'),
        speed_max=('speed', 'max'),
        speed_p95=('speed', lambda x: x.quantile(0.95)),
        distance_partial=('distance_km', lambda x: x.sum(min_count=1)),
    )
    return agg


def _route_features(events: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events['gate-name'] = events['gate-name'].str.strip().str.lower()
    events['car-type'] = events['car-type'].str.strip()
    events = events.sort_values(['car-id', 'Timestamp'], kind='stable').reset_index(drop=True)
    assert events[['car-id', 'Timestamp', 'gate-name', 'car-type']].notna().all().all()
    assert events.groupby('car-id')['car-type'].nunique().eq(1).all()
    g = events.groupby('car-id', sort=False)
    routes = g.agg(
        car_type=('car-type', 'first'),
        n_events=('Timestamp', 'size'),
        n_sensors=('gate-name', 'nunique'),
        first=('Timestamp', 'min'),
        last=('Timestamp', 'max'),
    )
    routes['duration_h'] = (routes['last'] - routes['first']).dt.total_seconds() / 3600
    visits = pd.crosstab(events['car-id'], events['gate-name']).gt(0).astype(int).add_prefix('visit_')
    routes = routes.join(visits)
    agg = _segment_statistics(events, dist)
    routes = routes.join(agg)
    counts = ['valid_segments', 'invalid_segments', 'repeats', 'missing_paths', 'nonpositive', 'camping_hours']
    routes[counts] = routes[counts].fillna(0)
    complete = routes.n_events.gt(1) & routes.missing_paths.eq(0) & routes.nonpositive.eq(0)
    routes['distance_km'] = routes.distance_partial.where(complete)
    routes['travel_hours'] = routes.duration_h - routes.camping_hours
    routes['route_speed'] = routes.distance_km / routes.travel_hours.where(routes.travel_hours.gt(0))
    routes['camping_visited'] = events.assign(camp=events['gate-name'].str.match('^camping\\d+$')).groupby('car-id').camp.any().astype(int)
    assert len(routes) == events['car-id'].nunique() and routes.index.is_unique
    assert routes.n_events.sum() == len(events)
    return routes.drop(columns=['first', 'last', 'distance_partial'])

def _representation(
    events: pd.DataFrame, distances: pd.DataFrame, positions: dict
) -> Representation:
    """Fixed-length route features, before learned transformations.

    A hashed bag of directed bigrams and trigrams captures local order; the
    12 interpolated positions encode geographic shape and chronology across
    the *whole* route, independent of route length. The time channel uses
    elapsed fraction at each event-progress position, not clock time.
    """
    events = events.sort_values(['car-id', 'Timestamp'], kind='stable').copy()
    base = _route_features(events, distances)
    base = base.drop(columns=['car_type'] + [c for c in base if c.startswith('visit_')])
    grouped = events.groupby('car-id', sort=False)
    groups = {k: g for k, g in grouped}
    sensors = sorted(positions)
    sensor_idx = {s: i for i, s in enumerate(sensors)}
    rows = []
    geometry = []
    types = []
    seasons = []
    sample_at = np.linspace(0, 1, 12)
    for route_id in base.index:
        g = groups[route_id]
        names = g['gate-name'].tolist()
        seq = [sensor_idx[s] for s in names]
        tokens = {}
        for n in (2, 3):
            for i in range(len(seq) - n + 1):
                token = '-'.join(map(str, seq[i:i + n]))
                tokens[token] = tokens.get(token, 0) + 1
        counts = {n: max(1, len(seq) - n + 1) for n in (2, 3)}
        rows.append({k: v / counts[k.count('-') + 1] for k, v in tokens.items()})
        xy = np.asarray([positions[s] for s in names], dtype=float) / 200
        progress = np.linspace(0, 1, len(xy)) if len(xy) > 1 else np.array([0.0])
        t = (g.Timestamp - g.Timestamp.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        time_fraction = t / t[-1] if t[-1] > 0 else np.zeros(len(t))
        x = np.interp(sample_at, progress, xy[:, 0])
        y = np.interp(sample_at, progress, xy[:, 1])
        ts = np.interp(sample_at, progress, time_fraction)
        geometry.append(np.r_[x, y, ts, np.diff(x), np.diff(y)])
        types.append(g['car-type'].iloc[0])
        month = g.Timestamp.iloc[0].month
        if month in (12, 1, 2):
            season = 'winter'
        elif month in (3, 4, 5):
            season = 'spring'
        elif month in (6, 7, 8):
            season = 'summer'
        else:
            season = 'autumn'
        seasons.append(season)
    geo = pd.DataFrame(geometry, index=base.index, columns=[f'geo_{i}' for i in range(58)])
    tokens = FeatureHasher(n_features=4096, input_type='dict', alternate_sign=False).transform(rows).tocsr()
    return (
        pd.concat([base, geo], axis=1),
        tokens,
        pd.Series(types, index=base.index),
        pd.Series(seasons, index=base.index),
    )

class RouteFeatureExtractor:
    """Build the 76 numeric and 4096 hashed route features used by V3."""

    def __init__(
        self, distances: pd.DataFrame | None = None, positions: dict | None = None
    ) -> None:
        data = files(__package__).joinpath('data')
        if distances is None:
            with data.joinpath('sensor_distances.csv').open('rb') as stream:
                distances = pd.read_csv(stream)
        if positions is None:
            positions = json.loads(data.joinpath('sensor_positions.json').read_text(encoding='utf-8'))
        self.distances = distances.copy()
        self.positions = dict(positions)

    def transform(self, events: pd.DataFrame) -> Representation:
        """Return numeric frame, sparse n-grams, vehicle types, and seasons."""
        return _representation(events, self.distances, self.positions)

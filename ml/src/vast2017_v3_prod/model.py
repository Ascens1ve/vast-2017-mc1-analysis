"""V3 route model: type-specific isolation forests and self-organizing maps."""
from pathlib import Path
import os
import tempfile

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .features import RouteFeatureExtractor

SEED = 42
SEASONS = ('winter', 'spring', 'summer', 'autumn')
MIN_CALIBRATION = 30


class SelfOrganizingMap:
    """Batch-trained 5x5 SOM, with the exact V3 training and distance rules."""

    @staticmethod
    def fit(x, seed=SEED, side=5, epochs=12):
        rng = np.random.default_rng(seed)
        nodes = side * side
        weights = x[rng.choice(len(x), size=nodes, replace=len(x) < nodes)].copy()
        coords = np.stack(np.unravel_index(np.arange(nodes), (side, side)), axis=1)
        grid = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        for epoch in range(epochs):
            radius = max(0.6, 2.5 * (0.6 / 2.5) ** (epoch / max(1, epochs - 1)))
            neighborhood = np.exp(-grid / (2 * radius ** 2))
            order = rng.permutation(len(x))
            for begin in range(0, len(x), 256):
                sample = x[order[begin:begin + 256]]
                distances = ((sample[:, None, :] - weights[None, :, :]) ** 2).sum(axis=2)
                influence = neighborhood[np.argmin(distances, axis=1)]
                target = (influence.T @ sample) / np.maximum(influence.sum(axis=0)[:, None], 1e-12)
                weights += .35 * (target - weights)
        return weights

    @staticmethod
    def distance(x, weights):
        result = []
        for begin in range(0, len(x), 512):
            distances = ((x[begin:begin + 512, None, :] - weights[None, :, :]) ** 2).mean(axis=2)
            result.extend(np.sqrt(distances.min(axis=1)))
        return np.asarray(result)


class RouteAnomalyModel:
    """Model and saved feature configuration; scores are empirical percentiles."""

    def __init__(self):
        self.artifact = None
        self.distances = None
        self.positions = None

    @staticmethod
    def _rank(reference, values):
        return np.searchsorted(reference, values, side='right') / len(reference)

    @staticmethod
    def _matrix(frame, tokens, artifact, fit=False):
        frame = frame.reindex(columns=artifact['columns'], fill_value=0)
        if fit:
            numeric = artifact['imputer'].fit_transform(frame)
            embedding = artifact['svd'].fit_transform(tokens)
        else:
            numeric = artifact['imputer'].transform(frame)
            embedding = artifact['svd'].transform(tokens)
        return np.column_stack([numeric, embedding])

    @staticmethod
    def _train_group(x):
        forest = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(x)
        scaler = RobustScaler().fit(x)
        weights = SelfOrganizingMap.fit(np.clip(scaler.transform(x), -8, 8))
        return {'forest': forest, 'scaler': scaler, 'weights': weights}

    @staticmethod
    def _raw(x, group):
        normalized = np.clip(group['scaler'].transform(x), -8, 8)
        return np.column_stack([-group['forest'].score_samples(x),
                                SelfOrganizingMap.distance(normalized, group['weights'])])

    def fit(self, events, distances=None):
        extractor = RouteFeatureExtractor(distances=distances)
        frame, tokens, types, seasons = extractor.transform(events)
        self.distances = extractor.distances.copy()
        self.positions = extractor.positions.copy()
        fit_idx, cal_idx = train_test_split(np.arange(len(frame)), test_size=.2,
                                            random_state=SEED, stratify=types)
        calibration_counts = types.iloc[cal_idx].value_counts().reindex(types.unique(), fill_value=0)
        insufficient = calibration_counts[calibration_counts < MIN_CALIBRATION]
        if not insufficient.empty:
            counts = ', '.join(f'{name}: {count}' for name, count in insufficient.items())
            raise ValueError(
                f'At least {MIN_CALIBRATION} calibration routes are required per vehicle type; '
                f'insufficient calibration counts: {counts}'
            )
        artifact = {'version': 'v3', 'columns': list(frame.columns),
                    'imputer': SimpleImputer(strategy='median', keep_empty_features=True),
                    'svd': TruncatedSVD(n_components=24, random_state=SEED),
                    'groups': {}, 'calibration': {}, 'counts': {}}
        artifact['split_ids'] = {'fit': frame.index[fit_idx].tolist(),
                                 'calibration': frame.index[cal_idx].tolist()}
        x_fit = self._matrix(frame.iloc[fit_idx], tokens[fit_idx], artifact, fit=True)
        x_cal = self._matrix(frame.iloc[cal_idx], tokens[cal_idx], artifact)
        fit_types = types.iloc[fit_idx].to_numpy()
        cal_types = types.iloc[cal_idx].to_numpy()
        cal_seasons = seasons.iloc[cal_idx].to_numpy()
        for name in ['__global__'] + sorted(types.unique().tolist()):
            fm = np.ones(len(fit_idx), dtype=bool) if name == '__global__' else fit_types == name
            cm = np.ones(len(cal_idx), dtype=bool) if name == '__global__' else cal_types == name
            group = self._train_group(x_fit[fm])
            artifact['groups'][name] = group
            raw = self._raw(x_cal[cm], group)
            artifact['counts'][name] = {'fit': int(fm.sum()), 'calibration': int(cm.sum())}
            for season in ('__all__',) + SEASONS:
                mask = np.ones(len(raw), dtype=bool) if season == '__all__' else cal_seasons[cm] == season
                if mask.sum() < MIN_CALIBRATION:
                    continue
                arr = raw[mask]
                first, second = np.sort(arr[:, 0]), np.sort(arr[:, 1])
                combined = (self._rank(first, arr[:, 0]) + self._rank(second, arr[:, 1])) / 2
                artifact['calibration'][(name, season)] = {
                    'iforest': first, 'som': second, 'ensemble': np.sort(combined)}
        self.artifact = artifact
        return self

    def score(self, events, distances=None, return_details=False, force_global=False):
        if self.artifact is None:
            raise ValueError('Model has not been trained or loaded')
        extractor = RouteFeatureExtractor(distances=self.distances if distances is None else distances,
                                          positions=self.positions)
        frame, tokens, types, seasons = extractor.transform(events)
        artifact = self.artifact
        x = self._matrix(frame, tokens, artifact)
        out = np.empty((len(frame), 3))
        fallbacks = []
        type_values, season_values = types.to_numpy(), seasons.to_numpy()
        for kind in np.unique(type_values):
            name = kind if not force_global and kind in artifact['groups'] else '__global__'
            for season in np.unique(season_values[type_values == kind]):
                mask = (type_values == kind) & (season_values == season)
                key = (name, season) if (name, season) in artifact['calibration'] else (name, '__all__')
                ref = artifact['calibration'][key]
                raw = self._raw(x[mask], artifact['groups'][name])
                first = self._rank(ref['iforest'], raw[:, 0])
                second = self._rank(ref['som'], raw[:, 1])
                out[mask] = np.column_stack([first, second,
                                             self._rank(ref['ensemble'], (first + second) / 2)])
                fallbacks.append((kind, season, key, int(mask.sum())))
        return (frame.index, out, fallbacks) if return_details else (frame.index, out[:, 2])

    def save(self, path):
        """Atomically replace one complete joblib file, including distances."""
        if self.artifact is None:
            raise ValueError('Model has not been trained')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            joblib.dump({'artifact': self.artifact, 'distances': self.distances,
                         'positions': self.positions}, temp)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    @classmethod
    def load(cls, path):
        """Load only trusted model files: joblib deserialization executes code."""
        saved = joblib.load(path)
        model = cls()
        model.artifact = saved['artifact']
        model.distances = saved['distances']
        model.positions = saved['positions']
        if model.artifact.get('version') != 'v3':
            raise ValueError('Unsupported model version')
        return model

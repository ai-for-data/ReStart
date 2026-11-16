"""Vector execution adapters around the official scalar TabICL estimator.

The explicit shared adapter reuses weights and native query preprocessing while
retaining one original cache per output. Explicit shared-batched execution
groups channels through the original scalar cached-batch entry.
"""
from pathlib import Path

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from tabicl import TabICLRegressor


class TabICLNoisePredictor:
    def __init__(self, *, checkpoint: str | Path, device: str, kv_cache=False,
                 execution='native'):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if kv_cache not in (False, True, 'repr') or execution not in ('native', 'shared', 'shared-batched', 'pr87'):
            raise ValueError('Declare native/shared execution and an original cache mode explicitly.')
        if execution.startswith('shared') and kv_cache is False:
            raise ValueError('The validated shared adapter requires independent native per-output caches.')
        self.execution, self.kv_cache = execution, kv_cache
        parameters = dict(model_path=self.checkpoint, device=device, kv_cache=kv_cache)
        # Preserve the native model, precision, offload and all eight ensembles.
        # The batched adapter explicitly uses public batch_size=None under the
        # user's maximum-batching instruction; other execution cohorts retain
        # their original parameters. Never silently substitute a cache mode.
        if execution == 'pr87':
            if kv_cache is not False:
                raise ValueError('Original PR87 multi-target execution requires cache=False.')
            self.estimator = TabICLRegressor(**parameters, batch_size=None)
        elif execution.startswith('shared'):
            from embodpfn.tabicl_shared import (SharedModelCachedMultiOutputRegressor,
                                               BatchedSharedModelCachedMultiOutputRegressor)
            adapter = (BatchedSharedModelCachedMultiOutputRegressor if execution == 'shared-batched'
                       else SharedModelCachedMultiOutputRegressor)
            self.estimator = adapter(**parameters)
        else:
            self.estimator = MultiOutputRegressor(TabICLRegressor(**parameters))
        self.context_version = 0
        self.has_context = False

    def update_context(self, conditions: np.ndarray, inverted_noise: np.ndarray):
        conditions = np.asarray(conditions, dtype=np.float32)
        inverted_noise = np.asarray(inverted_noise, dtype=np.float32)
        if conditions.ndim != 2 or inverted_noise.ndim < 2:
            raise ValueError('Expected conditions [N,Dc] and inverted_noise [N,*noise_shape].')
        if len(conditions) != len(inverted_noise) or len(conditions) < 2:
            raise ValueError('Context needs matching input/target rows and at least two examples.')
        if not np.isfinite(conditions).all() or not np.isfinite(inverted_noise).all():
            raise ValueError('Condition features and inverted noise must be finite.')
        self.has_context = False
        self.noise_shape = inverted_noise.shape[1:]
        self.condition_dim = conditions.shape[1]
        self.support_conditions = conditions.copy()
        self.support_noise = inverted_noise.reshape(len(inverted_noise), -1).copy()
        self.estimator.fit(self.support_conditions, self.support_noise)
        self.context_version += 1
        self.has_context = True
        return self

    def predict_initial_noise(self, conditions: np.ndarray) -> np.ndarray:
        if not self.has_context:
            raise RuntimeError('Provide an explicit support context before prediction.')
        conditions = np.asarray(conditions, dtype=np.float32)
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dim:
            raise ValueError('Query conditions must match the support feature interface.')
        if not np.isfinite(conditions).all():
            raise ValueError('Query conditions must be finite.')
        # Each native estimator uses all support rows and all eight default
        # ensemble members. No coordinate subsampling or ensemble reduction.
        prediction = self.estimator.predict(conditions)
        if not np.isfinite(prediction).all():
            raise RuntimeError('The official regressor produced non-finite noise.')
        return np.asarray(prediction, dtype=np.float32).reshape(len(conditions), *self.noise_shape)

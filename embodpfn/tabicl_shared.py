"""Explicit shared-weight execution adapter around native TabICL 2.2.0.

This is our adapter, not an official released multi-output estimator. Native
scalar fit/predict, label scaling, ensemble generation, cache construction,
cache slicing and neural forwards remain delegated to TabICL. Each output has
its own complete native cache. Only weights and target-independent query
preprocessing are shared. Native per-output fit setup is retained.
"""
from collections import OrderedDict
from threading import Lock

import numpy as np
import torch
from tabicl import TabICLRegressor


class _ModelBundle:
    model = None


class _SharedWeightRegressor(TabICLRegressor):
    """Override only loading; never implement a network or cache operation."""

    def _load_model(self):
        bundle = self._embodpfn_model_bundle
        if bundle.model is None:
            super()._load_model()
            bundle.model = self.model_
            bundle.path = self.model_path_
            bundle.config = self.model_config_
        else:
            self.model_ = bundle.model
            self.model_path_ = bundle.path
            self.model_config_ = bundle.config


class _QueryPreprocessing:
    def __init__(self, encoder, ensemble):
        self.encoder, self.ensemble = encoder, ensemble
        self.active = False

    def begin(self, query):
        if self.active:
            raise RuntimeError('Shared native model queries must be serialized.')
        self.active = True
        self.query = query.copy()
        self.encoded = self.views = None
        self.encoder_calls = self.ensemble_calls = 0

    def end(self):
        self.active = False
        self.query = self.encoded = self.views = None

    def encode(self, query):
        if not self.active or not np.array_equal(query, self.query):
            raise RuntimeError('Every output must receive the same complete finite query.')
        if self.encoded is None:
            self.encoded = self.encoder.transform(query)
            if not isinstance(self.encoded, np.ndarray):
                raise ValueError('This embedding adapter requires native numerical ndarray preprocessing.')
            self.encoder_calls += 1
        return self.encoded.copy()

    def transform_test(self, encoded):
        if not self.active or not np.array_equal(encoded, self.encoded):
            raise RuntimeError('Target-independent preprocessing inputs diverged.')
        if self.views is None:
            self.views = self.ensemble.transform(encoded, mode='test')
            self.ensemble_calls += 1
        # Preserve the original estimator's ownership of its forward inputs.
        # No y, representation or neural KV cache is shared here.
        return OrderedDict((name, (values[0].copy(),)) for name, values in self.views.items())


class _EncoderProxy:
    def __init__(self, native, shared):
        self.native, self.shared = native, shared

    def transform(self, query):
        return self.shared.encode(query)

    def __getattr__(self, name):
        return getattr(self.native, name)


class _EnsembleProxy:
    def __init__(self, native, shared):
        self.native, self.shared = native, shared

    def transform(self, X=None, mode='both'):
        if mode == 'test':
            return self.shared.transform_test(X)
        # Label-bearing modes always delegate to this channel's own generator.
        return self.native.transform(X, mode=mode)

    def __getattr__(self, name):
        return getattr(self.native, name)


class SharedModelCachedMultiOutputRegressor:
    def __init__(self, **native_parameters):
        if native_parameters.get('kv_cache') not in (True, 'kv', 'repr'):
            raise ValueError('This adapter requires a complete native per-channel cache.')
        self.parameters = dict(native_parameters)
        self.lock = Lock()
        self.estimators_ = []

    def fit(self, X, Y):
        X, Y = np.asarray(X), np.asarray(Y)
        if (X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y)
                or not np.isfinite(X).all() or not np.isfinite(Y).all()):
            raise ValueError('Provide all finite numerical features and every output coordinate.')
        with self.lock:
            # Context updates rebuild all channel caches through original fit.
            self.fitted_ = False
            self.estimators_ = []
            bundle = _ModelBundle()
            reference_encoded = reference_views = None
            for index in range(Y.shape[1]):
                estimator = _SharedWeightRegressor(**self.parameters)
                if estimator.get_params()['n_estimators'] != 8:
                    raise ValueError('Keep the eight official default ensembles.')
                estimator._embodpfn_model_bundle = bundle
                estimator.fit(X, Y[:, index])
                # Check the complete native train views before sharing any X
                # preprocessing. The y arrays/caches remain channel-specific.
                encoded = estimator.X_encoder_.transform(X)
                views = estimator.ensemble_generator_.transform(X=None, mode='train')
                if reference_encoded is None:
                    reference_encoded = encoded
                    reference_views = OrderedDict((name, values[0].copy())
                                                  for name, values in views.items())
                elif (not np.array_equal(encoded, reference_encoded)
                        or list(views) != list(reference_views)
                        or any(not np.array_equal(views[name][0], reference_views[name])
                               for name in views)):
                    raise RuntimeError('Native feature preprocessing differs between output channels.')
                if not estimator.model_kv_cache_:
                    raise RuntimeError('The requested native per-channel cache was not built.')
                self.estimators_.append(estimator)
            if len({id(est.model_) for est in self.estimators_}) != 1:
                raise RuntimeError('The adapter did not retain exactly one model.')
            if len({id(est.model_kv_cache_) for est in self.estimators_}) != len(self.estimators_):
                raise RuntimeError('Never share a label-conditioned cache between output channels.')
            first = self.estimators_[0]
            self.query_preprocessing = _QueryPreprocessing(first.X_encoder_, first.ensemble_generator_)
            for estimator in self.estimators_:
                estimator.X_encoder_ = _EncoderProxy(estimator.X_encoder_, self.query_preprocessing)
                estimator.ensemble_generator_ = _EnsembleProxy(
                    estimator.ensemble_generator_, self.query_preprocessing)
            self.n_features_in_ = X.shape[1]
            self.n_outputs_ = Y.shape[1]
            self.resident_model_count_ = 1
            self.native_cache_count_ = len(self.estimators_)
            self.fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        if (not getattr(self, 'fitted_', False) or X.ndim != 2 or X.shape[1] != self.n_features_in_
                or not np.isfinite(X).all()):
            raise ValueError('Use the same complete finite feature interface as fit.')
        with self.lock:
            self.query_preprocessing.begin(X)
            try:
                # Native predict passes this channel's own cache explicitly to
                # the common model. Do not run concurrent forwards against its
                # mutable native cache pointer.
                prediction = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
                self.last_query_preprocessing_ = {
                    'native_encoder_calls': self.query_preprocessing.encoder_calls,
                    'native_ensemble_transform_calls': self.query_preprocessing.ensemble_calls}
                return prediction
            finally:
                self.query_preprocessing.end()


class BatchedSharedModelCachedMultiOutputRegressor(SharedModelCachedMultiOutputRegressor):
    """Batch independent y channels through the native public cache API.

    Original per-channel caches stay separate. TabICLCache.concat creates only
    a temporary native batch. The original scalar estimator's cached-batch
    entry preserves its batch_size split and native memory management. The final inverse
    StandardScaler and ensemble mean follow the original scalar regressor.
    """

    def __init__(self, *, target_batch_size=None, **native_parameters):
        # Explicit user-authorized execution setting: remove the scalar
        # estimator's outer batch8 cap. Its original InferenceManager still
        # selects/splits neural batches and handles its own OOM recovery.
        native_parameters.setdefault('batch_size', None)
        super().__init__(**native_parameters)
        if target_batch_size is not None and (isinstance(target_batch_size, bool)
                or not isinstance(target_batch_size, int) or target_batch_size < 1):
            raise ValueError('Target batch size must be a positive integer or None for all outputs.')
        self.target_batch_size = target_batch_size
        self._target_limits = {}

    def fit(self, X, Y):
        result = super().fit(X, Y)
        self._target_limits = {}
        return result

    @staticmethod
    def _native_group(first, Xs, group, normalization):
        # Keep temporary allocations in this frame so an OOM unwinds and
        # releases them before the execution scheduler retries a smaller group.
        try:
            native_caches = [estimator.model_kv_cache_[normalization] for estimator in group]
            cache = type(native_caches[0]).concat(native_caches)
            features = np.concatenate([Xs] * len(group), axis=0)
            return first._batch_forward_with_cache(features, cache, output_type='mean')
        finally:
            first.model_.clear_cache()

    def predict(self, X):
        X = np.asarray(X)
        if (not getattr(self, 'fitted_', False) or X.ndim != 2
                or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all()):
            raise ValueError('Use the same complete finite feature interface as fit.')
        with self.lock:
            self.query_preprocessing.begin(X)
            first = self.estimators_[0]
            model = first.model_
            calls, native_batches, oom_retries, actual_groups = 0, [], [], []
            try:
                encoded = self.query_preprocessing.encode(X)
                views = self.query_preprocessing.transform_test(encoded)
                pieces = [[] for _ in self.estimators_]
                for normalization, (Xs,) in views.items():
                    ensembles = len(Xs)
                    requested = min(self.target_batch_size or self.n_outputs_, self.n_outputs_)
                    limit = min(requested, self._target_limits.get(normalization, requested))
                    start = 0
                    while start < self.n_outputs_:
                        group = self.estimators_[start:start + limit]
                        try:
                            # All cache operations and inference remain native.
                            # Cache concatenation occurs before InferenceManager;
                            # only an actual accelerator OOM reduces this outer
                            # target group. This is our allocation scheduler,
                            # not an alternative model/cache implementation.
                            values = self._native_group(first, Xs, group, normalization)
                        except torch.OutOfMemoryError as error:
                            if len(group) == 1:
                                raise
                            oom_retries.append({'normalization': normalization, 'target_start': start,
                                'failed_targets': len(group), 'next_targets': max(1, len(group) // 2),
                                'error': str(error)})
                            values = None
                        if values is None:
                            # The exception frame has now been released. No
                            # source cache is moved, dropped or rewritten.
                            if first.device_.type == 'cuda':
                                with torch.cuda.device(first.device_):
                                    torch.cuda.empty_cache()
                            limit = max(1, len(group) // 2)
                            self._target_limits[normalization] = limit
                            continue
                        if values.shape != (len(group) * ensembles, len(X)):
                            raise RuntimeError('Native batched cache output shape changed.')
                        values = values.reshape(len(group), ensembles, len(X))
                        for offset in range(len(group)):
                            pieces[start + offset].append(values[offset].copy())
                        calls += 1
                        native_batches.append(len(group) * ensembles)
                        actual_groups.append({'normalization': normalization,
                                              'target_start': start, 'targets': len(group)})
                        start += len(group)
                        del values
                prediction = []
                for estimator, chunks in zip(self.estimators_, pieces):
                    values = np.concatenate(chunks, axis=0)
                    # Same inverse scaling and ensemble aggregation as native
                    # TabICLRegressor.predict; all eight ensembles are retained.
                    shape = values.shape
                    values = estimator.y_scaler_.inverse_transform(
                        values.reshape(-1, 1)).reshape(shape)
                    prediction.append(np.mean(values, axis=0))
                self.last_query_preprocessing_ = {
                    'native_encoder_calls': self.query_preprocessing.encoder_calls,
                    'native_ensemble_transform_calls': self.query_preprocessing.ensemble_calls,
                    'native_cached_batch_groups': calls,
                    'native_requested_independent_dataset_batches': native_batches,
                    'native_estimator_max_batch_size': first.batch_size,
                    'target_batch_size': self.target_batch_size,
                    'actual_target_groups': actual_groups,
                    'target_group_oom_retries': oom_retries}
                return np.column_stack(prediction)
            finally:
                model.clear_cache()
                self.query_preprocessing.end()

"""Support-buffer updates and prediction; no policy/PFN gradient updates."""
import json
from pathlib import Path

import numpy as np

from embodpfn.interfaces import InvertedCorrection, PolicyCondition
from embodpfn.noise_basis import CenteredSupportNoiseBasis


class ContextInferenceFailure(RuntimeError):
    """An integration failure must not be reported as robot task performance."""


class ContextAdapter:
    def __init__(self, backend, predictor, *, context_rule=None, support_context=None):
        self.backend = backend
        self.predictor = predictor
        self.context_rule = context_rule
        self.support_context = support_context
        self.corrections = []
        self._fitted_version = -1
        self.target_codec = backend.target_codec
        if self.target_codec not in {'direct-noise.v1', 'joint-noise-centered-support-basis.v1'}:
            raise ValueError('Unrecognized explicit noise-target codec.')
        self.noise_basis = None

    def add_demonstration(self, official_prepared_inputs, official_model_space_actions, *,
                          expert_action_is_pad=None):
        condition = self.backend.capture(official_prepared_inputs)
        if expert_action_is_pad is None:
            correction = self.backend.invert_expert_action(condition, official_model_space_actions)
        else:
            # Offline original-reader metadata only. It never enters capture,
            # official policy kwargs, readout features or the PFN condition.
            correction = self.backend.invert_expert_action(condition, official_model_space_actions,
                                                          expert_action_is_pad=expert_action_is_pad)
        if self.corrections:
            previous = self.corrections[0]
            if (condition.readout_id != previous.condition.readout_id or
                    condition.policy_checkpoint_sha256 != previous.condition.policy_checkpoint_sha256 or
                    correction.initial_noise.shape[1:] != previous.initial_noise.shape[1:] or
                    correction.inversion_recipe_id != previous.inversion_recipe_id):
                raise ValueError('Support rows must share checkpoint, readout, noise contract and inverse recipe.')
        # No reward filtering, row truncation, selection or augmentation.
        # Persist every full inverse label. Joint-noise basis coordinates are
        # derived later from the complete context; they do not replace labels.
        # Retaining every example's full VLM KV/video state would keep all
        # temporary policy activations alive on the GPU.
        stored_condition = PolicyCondition(None, condition.features.copy(), condition.readout_id,
                                           condition.policy_checkpoint_sha256)
        stored = InvertedCorrection(stored_condition, correction.initial_noise.copy(),
                                     correction.reconstruction_error, correction.inversion_recipe_id,
                                     dict(correction.reconstruction_metrics),
                                     None if correction.native_inverse_initial_point is None
                                     else correction.native_inverse_initial_point.copy())
        self.corrections.append(stored)
        return stored

    def prepare_context(self):
        if not self.corrections:
            raise RuntimeError('An adapted run requires expert context; run the unchanged official entry for a baseline.')
        if self._fitted_version != len(self.corrections):
            features = np.concatenate([c.condition.features for c in self.corrections])
            noise = np.concatenate([c.initial_noise for c in self.corrections])
            if self.context_rule is not None:
                # Apply the author's buffer rule to the recorded complete
                # native reconstruction. Raw labels remain in the artifact.
                groups = [group for correction in self.corrections for group in
                          correction.reconstruction_metrics.get(
                              'artifact_reconstruction_metrics', [correction.reconstruction_metrics])]
                mse = np.asarray([value for group in groups
                                  for value in group['action_mse_per_chunk']])
                if len(mse) != len(features) or not np.isfinite(mse).all():
                    raise ValueError('Every context row requires its actual native chunk reconstruction MSE.')
                keep = mse <= self.context_rule['threshold']
                print(json.dumps({'official_context_admission': self.context_rule,
                    'raw_rows': len(features), 'kept_rows': int(keep.sum()),
                    'excluded_row_indices': np.flatnonzero(~keep).tolist()}), flush=True)
                if keep.sum() < 2:
                    raise RuntimeError('Fewer than two labels satisfy the author rule; no alternate solver or baseline fallback.')
                features, noise = features[keep], noise[keep]
            if self.target_codec == 'joint-noise-centered-support-basis.v1':
                self.noise_basis = CenteredSupportNoiseBasis()
                targets = self.noise_basis.fit_transform(noise)
            else:
                targets = noise
            self.predictor.update_context(features, targets)
            self._fitted_version = len(self.corrections)

    def predict(self, official_prepared_inputs):
        try:
            self.prepare_context()
            # Each backend declares its feature-readout and noise boundary.
            # RoboTwin reads the original encoder before its unchanged
            # sampler so epsilon is ready at the native random factories.
            def predict_noise(features):
                predicted = self.predictor.predict_initial_noise(features)
                if self.noise_basis is not None:
                    return self.noise_basis.inverse_transform(predicted)
                return predicted
            return self.backend.act_conditioned(official_prepared_inputs, predict_noise)
        except Exception as error:
            raise ContextInferenceFailure('Context-conditioned policy inference failed.') from error

    def save_support(self, path):
        """Export finite inverse labels and their actual reconstruction diagnostics."""
        if not self.corrections:
            raise ValueError('No expert inverse labels.')
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('Use an explicit .npz support filename.')
        if path.exists():
            raise FileExistsError(path)
        first = self.corrections[0]
        manifest = {
            'checkpoint_sha256': first.condition.policy_checkpoint_sha256,
            'readout_id': first.condition.readout_id,
            'inversion_recipe_id': first.inversion_recipe_id,
            'target_codec': self.target_codec,
            'support_rows': sum(len(c.initial_noise) for c in self.corrections),
            'reconstruction_max_errors': [c.reconstruction_error for c in self.corrections],
            'reconstruction_metrics': [c.reconstruction_metrics for c in self.corrections],
            'selection': 'all finite raw inverse labels are preserved; PFN admission is declared separately',
            'pfn_context_admission': self.context_rule,
        }
        if self.support_context is not None:
            manifest['context_coverage'] = self.support_context
        values = dict(conditions=np.concatenate([c.condition.features for c in self.corrections]),
                      noise=np.concatenate([c.initial_noise for c in self.corrections]),
                      manifest=np.asarray(json.dumps(manifest)))
        points = [c.native_inverse_initial_point for c in self.corrections]
        if any(point is not None for point in points):
            if any(point is None for point in points):
                raise ValueError('Do not mix initial-point and epsilon artifact contracts.')
            values['native_inverse_initial_points'] = np.concatenate(points)
            manifest['noise_parameterization'] = 'epsilon; native history center and sigma retained'
            values['manifest'] = np.asarray(json.dumps(manifest))
        np.savez(path, **values)

    def load_support(self, path):
        """Restore an explicit support artifact, including approximation diagnostics."""
        if self.corrections:
            raise RuntimeError('Load into an empty context; existing rows are never overwritten implicitly.')
        with np.load(path, allow_pickle=False) as artifact:
            manifest = json.loads(str(artifact['manifest'].item()))
            features = artifact['conditions'].copy()
            noise = artifact['noise'].copy()
            raw_points = (artifact['native_inverse_initial_points'].copy()
                          if 'native_inverse_initial_points' in artifact.files else None)
        if (manifest['checkpoint_sha256'] != self.backend.asset['weight_sha256'] or
                manifest['readout_id'] != self.backend.readout_id or
                manifest['target_codec'] != self.target_codec):
            raise ValueError('Support artifact is for another checkpoint/readout.')
        if (self.support_context is not None
                and manifest.get('context_coverage') != self.support_context):
            raise ValueError('Support must match the explicit full-trajectory context; sampled legacy labels are not a complete support set.')
        if (features.ndim != 2 or noise.ndim < 2 or len(features) != len(noise) or
                len(features) != manifest['support_rows'] or
                not np.isfinite(features).all() or not np.isfinite(noise).all()):
            raise ValueError('Invalid support artifact.')
        # These are distinct quantities: action-only epsilon can be224-D
        # while its archived original joint inverse point is46304-D. Raw
        # points are diagnostics, never PFN targets; preserve every coordinate.
        if raw_points is not None and (raw_points.ndim < 2 or
                                       len(raw_points) != len(noise) or
                                       raw_points.size == 0 or
                                       not np.isfinite(raw_points).all()):
            raise ValueError('Invalid archived original inverse initial points.')
        if hasattr(self.backend, 'requires_native_history_epsilon') and (
                raw_points is None or manifest.get('noise_parameterization') !=
                'epsilon; native history center and sigma retained'):
            raise ValueError('This backend requires the explicit native-history epsilon artifact.')
        condition = PolicyCondition(None, features, manifest['readout_id'], manifest['checkpoint_sha256'])
        self.corrections.append(InvertedCorrection(condition, noise,
            max(manifest['reconstruction_max_errors']), manifest['inversion_recipe_id'],
            {'artifact_reconstruction_metrics': manifest.get('reconstruction_metrics', [])}, raw_points))
        return self

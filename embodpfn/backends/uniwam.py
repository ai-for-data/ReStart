"""Joint-noise inversion/conditioning around the unchanged UniWAM entry.

RoboTwin uses an explicit Torch port of FlowDAgger's fixed-point controller.
The unchanged original policy computes every velocity and reconstruction.
Other native runtimes need their own native callback integration.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from threading import Lock
import json
import time

import numpy as np

import torch
from torch.overrides import TorchFunctionMode

from embodpfn.interfaces import InvertedCorrection, PolicyCondition
from embodpfn.inversion import JointLatent
from embodpfn.flowdagger import invert_native_joint, invert_native_action_only
from embodpfn.backends.native_steps import NativeJointSteps
from embodpfn.upstream import verify_upstream
from embodpfn.native_history_condition import (
    HISTORY_READOUT_ID, JOINT_HISTORY_READOUT_ID, append_action_history)


class _ConditionCaptured(Exception):
    pass


@contextmanager
def _observe_method(owner, name, observer):
    """Delegate to the existing bound method; restore its exact binding."""
    original = getattr(owner, name)
    had_local, previous = name in owner.__dict__, owner.__dict__.get(name)

    def observed(*args, **kwargs):
        return observer(original, args, kwargs)

    setattr(owner, name, observed)
    try:
        yield
    finally:
        if had_local:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


@dataclass
class UniWAMPreparedCondition:
    inputs: dict
    video_initial: torch.Tensor
    action_initial: torch.Tensor
    state: torch.Tensor
    native_action_source: torch.Tensor | None = None
    native_cuda_rng_state: torch.Tensor | None = None
    native_video_random_epsilon: torch.Tensor | None = None


class _NativeHistoryEpsilon(TorchFunctionMode):
    """Replace only native random-factory outputs, before history arithmetic.

    Always execute the original factory first, preserving its RNG consumption.
    Native inference still copies its condition frame and computes history +
    sigma * epsilon. This mode never replaces a composed action latent.
    """

    def __init__(self, backend, inputs, epsilon=None):
        super().__init__()
        self.backend = backend
        self.batch = len(inputs['first_frame'])
        self.action_shape = (self.batch, backend.model.config.action_chunk_size,
                             backend.model.config.action_dim)
        self.epsilon = epsilon
        self.video_count = self.action_count = 0
        self.vector = None
        self.video_coordinates = None
        self.action_source = None
        self.video_random_epsilon = None
        self.video_factory_output = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if (func is torch.randn and output.ndim == 5 and output.shape[0] == self.batch
                and output.shape[2] == 1 + self.backend.model.config.num_video_frames // 4):
            self.video_count += 1
            if self.video_count != 1:
                raise RuntimeError('More than one native video epsilon factory was reached.')
            self.video_factory_output = output
            self.video_coordinates = output[:, :, 1:].numel() // self.batch
            action_only = self.backend.noise_target == 'action'
            if action_only:
                self.video_random_epsilon = output[:, :, 1:].detach().clone()
            if self.epsilon is not None:
                self.vector = torch.as_tensor(self.epsilon, device=output.device, dtype=torch.float32)
                na = self.action_shape[1] * self.action_shape[2]
                if (self.vector.shape != (self.batch, na if action_only else self.video_coordinates + na)
                        or not torch.isfinite(self.vector).all()):
                    raise ValueError('Require every finite future-video and action epsilon coordinate.')
                if not action_only:
                    output[:, :, 1:] = self.vector[:, :self.video_coordinates].reshape_as(
                        output[:, :, 1:]).to(output.dtype)
        elif (func is torch.randn_like and args
                and tuple(args[0].shape) == self.action_shape):
            self.action_count += 1
            if self.action_count != 1 or self.video_count != 1:
                raise RuntimeError('Native history epsilon factory ordering changed.')
            self.action_source = args[0].detach().clone()
            if self.epsilon is not None:
                selected = self.vector if self.backend.noise_target == 'action' else self.vector[:, self.video_coordinates:]
                output = selected.reshape_as(output).to(output.dtype)
        return output

    def require_complete(self):
        if self.video_count != 1 or self.action_count != 1 or self.action_source is None:
            raise RuntimeError('The complete original history/gaussian epsilon boundary was not reached.')


class UniWAMNoiseBackend:
    noise_target = 'joint'
    readout_id = 'uniwam.visible-vlm-adapter-mean+prepared-input-state.joint-noise.v2'
    target_codec = 'joint-noise-centered-support-basis.v1'
    model_source = 'models/motus.py'
    deployed_subtree = None
    supported_source_modes = {'gaussian'}
    libero_release = None

    def __init__(self, official_model, *, inversion_settings, checkpoint_receipt=None):
        self.asset = verify_upstream('uniwam', official_model, model_source=self.model_source,
                                     deployed_subtree=self.deployed_subtree, checkpoint_receipt=checkpoint_receipt,
                                     libero_release=self.libero_release)
        self.model = official_model
        self.official_entry = official_model.inference_step
        self.entry_name = 'inference_step'
        self.settings = inversion_settings
        self._lock = Lock()
        if (official_model.config.flow_source_mode not in self.supported_source_modes or
                official_model.config.flow_source_video_mode not in self.supported_source_modes):
            raise RuntimeError('The native checkpoint source mode is unsupported; never change its configuration to bypass this check.')

    @torch.no_grad()
    def act_official(self, official_inputs):
        return self.official_entry(**official_inputs)

    def _features(self, official_inputs, tokens):
        lengths = self._visible_lengths(official_inputs, tokens)
        if lengths is None:
            pooled = tokens.mean(1)
        else:
            visible = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
            weights = visible.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        # Read the real entry input, without implementing its normalization.
        # Initial video noise is now an output, not part of the condition.
        state = official_inputs['state'].detach().to(pooled.device).float().flatten(1)
        return torch.cat([pooled.float(), state], dim=1).cpu().numpy()

    @staticmethod
    def _pack(latent):
        return torch.cat([latent.video[:, :, 1:].float().flatten(1),
                          latent.action.float().flatten(1)], dim=1)

    def _unpack(self, noise, video_template, action_shape):
        vector = torch.as_tensor(noise, device=video_template.device)
        nv = video_template[:, :, 1:].numel() // len(video_template)
        na = 1
        for size in action_shape[1:]:
            na *= size
        if vector.shape != (len(video_template), nv + na) or not torch.isfinite(vector).all():
            raise ValueError('PFN must provide every future-video and action noise coordinate.')
        video = video_template.clone()
        video[:, :, 1:] = vector[:, :nv].reshape_as(video[:, :, 1:]).to(video.dtype)
        action = vector[:, nv:].reshape(action_shape).to(self.model.dtype)
        return JointLatent(video, action)

    @contextmanager
    def _boundaries(self, official_inputs, *, prepared=None, initial=None, stop=False,
                    predict=None, endpoints=False):
        seen = {'video': 0, 'action': 0}
        captured = {}

        def understanding(module, args, output):
            if 'understanding' not in captured:
                captured['understanding'] = output.detach().clone()

        def video_before(module, args):
            index = seen['video']
            seen['video'] += 1
            latent = args[0]
            if index == 0:
                captured['video_initial'] = latent.detach().clone()
                if prepared is not None and not torch.equal(latent[:, :, :1], prepared.video_initial[:, :, :1]):
                    raise ValueError('Native observed first frame changed during replay.')
                if predict is not None:
                    action_shape = (len(latent), self.model.config.action_chunk_size, self.model.config.action_dim)
                    features = self._features(official_inputs, captured['understanding'])
                    captured['predicted'] = self._unpack(predict(features), latent, action_shape)
                source = initial or captured.get('predicted')
                if source is not None:
                    if source.video.shape != latent.shape or source.video.dtype != latent.dtype:
                        raise ValueError('Joint video noise must match the original source shape/dtype.')
                    if not torch.equal(source.video[:, :, :1], latent[:, :, :1]):
                        raise ValueError('Initial-noise adaptation cannot alter the observed frame.')
                    latent.copy_(source.video)

        def action_before(module, args):
            index = seen['action']
            seen['action'] += 1
            if len(args) != 3:
                raise RuntimeError('Unexpected original StateActionEncoder interface.')
            state, latent, registers = args
            if index == 0:
                captured['state'] = state.detach().clone()
                captured['action_initial'] = latent.detach().clone()
            if prepared is not None and not torch.equal(state, prepared.state):
                raise ValueError('Native prepared state changed during replay.')
            source = initial or captured.get('predicted')
            if index == 0 and source is not None:
                if source.action.shape != latent.shape or source.action.dtype != latent.dtype:
                    raise ValueError('Joint action noise must match the original source shape/dtype.')
                latent.copy_(source.action)
            if stop and index == 0:
                raise _ConditionCaptured()

        def decode(original, args, kwargs):
            if not args:
                raise RuntimeError('Native video decoder call contract changed.')
            captured['video_endpoint'] = args[0].detach().clone()
            return original(*args, **kwargs)

        with ExitStack() as stack:
            if endpoints:
                stack.enter_context(_observe_method(self.model.video_model, 'decode_video', decode))
            if stop or predict is not None:
                stack.callback(self.model.und_expert.vlm_adapter.register_forward_hook(understanding).remove)
            stack.callback(self.model.video_model.wan_model.patch_embedding.register_forward_pre_hook(video_before).remove)
            stack.callback(self.model.action_expert.input_encoder.register_forward_pre_hook(action_before).remove)
            yield captured

    @torch.no_grad()
    def capture(self, official_inputs):
        with self._lock, self._boundaries(official_inputs, stop=True) as captured:
            try:
                self.act_official(official_inputs)
            except _ConditionCaptured:
                pass
            else:
                raise RuntimeError('The native initial-noise boundary was not reached.')
        prepared = UniWAMPreparedCondition(dict(official_inputs), captured['video_initial'],
                                           captured['action_initial'], captured['state'])
        return PolicyCondition(prepared, self._features(official_inputs, captured['understanding']),
                               self.readout_id, self.asset['weight_sha256'])

    @torch.no_grad()
    def act_conditioned(self, official_inputs, predict_noise):
        with self._lock, self._boundaries(official_inputs, predict=predict_noise):
            return self.act_official(official_inputs)

    def _visible_lengths(self, official_inputs, tokens):
        return self.model._build_und_k_lens(official_inputs['vlm_inputs'], tokens)

    def _replay(self, condition, initial):
        prepared = self._prepared(condition)
        device = prepared.action_initial.device
        if device.type != 'cuda':
            raise RuntimeError('This pinned original UniWAM runtime requires CUDA.')
        with self._lock, torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            with self._boundaries(prepared.inputs, prepared=prepared, initial=initial,
                                  endpoints=True) as captured:
                output = self.act_official(prepared.inputs)
                return output, captured

    @torch.no_grad()
    def act_with_initial_noise(self, condition, initial_noise):
        prepared = self._prepared(condition)
        initial = self._unpack(initial_noise, prepared.video_initial, prepared.action_initial.shape)
        return self._replay(condition, initial)[0]

    @torch.no_grad()
    def invert_expert_action(self, condition, official_model_space_actions):
        if self.settings is None:
            raise ValueError('Explicit inverse settings are required.')
        prepared = self._prepared(condition)
        initial = JointLatent(prepared.video_initial, prepared.action_initial)
        target = torch.as_tensor(official_model_space_actions,
            device=initial.action.device, dtype=torch.float32)
        if target.shape != initial.action.shape:
            raise ValueError('Expert target must have the complete native action shape.')

        def full_forward(noise):
            output, observed = self._replay(condition, noise)
            return JointLatent(observed['video_endpoint'], output[1])

        result = self._solve_inverse(condition, target, initial, full_forward)
        return InvertedCorrection(condition, self._pack(result.noise).cpu().numpy(),
            result.reconstruction_max_error, self.settings.recipe_id,
            result.reconstruction_metrics)

    def _solve_inverse(self, condition, target, initial, full_forward):
        raise NotImplementedError('This native runtime has no verified discrete-step root protocol. '
                                  'Do not fall back to gradient loss optimization.')

    def _prepared(self, condition):
        if (condition.readout_id != self.readout_id or
                condition.policy_checkpoint_sha256 != self.asset['weight_sha256']):
            raise ValueError('Condition belongs to another checkpoint/readout.')
        return condition.official_condition


class RoboTwinUniWAMNoiseBackend(UniWAMNoiseBackend):
    """Observe the original RoboTwin source, including validated history flow."""
    supported_source_modes = {'gaussian', 'history'}
    model_source = 'inference/robotwin/uniwam/models/motus.py'
    deployed_subtree = 'inference/robotwin/uniwam'
    readout_id = JOINT_HISTORY_READOUT_ID
    requires_native_history_epsilon = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.model.config
        if (config.flow_source_mode != 'history' or config.flow_source_video_mode != 'gaussian'
                or config.flow_source_action_noise_std <= 0):
            raise RuntimeError('This epsilon contract requires the unchanged native history-action/gaussian-video profile.')
        self.query_trace = None
        self.query_count = 0

    @torch.no_grad()
    def _capture_native_condition(self, official_inputs):
        with _NativeHistoryEpsilon(self, official_inputs) as observed:
            condition = super().capture(official_inputs)
        observed.require_complete()
        condition.official_condition.native_action_source = observed.action_source
        condition.official_condition.native_video_random_epsilon = observed.video_random_epsilon
        return condition

    @torch.no_grad()
    def capture(self, official_inputs):
        condition = self._capture_native_condition(official_inputs)
        features = append_action_history(condition.features,
            condition.official_condition.native_action_source.detach().float().cpu().numpy())
        return PolicyCondition(condition.official_condition, features, self.readout_id,
                               condition.policy_checkpoint_sha256)

    @torch.no_grad()
    def act_conditioned(self, official_inputs, predict_noise):
        with self._lock:
            # Keep the existing VLM readout; the native factories expose H
            # after their original RNG draws and before any denoising step.
            tokens = self.model.und_module.extract_und_features(official_inputs['vlm_inputs'])
            features = self._features(official_inputs, tokens)
            with _NativeHistoryPrediction(self, official_inputs, features, predict_noise) as injected:
                output = self.act_official(official_inputs)
            injected.require_complete()
            return output

    def begin_history_prediction(self, features, original_random):
        self.query_count += 1
        if self.query_trace is not None:
            record = {'kind': self._history_trace_kind(), 'event': 'query-start',
                'action_history_PFN_query': self.query_count, 'noise_target': self.noise_target,
                'features': features.tolist()}
            if self.noise_target == 'action':
                record['original_random_epsilon'] = original_random.detach().float().flatten(1).cpu().tolist()
            with self.query_trace.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        print(json.dumps({'action_history_PFN_query_started': self.query_count,
            'condition_features': features.shape[1], 'noise_target': self.noise_target}), flush=True)
        return self.query_count

    def complete_history_prediction(self, query_id, features, epsilon, applied, elapsed):
        summary = {'action_history_PFN_query': query_id,
            'condition_features': features.shape[1], 'epsilon_dimensions': epsilon.shape[1],
            'PFN_callback_wall_seconds': elapsed, 'noise_target': self.noise_target,
            'native_history_and_random_factories_retained': True}
        if self.query_trace is not None:
            record = dict(summary, kind=self._history_trace_kind(), event='query-completed')
            if self.noise_target == 'action':
                record.update(predicted_epsilon=epsilon.tolist(),
                    applied_epsilon=applied.detach().float().flatten(1).cpu().tolist())
            with self.query_trace.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        print(json.dumps(summary), flush=True)

    def _history_trace_kind(self):
        return ('native-history-conditioned-action-query.v1' if self.noise_target == 'action'
                else 'native-history-conditioned-joint-query.v1')

    @torch.no_grad()
    def act_with_initial_noise(self, condition, initial_noise):
        prepared = self._prepared(condition)
        device = prepared.action_initial.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with self._lock, torch.random.fork_rng(devices=devices):
            if prepared.native_cuda_rng_state is not None:
                torch.cuda.set_rng_state(prepared.native_cuda_rng_state, device)
            with _NativeHistoryEpsilon(self, prepared.inputs, initial_noise) as injected:
                output = self.act_official(prepared.inputs)
            injected.require_complete()
            if (self.noise_target == 'action' and not torch.equal(
                    injected.video_random_epsilon, prepared.native_video_random_epsilon)):
                raise ValueError('Action-only reconstruction must retain its original native random video sample.')
            if not torch.equal(injected.action_source, prepared.native_action_source):
                raise ValueError('Native history center changed during epsilon reconstruction.')
            return output

    @torch.no_grad()
    def invert_expert_action(self, condition, official_model_space_actions):
        # The unchanged fixed-point port solves x0*. Reparameterization is
        # analytic, with no extra solver iterations or acceptance tolerance.
        correction = super().invert_expert_action(condition, official_model_space_actions)
        prepared = self._prepared(condition)
        point = torch.as_tensor(correction.initial_noise, device=prepared.action_initial.device)
        epsilon = point.clone()
        na = prepared.action_initial.numel() // len(prepared.action_initial)
        sigma = self.model.config.flow_source_action_noise_std
        epsilon[:, -na:] = (point[:, -na:] - prepared.native_action_source.float().flatten(1)) / sigma
        if self.noise_target == 'action':
            epsilon = epsilon[:, -na:]
        if not torch.isfinite(epsilon).all():
            raise RuntimeError('Native history reparameterization produced nonfinite epsilon.')
        epsilon = epsilon.cpu().numpy()
        _, reconstructed = self.act_with_initial_noise(condition, epsilon)
        target = torch.as_tensor(official_model_space_actions, device=reconstructed.device,
                                 dtype=torch.float32)
        error = reconstructed.float() - target
        if not torch.isfinite(error).all():
            raise RuntimeError('Native epsilon replay produced nonfinite actions.')
        metrics = dict(correction.reconstruction_metrics)
        metrics['initial_point_reconstruction'] = {
            key: metrics[key] for key in ('action_mse', 'action_mse_per_chunk', 'action_max_error',
                                         'video_future_mse')}
        # The old video metric describes x0 replay, not this fresh epsilon
        # replay. Keep it only inside its explicitly named original report.
        metrics.pop('video_future_mse')
        metrics.update(noise_parameterization='epsilon; native history + native sigma * epsilon',
            noise_target=self.noise_target,
            native_action_noise_std=float(sigma),
            native_action_source=prepared.native_action_source.float().cpu().tolist(),
            action_mse=float(error.square().mean()),
            action_mse_per_chunk=error.square().mean(dim=(-2, -1)).cpu().tolist(),
            action_max_error=float(error.abs().max()),
            epsilon_reconstruction_velocity_evaluations=int(prepared.inputs['num_inference_steps']))
        return InvertedCorrection(condition, epsilon, metrics['action_max_error'],
            correction.inversion_recipe_id, metrics, correction.initial_noise)

    def _visible_lengths(self, official_inputs, tokens):
        return torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device)

    def _solve_inverse(self, condition, target, initial, full_forward):
        replay = NativeJointSteps(self, condition, _observe_method)
        return invert_native_joint(target_action=target, replay=replay, settings=self.settings)


class RoboTwinUniWAMActionNoiseBackend(RoboTwinUniWAMNoiseBackend):
    """Preserve native random video; condition action epsilon on native H."""
    noise_target = 'action'
    readout_id = HISTORY_READOUT_ID

    @torch.no_grad()
    def capture(self, official_inputs):
        device = next(self.model.parameters()).device
        rng = torch.cuda.get_rng_state(device).clone()
        condition = super().capture(official_inputs)
        condition.official_condition.native_cuda_rng_state = rng
        return condition

    def _solve_inverse(self, condition, target, initial, full_forward):
        replay = NativeJointSteps(self, condition, _observe_method)
        return invert_native_action_only(target_action=target, replay=replay, settings=self.settings)


class _NativeHistoryPrediction(_NativeHistoryEpsilon):
    """Query once at actual H, then retain the native latent composition."""

    def __init__(self, backend, inputs, features, predict_noise):
        super().__init__(backend, inputs)
        self.features = features
        self.predict_noise = predict_noise

    def __torch_function__(self, func, types, args=(), kwargs=None):
        previous = self.action_count
        # The original factory and its RNG consumption always execute first.
        output = super().__torch_function__(func, types, args, kwargs)
        if self.action_count == previous:
            return output
        features = append_action_history(self.features,
            self.action_source.detach().float().cpu().numpy())
        query_id = self.backend.begin_history_prediction(features, output)
        started = time.perf_counter()
        epsilon = np.asarray(self.predict_noise(features), dtype=np.float32)
        elapsed = time.perf_counter() - started
        action_width = self.action_shape[1] * self.action_shape[2]
        action_only = self.backend.noise_target == 'action'
        width = action_width if action_only else self.video_coordinates + action_width
        if epsilon.shape != (self.batch, width) or not np.isfinite(epsilon).all():
            raise ValueError('Predict every executable native epsilon coordinate.')
        if not action_only:
            # motus.py:945-978 keeps this exact factory output as video_latent.
            # Only its observed first frame is assigned before the action
            # factory; no encoder/velocity has consumed the future frames.
            video = self.video_factory_output[:, :, 1:]
            video.copy_(torch.as_tensor(epsilon[:, :self.video_coordinates],
                device=video.device).reshape_as(video).to(video.dtype))
        selected = epsilon if action_only else epsilon[:, self.video_coordinates:]
        applied = torch.as_tensor(selected, device=output.device).reshape_as(output).to(output.dtype)
        self.backend.complete_history_prediction(query_id, features, epsilon, applied, elapsed)
        return applied


class LiberoUniWAMNoiseBackend(RoboTwinUniWAMActionNoiseBackend):
    """Native LIBERO sampler; predict action epsilon, preserve random video."""
    model_source = 'models/motus.py'
    deployed_subtree = None
    native_schedule_kind = 'inline-linspace'
    readout_id = 'uniwam-libero.visible-vlm-adapter-mean+native-state+native-action-source.native-action-epsilon.v3'

    def __init__(self, *args, checkpoint_receipt=None, **kwargs):
        import json
        from pathlib import Path
        from embodpfn.upstream import LIBERO_RELEASES
        receipt = json.loads(Path(checkpoint_receipt).read_text())
        matches = [mode for mode, (repo, revision, _) in LIBERO_RELEASES.items()
                   if (receipt['official_release']['repository'], receipt['official_release']['revision'])
                   == (repo, revision)]
        if len(matches) != 1:
            raise ValueError('Bind the actual original LIBERO/Plus release before native loading.')
        self.libero_release = matches[0]
        super().__init__(*args, checkpoint_receipt=checkpoint_receipt, **kwargs)
        if (self.model.config.action_dim, self.model.config.action_state_dim) != (7, 8):
            raise ValueError('Use the unchanged original seven-action/eight-state LIBERO model.')

    def _visible_lengths(self, official_inputs, tokens):
        return UniWAMNoiseBackend._visible_lengths(self, official_inputs, tokens)

    @torch.no_grad()
    def act_conditioned(self, official_inputs, predict_noise):
        with self._lock:
            tokens = self.model.und_module.extract_und_features(official_inputs['vlm_inputs'])
            if isinstance(tokens, tuple):
                tokens = tokens[0]  # Same return contract handled by native root Motus.
            features = self._features(official_inputs, tokens)
            with _NativeHistoryPrediction(self, official_inputs, features, predict_noise) as injected:
                output = self.act_official(official_inputs)
            injected.require_complete()
            return output

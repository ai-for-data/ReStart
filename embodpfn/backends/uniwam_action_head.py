"""Declared H-anchor readout; original sampler still executes all four steps."""
from contextlib import ExitStack
import hashlib
import inspect
import json
import time

import numpy as np
import torch

from embodpfn.backends.uniwam import (RoboTwinUniWAMActionNoiseBackend,
    _NativeHistoryEpsilon, _observe_method)
from embodpfn.native_history_condition import append_action_history


READOUT_ID = 'uniwam-robotwin.native-H-anchor-first-action-head-mean+state+complete-H.action-epsilon.v1'
READOUT_FEATURES = 1262


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


class _HeadCaptured(Exception):
    """Own auxiliary-call stop signal; never catch a native model failure."""


class _AnchorMode(_NativeHistoryEpsilon):
    def __init__(self, backend, inputs, archived_video=None):
        zeros = np.zeros((len(inputs['first_frame']), 224), dtype=np.float32)
        super().__init__(backend, inputs, zeros)
        self.archived_video = archived_video

    def __torch_function__(self, func, types, args=(), kwargs=None):
        previous = self.video_count
        output = super().__torch_function__(func, types, args, kwargs)
        if self.video_count != previous:
            if self.archived_video is not None:
                saved = np.asarray(self.archived_video)
                actual = output[:, :, 1:]
                if saved.shape != (len(output), actual.numel() // len(output)) or saved.dtype != np.float32:
                    raise ValueError('Replay every archived original future-video coordinate, without projection.')
                actual.copy_(torch.as_tensor(saved, device=output.device).reshape_as(actual).to(actual.dtype))
                if not np.array_equal(actual.detach().float().flatten(1).cpu().numpy(), saved):
                    raise ValueError('Archived V0 is not exactly representable in its original native dtype.')
            self.video_random_epsilon = output[:, :, 1:].detach().clone()
        return output


class _MainEpsilonMode(_NativeHistoryEpsilon):
    def __init__(self, backend, inputs, epsilon, anchor):
        super().__init__(backend, inputs)
        self.prediction = epsilon
        self.anchor = anchor
        self.original_epsilon = self.applied_epsilon = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        previous = self.action_count
        output = super().__torch_function__(func, types, args, kwargs)
        if self.action_count != previous:
            future = self.video_random_epsilon.detach().float().flatten(1).cpu().numpy()
            full_video = self.video_factory_output.detach().float().flatten(1).cpu().numpy()
            history = self.action_source.detach().float().cpu().numpy()
            if (not np.array_equal(future, self.anchor['future_V0'])
                    or not np.array_equal(full_video, self.anchor['full_V0'])
                    or not np.array_equal(history, self.anchor['H'])):
                raise ValueError('Auxiliary and executed main calls must have the exact same original V0 and H.')
            self.original_epsilon = output.detach().float().flatten(1).cpu().numpy().copy()
            self.main_V0_sha256 = array_sha(future)
            self.main_full_V0_sha256 = array_sha(full_video)
            self.main_H_sha256 = array_sha(history)
            applied = torch.as_tensor(self.prediction, device=output.device).reshape_as(output).to(output.dtype)
            self.applied_epsilon = applied.detach().float().flatten(1).cpu().numpy().copy()
            if not np.isfinite(self.applied_epsilon).all():
                raise ValueError('The native epsilon dtype conversion must remain finite.')
            self.applied_native_dtype = str(applied.dtype)
            return applied
        return output


class RoboTwinUniWAMHAnchorHeadBackend(RoboTwinUniWAMActionNoiseBackend):
    readout_id = READOUT_ID
    readout_features = READOUT_FEATURES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.model.config
        expert = self.model.action_expert
        if ((cfg.action_chunk_size, cfg.action_dim, cfg.action_expert_dim, cfg.num_layers) != (16, 14, 1024, 30)
                or expert.config.num_registers != 4 or cfg.flow_source_action_noise_std != .02):
            raise ValueError('Use the unchanged original16x14/1024 head and30-layer history-flow profile.')
        self.head_trace = None

    def capture(self, official_inputs):
        raise RuntimeError('This readout requires explicit same-source saved-label feature replay; no implicit inversion.')

    def invert_expert_action(self, condition, official_model_space_actions):
        raise RuntimeError('Do not silently recompute labels for this feature-only reuse method.')

    def _validate_inputs(self, inputs):
        if (inputs.get('num_inference_steps') != 4 or inputs.get('future_video_denoise_fraction') != 1.0
                or inputs['state'].shape != (1, 14) or inputs['first_frame'].shape[0] != 1):
            raise ValueError('Keep the complete original single-query4-step native recipe.')

    def _anchor_locked(self, inputs, archived_video=None):
        """Call only with our existing backend lock held exactly once."""
        self._validate_inputs(inputs)
        device = inputs['first_frame'].device
        if device.type != 'cuda':
            raise ValueError('The pinned original UniWAM pipeline is CUDA-native.')
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone()
        captured = {}
        observations = 0
        anchor_inputs = 0
        auxiliary_schedules = 0

        def anchored_input(module, arguments):
            nonlocal anchor_inputs
            anchor_inputs += 1
            if (anchor_inputs != 1 or len(arguments) != 3
                    or arguments[1].shape != (1, 16, 14)
                    or not torch.equal(arguments[1], mode.action_source)):
                raise RuntimeError('The auxiliary original action encoder must receive exactly H, never solved A0*.')

        def auxiliary_schedule(original, args, kwargs):
            nonlocal auxiliary_schedules
            result = original(*args, **kwargs)
            auxiliary_schedules += 1
            if auxiliary_schedules != 1 or any(len(grid) != 5 or float(grid[0]) != 1.0 or float(grid[-1]) != 0.0 for grid in result):
                raise RuntimeError('The auxiliary readout must reach the original first-time1 four-step grid.')
            return result

        def head_before(module, arguments):
            nonlocal observations
            observations += 1
            if observations != 1 or len(arguments) != 1 or arguments[0].shape != (1, 21, 1024):
                raise RuntimeError('Read the first real native head input: state1 + actions16 + registers4.')
            mode.require_complete()
            z = arguments[0]
            pooled = z[:, 1:17].float().mean(1).detach().cpu().numpy()
            state = inputs['state'].detach().float().cpu().numpy()
            H = mode.action_source.detach().float().cpu().numpy().copy()
            captured['features'] = append_action_history(np.concatenate((pooled, state), axis=1), H)
            captured['H'] = H
            captured['future_V0'] = mode.video_random_epsilon.detach().float().flatten(1).cpu().numpy().copy()
            captured['full_V0'] = mode.video_factory_output.detach().float().flatten(1).cpu().numpy().copy()
            raise _HeadCaptured()

        started = time.perf_counter()
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            with ExitStack() as stack:
                stack.callback(self.model.action_expert.decoder.action_head.register_forward_pre_hook(head_before).remove)
                stack.callback(self.model.action_expert.input_encoder.register_forward_pre_hook(anchored_input).remove)
                stack.enter_context(_observe_method(inspect.getmodule(self.official_entry), 'build_joint_denoising_timesteps', auxiliary_schedule))
                with _AnchorMode(self, inputs, archived_video) as mode:
                    try:
                        self.official_entry(**inputs)
                    except _HeadCaptured:
                        pass
                    else:
                        raise RuntimeError('The auxiliary call must stop at its original first head before Euler.')
        if observations != 1 or anchor_inputs != 1 or auxiliary_schedules != 1 or not torch.equal(cpu_before, torch.get_rng_state()) or not torch.equal(cuda_before, torch.cuda.get_rng_state(device)):
            raise RuntimeError('The auxiliary original call must restore CPU/CUDA RNG exactly.')
        captured['CPU_rng_before'] = cpu_before
        captured['CUDA_rng_before'] = cuda_before
        captured['proof'] = {'readout_id': self.readout_id, 'condition_features': 1262,
            'native_head_input_shape': [1, 21, 1024], 'native_action_positions': [1, 17],
            'auxiliary_first_head_prefixes': 1, 'anchor_epsilon': 0,
            'actual_auxiliary_action_initial_equals_H': True,
            'first_native_time': 1.0,
            'CPU_CUDA_rng_restored': True, 'H_sha256': array_sha(captured['H']),
            'future_V0_sha256': array_sha(captured['future_V0']),
            'full_V0_sha256': array_sha(captured['full_V0']),
            'auxiliary_wall_seconds': time.perf_counter() - started}
        return captured

    @torch.no_grad()
    def extract_h_anchor(self, official_inputs, archived_video):
        with self._lock:
            return self._anchor_locked(official_inputs, archived_video)

    @torch.no_grad()
    def act_conditioned(self, official_inputs, predict_noise):
        with self._lock:
            anchor = self._anchor_locked(official_inputs)
            self.query_count += 1
            query = self.query_count
            began = dict(anchor['proof'], kind='native-H-anchor-action-head-query.v1', event='query-start',
                query=query, features=anchor['features'].tolist())
            self._trace(began)
            started = time.perf_counter()
            epsilon = np.asarray(predict_noise(anchor['features']), dtype=np.float32)
            callback_seconds = time.perf_counter() - started
            if epsilon.shape != (1, 224) or not np.isfinite(epsilon).all():
                raise ValueError('Predict all224 executable native action epsilon coordinates.')
            device = official_inputs['first_frame'].device
            if (not torch.equal(anchor['CPU_rng_before'], torch.get_rng_state())
                    or not torch.equal(anchor['CUDA_rng_before'], torch.cuda.get_rng_state(device))):
                raise RuntimeError('Feature readout and PFN callback must leave native main-call CPU/CUDA RNG unchanged.')
            heads = schedules = 0

            def head_after(module, args, result):
                nonlocal heads
                heads += 1
                if result.shape != (1, 21, 14):
                    raise RuntimeError('Keep every original head output before the native action slice.')

            def schedule(original, args, kwargs):
                nonlocal schedules
                result = original(*args, **kwargs)
                schedules += 1
                if schedules != 1 or any(len(grid) != 5 or float(grid[0]) != 1.0 or float(grid[-1]) != 0.0 for grid in result):
                    raise RuntimeError('Use the original first-time1 to0 four-step grids.')
                return result

            native = inspect.getmodule(self.official_entry)
            main_started = time.perf_counter()
            with ExitStack() as stack:
                stack.callback(self.model.action_expert.decoder.action_head.register_forward_hook(head_after).remove)
                stack.enter_context(_observe_method(native, 'build_joint_denoising_timesteps', schedule))
                with _MainEpsilonMode(self, official_inputs, epsilon, anchor) as main:
                    result = self.official_entry(**official_inputs)
                main.require_complete()
            if heads != 4 or schedules != 1 or main.applied_epsilon is None:
                raise RuntimeError('The executed main call must complete all original4 action heads and one native grid.')
            ended = dict(anchor['proof'], kind='native-H-anchor-action-head-query.v1', event='query-completed',
                query=query, main_native_steps=4, auxiliary_main_V0_equal=True, auxiliary_main_H_equal=True,
                main_future_V0_sha256=main.main_V0_sha256, main_H_sha256=main.main_H_sha256,
                main_full_V0_sha256=main.main_full_V0_sha256, PFN_callback_preserved_CPU_CUDA_RNG=True,
                applied_native_dtype=main.applied_native_dtype,
                epsilon_dimensions=224, predicted_epsilon=epsilon.tolist(),
                applied_epsilon=main.applied_epsilon.tolist(), original_random_epsilon=main.original_epsilon.tolist(),
                PFN_callback_wall_seconds=callback_seconds,
                main_native_wall_seconds=time.perf_counter() - main_started,
                original_history_and_random_factories_retained=True)
            self._trace(ended)
            print(json.dumps({key: value for key, value in ended.items() if key not in
                ('features', 'predicted_epsilon', 'applied_epsilon', 'original_random_epsilon')}), flush=True)
            return result

    def _trace(self, value):
        if self.head_trace is not None:
            with self.head_trace.open('a') as stream:
                stream.write(json.dumps(value) + '\n')

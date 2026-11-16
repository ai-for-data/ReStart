"""Observe native Euler step maps; memoize only their unchanged prefixes.

No denoiser, time grid, velocity update or decoder is implemented here.
The pinned native entry executes them. Offline queries replace a step's
actual state before its native dtype cast, and observe the next actual state.
"""
from collections import defaultdict
from contextlib import ExitStack
import inspect
import time

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_map

from embodpfn.inversion import JointLatent, InversionFailure
from embodpfn.backends.condition_cache import FrozenConditionCache


def _constant(value):
    return tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x, value)


class _EndpointReady(Exception):
    def __init__(self, value):
        self.value = value


class _NativeVideoState(TorchFunctionMode):
    """Observe the original latent, before Tensor.to creates a cast copy."""

    def __init__(self, replay):
        super().__init__()
        self.replay = replay

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        replay = self.replay
        if (replay.velocity_time_index is not None and replay.index == replay.target
                and func in (torch.Tensor.__mul__, torch.Tensor.mul, torch.mul)):
            replay.observe_velocity_product(args)
        if func is torch.Tensor.to and args and not replay.in_decoder:
            dtype = kwargs.get('dtype')
            if dtype is None:
                dtype = next((x for x in args[1:] if isinstance(x, torch.dtype)), None)
            value = args[0]
            if (dtype == replay.model.dtype and isinstance(value, torch.Tensor)
                    and tuple(value.shape) == tuple(replay.initial.video.shape)):
                replay.video_input(value)
        output = func(*args, **kwargs)
        if func is torch.linspace and replay.inline_schedule:
            replay.observe_inline_schedule(args, kwargs, output)
        return output


class NativeJointSteps:
    """One real condition, one native reference, exact prefix reuse.

    Every queried prefix starts at the same captured original source.
    Its actual states are required to equal the reference states bitwise.
    Current-step layers always run natively; only preceding identical layers
    return copies of their native reference outputs. The online policy never
    installs these observers.
    """

    def __init__(self, backend, condition, observe_method):
        self.backend = backend
        self.model = backend.model
        self.condition = condition
        self.prepared = backend._prepared(condition)
        self.initial = JointLatent(self.prepared.video_initial, self.prepared.action_initial)
        self.steps = int(self.prepared.inputs['num_inference_steps'])
        self.observe_method = observe_method
        self.conditions = FrozenConditionCache(self.model)
        self.states = []
        self.outputs = {}
        self.hits = defaultdict(int)
        self.real_layers = 0
        self.queries = 0
        self.full_queries = 0
        self.full_verifications = 0
        self.released_prefix_entries = 0
        self.query_seconds = 0.0
        self.cuda_events = []
        self.timesteps = None
        self.velocity_time_index = None
        self.inline_schedule = getattr(backend, 'native_schedule_kind', None) == 'inline-linspace'

    def observe_inline_schedule(self, args, kwargs, grid):
        # Root Motus creates this grid inside its real inference_step rather
        # than calling RoboTwin's separate schedule helper. Observe the actual
        # factory return; never replace the factory or manufacture a grid.
        if (len(args) < 3 or args[:3] != (1.0, 0.0, self.steps + 1)
                or kwargs.get('dtype') != self.model.dtype):
            return
        if self.schedule_observations:
            raise InversionFailure('More than one native root denoising grid was observed.')
        self.schedule_observations += 1
        grids = (grid, grid)
        if self.timesteps is None:
            self.timesteps = tuple(value.detach().clone() for value in grids)
        elif not all(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
                     and torch.equal(a, b) for a, b in zip(self.timesteps, grids)):
            raise InversionFailure('The original root finite-precision time grid changed.')

    def _schedule(self, original, args, kwargs):
        grids = original(*args, **kwargs)
        if self.timesteps is None:
            self.timesteps = tuple(x.detach().clone() for x in grids)
        elif not all(torch.equal(a, b) for a, b in zip(self.timesteps, grids)):
            raise InversionFailure('The original native time grid changed between queries.')
        return grids

    def step_deltas(self, step):
        return tuple((grid[step + 1] - grid[step]).float() for grid in self.timesteps)

    def _time_embedding(self, modality):
        def call(original, args, kwargs):
            if self.velocity_time_index is not None and self.index == self.target:
                # The existing native method expects t scaled by 1000. Only
                # offline velocity queries supply another original grid point;
                # the original sampler's grid and updates are never changed.
                t = self.timesteps[modality][self.velocity_time_index]
                scaled = (t * 1000).expand(self.initial.action.shape[0]).to(self.model.dtype)
                return original(scaled, *args[1:], **kwargs)
            return original(*args, **kwargs)
        return call

    def observe_velocity_product(self, args):
        """Read the real v operand of the native Euler multiply, not (g-x)/h."""
        if len(args) < 2 or not all(isinstance(x, torch.Tensor) for x in args[:2]):
            return
        for value, delta in (args[:2], args[:2][::-1]):
            if delta.ndim != 0:
                continue
            for modality, latent in enumerate((self.initial.video, self.initial.action)):
                if tuple(value.shape) != tuple(latent.shape):
                    continue
                grid = self.timesteps[modality]
                if torch.equal(delta, grid[self.target + 1] - grid[self.target]):
                    self.velocities[modality] = value
        if len(self.velocities) == 2:
            raise _EndpointReady(JointLatent(self.velocities[0], self.velocities[1]))

    def video_input(self, value):
        self.index += 1
        if self.index >= self.steps:
            raise InversionFailure('Unexpected native video-state cast boundary.')
        if self.index == 0:
            if not torch.equal(value[:, :, :1], self.initial.video[:, :, :1]):
                raise InversionFailure('Native observed frame changed.')
            value.copy_(self.initial.video)
        if self.target is not None:
            if self.index < self.target:
                if not torch.equal(value, self.states[self.index].video):
                    raise InversionFailure('Native prefix video no longer equals its reference.')
            elif self.index == self.target:
                self._replace(value, self.candidate.video)
        self.current_video = value

    @staticmethod
    def _replace(native, candidate):
        if native.shape != candidate.shape or native.dtype != candidate.dtype:
            raise InversionFailure('Step candidate must retain the original state shape and dtype.')
        native.copy_(candidate)

    def action_input(self, module, args):
        if len(args) != 3 or not torch.equal(args[0], self.prepared.state):
            raise InversionFailure('Original state/action encoder contract changed.')
        value = args[1]
        if self.index == 0:
            value.copy_(self.initial.action)
        if self.target is None:
            self.states.append(JointLatent(self.current_video.detach().clone(), value.detach().clone()))
        elif self.index < self.target:
            if not torch.equal(value, self.states[self.index].action):
                raise InversionFailure('Native prefix action no longer equals its reference.')
        elif self.index == self.target:
            self._replace(value, self.candidate.action)
        elif self.index == self.target + 1:
            # Both values are the real outputs of the native preceding update.
            # Preserve their graph; do not reconstruct an Euler update here.
            raise _EndpointReady(JointLatent(self.current_video, value))

    def _memo(self, key):
        def call(original, args, kwargs):
            position = self.counts[(key, self.index)]
            self.counts[(key, self.index)] += 1
            cache_key = (key, self.index, position)
            if self.target is not None and self.index < self.target:
                if cache_key not in self.outputs:
                    raise InversionFailure('Missing original prefix output: ' + key)
                self.hits[key] += 1
                return _constant(self.outputs[cache_key])
            output = original(*args, **kwargs)
            if key == 'joint_attention':
                self.real_layers += 1
            if self.target is None:
                self.outputs[cache_key] = _constant(output)
            return output
        return call

    def _run(self, target=None, candidate=None, *, velocity_time_index=None):
        self.target, self.candidate = target, candidate
        self.velocity_time_index = velocity_time_index
        self.velocities = {}
        self.index, self.in_decoder = -1, False
        self.counts = defaultdict(int)
        self.current_video = None
        self.schedule_observations = 0
        final_video = None

        def decode(original, args, kwargs):
            nonlocal final_video
            self.in_decoder = True
            # Keep the original tensor: cloning under native no_grad would
            # discard the current step's graph.
            final_video = args[0]
            return original(*args, **kwargs)

        methods = (
            ('video_input', self.model.video_module, 'prepare_input'),
            ('joint_attention', self.model.video_module, 'process_joint_attention'),
            ('video_cross', self.model.video_module, 'process_cross_attention'),
            ('video_ffn', self.model.video_module, 'process_ffn'),
            ('action_ffn', self.model.action_module, 'process_ffn'),
            ('understanding_ffn', self.model.und_module, 'process_ffn'),
            ('video_head', self.model.video_module, 'apply_output_head'),
            ('action_head', self.model.action_expert.decoder, 'forward'),
        )
        device = self.initial.action.device
        if device.type != 'cuda':
            raise InversionFailure('The pinned native step replay requires CUDA.')
        started = time.perf_counter()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with self.backend._lock, torch.random.fork_rng(devices=[device.index or 0]):
            with ExitStack() as stack:
                stack.enter_context(self.conditions.install())
                if not self.inline_schedule:
                    native_module = inspect.getmodule(self.backend.official_entry)
                    stack.enter_context(self.observe_method(
                        native_module, 'build_joint_denoising_timesteps', self._schedule))
                stack.enter_context(self.observe_method(
                    self.model.video_module, 'get_time_embedding', self._time_embedding(0)))
                stack.enter_context(self.observe_method(
                    self.model.action_module, 'get_time_embedding', self._time_embedding(1)))
                for key, owner, name in methods:
                    stack.enter_context(self.observe_method(owner, name, self._memo(key)))
                stack.enter_context(self.observe_method(self.model.video_model, 'decode_video', decode))
                stack.callback(self.model.action_expert.input_encoder.register_forward_pre_hook(self.action_input).remove)
                with torch.no_grad():
                    inputs = tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x,
                                      self.prepared.inputs)
                begin.record()
                try:
                    with torch.no_grad(), _NativeVideoState(self):
                        output = self.backend.official_entry(**inputs)
                except _EndpointReady as captured:
                    result = captured.value
                else:
                    if velocity_time_index is not None:
                        raise InversionFailure('Did not observe both original velocity operands.')
                    if final_video is None or self.index != self.steps - 1:
                        raise InversionFailure('Did not observe the complete original native step sequence.')
                    result = JointLatent(final_video, output[1])
                finally:
                    end.record()
        self.cuda_events.append((begin, end))
        if self.inline_schedule and self.schedule_observations != 1:
            raise InversionFailure('The original root denoising grid was not observed.')
        self.query_seconds += time.perf_counter() - started
        self.queries += 1
        return result

    def reference(self):
        endpoint = self._run()
        if len(self.states) != self.steps:
            raise InversionFailure('Native reference did not expose every original step input.')
        self.terminal = JointLatent(endpoint.video.detach().clone(), endpoint.action.detach().clone())
        return self.terminal

    def query(self, step, candidate):
        return self._run(step, candidate)

    def velocity(self, step, candidate, *, time_index):
        """Run the native denoiser at a supplied original grid time, offline."""
        if not 0 <= time_index <= self.steps:
            raise ValueError('Velocity time must be on the original native grid.')
        native = self.states[step]
        video = native.video.clone()
        video[:, :, 1:] = candidate.video[:, :, 1:].to(video.dtype)
        action = candidate.action.to(native.action.dtype)
        return self._run(step, JointLatent(video, action), velocity_time_index=time_index)

    def verify_step(self, step):
        actual = self.query(step, self.states[step])
        expected = self.states[step + 1] if step + 1 < self.steps else self.terminal
        if not (torch.equal(actual.video, expected.video) and torch.equal(actual.action, expected.action)):
            raise InversionFailure('Memoized native step differs from the original reference.')

    def verify_complete(self, noise):
        # Use the full original entry, with every condition encoder executing.
        # Verification observers compare outputs; they never substitute them.
        with self.conditions.install(verify=True):
            output, observed = self.backend._replay(self.condition, noise)
        self.full_verifications += 1
        return JointLatent(observed['video_endpoint'], output[1])

    def full_query(self, noise):
        """Complete original sampler; only frozen condition outputs are reused."""
        started = time.perf_counter()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        with self.conditions.install():
            output, observed = self.backend._replay(self.condition, noise)
        end.record()
        self.cuda_events.append((begin, end))
        self.query_seconds += time.perf_counter() - started
        self.full_queries += 1
        return JointLatent(observed['video_endpoint'], output[1])

    def release_prefix(self):
        """Reverse steps are complete; their reference activations are dead."""
        self.released_prefix_entries += len(self.outputs)
        self.outputs.clear()

    def diagnostics(self):
        torch.cuda.synchronize(self.initial.action.device)
        return {'native_step_queries_including_reference_and_checks': self.queries,
                'native_joint_attention_layer_evaluations': self.real_layers,
                'complete_native_equation_queries': self.full_queries,
                'fresh_complete_native_verifications': self.full_verifications,
                'reference_prefix_entries_released': self.released_prefix_entries,
                'derived_total_joint_attention_layer_evaluations': self.real_layers +
                    (self.full_queries + self.full_verifications) * self.steps * int(self.model.config.num_layers),
                'unchanged_prefix_cache_hits': dict(self.hits),
                'native_query_cuda_milliseconds': sum(a.elapsed_time(b) for a, b in self.cuda_events),
                'query_host_elapsed_seconds': self.query_seconds,
                'frozen_condition_cache': self.conditions.diagnostics()}

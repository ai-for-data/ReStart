"""Explicit Torch backend port of FlowDAgger's pi0.5 fixed-point controller.

The upstream controller is bound to JAX/nnx/PaliGemma. This is a port, not a
direct call to that implementation. Velocities, time grids and reconstruction
come from the unmodified target policy. No optimizer or convergence gate is
added. The upstream BC trainer and its buffer filtering are not this module.
"""
from dataclasses import asdict, dataclass
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time


REFERENCE_DIRECTORY = '.deps/upstream/FlowDAgger'
REFERENCE_REVISION = '3426f246510fd67a9e39c3c08bd5b02d586ad688'
CONTROLLER_SOURCE = 'flowdagger_pi05/flow_matching_inverter.py'
ENTRY_SOURCE = 'flowdagger_pi05/train_flowdagger.py'


def reference_recipe(root):
    """Read the author's entry default from the pristine pinned checkout."""
    source = Path(root) / REFERENCE_DIRECTORY
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if revision != REFERENCE_REVISION or dirty:
        raise RuntimeError('FlowDAgger reference source must remain pinned and pristine.')
    defaults = {}
    for node in ast.walk(ast.parse((source / ENTRY_SOURCE).read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {'--inversion_method', '--fp_per_step'}):
            for keyword in node.keywords:
                if keyword.arg == 'default':
                    defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    if defaults.get('--inversion_method') != 'perstep_fp':
        raise RuntimeError('The pinned author entry no longer selects per-step fixed point.')
    return {
        'algorithm': 'flowdagger-pi05-perstep-fp.torch-port.v1',
        'source_directory': REFERENCE_DIRECTORY,
        'source_revision': revision,
        'controller': CONTROLLER_SOURCE + ':_build_perstep_fp_reverse_fn',
        'entry': ENTRY_SOURCE,
        'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in (CONTROLLER_SOURCE, ENTRY_SOURCE)},
        'fp_per_step': int(defaults['--fp_per_step']),
        'initialization': 'reverse Euler, velocity evaluated at the endpoint time',
        'corrections': 'fixed count, velocity evaluated at the source time',
        'time_grid': 'unaltered original target-policy time grid',
        'reconstruction': 'complete original target-policy forward; action MSE reported',
        'support_admission': 'all supplied finite labels; no added precision gate',
        'upstream_bc_buffer': 'not used by the frozen-policy PFN context method',
    }


@dataclass(frozen=True)
class FlowDAggerSettings:
    reference: dict

    @property
    def fp_per_step(self):
        return self.reference['fp_per_step']

    @property
    def recipe_id(self):
        fingerprint = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return self.reference['algorithm'] + ':' + fingerprint

    def __post_init__(self):
        if (self.reference.get('source_revision') != REFERENCE_REVISION
                or self.reference.get('algorithm') != 'flowdagger-pi05-perstep-fp.torch-port.v1'
                or self.fp_per_step < 1):
            raise ValueError('Require the explicit pinned FlowDAgger reference recipe.')


class ActionOnlyFlowDAggerSettings(FlowDAggerSettings):
    """Distinct label contract for our fixed-random-video boundary problem."""

    @property
    def recipe_id(self):
        return super().recipe_id + ':fixed-random-video-forward-backward.v1'


class FixedDepthActionOnlyFlowDAggerSettings(FlowDAggerSettings):
    """PrismWAM*: one-way attention permits the original fixed depth path."""

    @property
    def recipe_id(self):
        return super().recipe_id + ':prismwam-star-fixed-native-depth.v1'


def invert_native_action(*, target_action, replay, settings):
    """Published per-step fixed-point recurrence for an action-only sampler.

    Unlike UniWAM's coupled ablation this native policy has no generated video
    path. The backend supplies original velocities/grid and its original
    executable-coordinate mask; no optimizer, residual gate or extra iteration.
    """
    import torch
    if type(settings) is not FlowDAggerSettings:
        raise TypeError('Use the pinned published per-step fixed-point recipe.')
    started = time.perf_counter()
    target = target_action.detach().float()
    if target.shape != replay.initial.shape or not torch.isfinite(target).all():
        raise ValueError('Require the complete native model-space expert chunk.')
    state = target
    reports = []
    with torch.no_grad():
        for step in reversed(range(replay.steps)):
            endpoint = state
            delta = replay.step_delta(step)
            state = replay.project(endpoint - delta * replay.velocity(step, endpoint, time_index=step + 1).float())
            for _ in range(settings.fp_per_step):
                state = replay.project(endpoint - delta * replay.velocity(step, state, time_index=step).float())
            reports.append({'native_step': step, 'fp_corrections': settings.fp_per_step,
                            'velocity_evaluations': 1 + settings.fp_per_step})
            print(json.dumps({'flowdagger_reference_step': reports[-1]}), flush=True)
        noise = state.to(replay.initial)
        if not torch.isfinite(noise).all():
            raise RuntimeError('Nonfinite native fixed-budget inverse.')
        replay.release_condition()
        reconstructed = replay.verify_complete(noise)
        error = replay.executable(reconstructed.float() - target)
        if not torch.isfinite(error).all():
            raise RuntimeError('Nonfinite complete native reconstruction.')
        metrics = {'solver': settings.reference['algorithm'], 'reference': settings.reference,
            'native_action_only_sampler': True, 'per_step': reports,
            'velocity_evaluations': replay.steps * (1 + settings.fp_per_step),
            'native_reference_velocity_evaluations': replay.steps,
            'native_reconstruction_velocity_evaluations': replay.steps,
            'action_mse': float(error.square().mean()),
            'action_mse_per_chunk': error.square().mean(dim=(-2, -1)).cpu().tolist(),
            'action_max_error': float(error.abs().max()), 'finite_label': True,
            'elapsed_seconds': time.perf_counter() - started,
            'native_execution': replay.diagnostics()}
        print(json.dumps({'flowdagger_reference_reconstruction': metrics}), flush=True)
        return noise, metrics


def invert_native_joint(*, target_action, replay, settings):
    """Port only the author's reverse-Euler initialization and FP recurrence.

    A WAM needs a joint endpoint: retain its native predicted video, replace
    its action with the expert action. Each native step supplies its own dt;
    video/action schedules are never replaced with a new grid. Controller
    arithmetic uses float32; every network query and the final noise injection
    retain the native policy's casts. The final reconstruction measures those
    real finite-precision computations against the float32 expert target.
    """
    import torch
    from embodpfn.inversion import JointInverseResult, JointLatent, InversionFailure

    if not isinstance(settings, FlowDAggerSettings):
        raise TypeError('The reference route does not accept custom solver settings.')
    started = time.perf_counter()
    reference = replay.reference()
    video_slice = getattr(replay, 'predicted_video_slice', slice(1, None))
    target_action = target_action.detach().float()
    if target_action.shape != reference.action.shape or not torch.isfinite(target_action).all():
        raise ValueError('Require the complete finite native expert action chunk.')
    state = JointLatent(reference.video.float(), target_action)
    velocity_evaluations = 0
    reports = []

    with torch.no_grad():
        for step in reversed(range(replay.steps)):
            endpoint = state
            video_dt, action_dt = replay.step_deltas(step)

            def reverse_update(velocity):
                video = endpoint.video - video_dt * velocity.video.float()
                if hasattr(replay, 'project_reverse_video'):
                    # Backend-native boundary: PrismWAM* keeps the observed
                    # depth separate from all generated depth latents. Its
                    # action-only variant retains the native random depth path.
                    video = replay.project_reverse_video(video, step)
                else:
                    video[:, :, :1] = replay.initial.video[:, :, :1]
                return JointLatent(video, endpoint.action - action_dt * velocity.action.float())

            # Upstream v_init(x_prev, t_prev), then exactly fp_per_step
            # v_est(x_est, t_next) calls. No residual-based early termination.
            state = reverse_update(replay.velocity(step, endpoint, time_index=step + 1))
            velocity_evaluations += 1
            for _ in range(settings.fp_per_step):
                state = reverse_update(replay.velocity(step, state, time_index=step))
                velocity_evaluations += 1
            report = {'native_step': step, 'fp_corrections': settings.fp_per_step,
                      'velocity_evaluations': 1 + settings.fp_per_step}
            reports.append(report)
            print(json.dumps({'flowdagger_reference_step': report}), flush=True)

        noise = JointLatent(state.video.to(replay.initial.video.dtype),
                            state.action.to(replay.initial.action.dtype))
        if not (torch.isfinite(noise.video).all() and torch.isfinite(noise.action).all()):
            raise InversionFailure('FlowDAgger fixed-budget inverse produced nonfinite noise.')
        replay.release_prefix()
        reconstructed = replay.verify_complete(noise)
        error = reconstructed.action.float() - target_action
        video_error = (reconstructed.video[:, :, video_slice].float()
                       - reference.video[:, :, video_slice].float())
        if not (torch.isfinite(error).all() and torch.isfinite(video_error).all()):
            raise InversionFailure('Original forward reconstruction produced nonfinite output.')
        metrics = {
            'solver': settings.reference['algorithm'],
            'reference': settings.reference,
            'velocity_evaluations': velocity_evaluations,
            'native_reference_velocity_evaluations': replay.steps,
            'native_reconstruction_velocity_evaluations': replay.steps,
            'per_step': reports,
            'action_mse': float(error.square().mean()),
            'action_mse_per_chunk': error.square().mean(dim=(-2, -1)).cpu().tolist(),
            'action_max_error': float(error.abs().max()),
            'video_future_mse': float(video_error.square().mean()),
            'finite_label': True,
            'elapsed_seconds': time.perf_counter() - started,
            'native_execution': replay.diagnostics(),
        }
        print(json.dumps({'flowdagger_reference_reconstruction': metrics}), flush=True)
        return JointInverseResult(noise, metrics['action_max_error'], velocity_evaluations + 1,
            settings.reference['algorithm'], asdict(settings), metrics)


def invert_native_action_only(*, target_action, replay, settings):
    """Our constrained port: V0 stays random; only action states are reversed.

    UniWAM attention is bidirectional. Therefore fixing a video trajectory
    from the first rollout would solve the wrong equations. Each correction
    sweep first advances video from the same native random V0 using native
    Euler steps and the current action path, then corrects the action path
    backwards. At a fixed point both original coupled Euler equations hold,
    with V0 fixed and A_T equal to the expert. There is no video inverse.

    Preserve one reverse-Euler initialization and exactly the author's five
    corrections per action step. Reordering corrections into forward/backward
    sweeps is an explicit method extension, not the published joint solver.
    Final admission still uses the original full native forward and .001 MSE.
    """
    import torch
    from embodpfn.inversion import JointInverseResult, JointLatent, InversionFailure
    if not isinstance(settings, ActionOnlyFlowDAggerSettings):
        raise TypeError('Action-only labels require their distinct constrained-inverse contract.')
    started = time.perf_counter()
    reference = replay.reference()
    target = target_action.detach().float()
    if target.shape != reference.action.shape or not torch.isfinite(target).all():
        raise ValueError('Require the complete finite float32 expert action chunk.')
    actions = [None] * replay.steps + [target]
    action_queries = video_queries = 0
    reports = []
    with torch.no_grad():
        for step in reversed(range(replay.steps)):
            endpoint_video = (replay.states[step + 1].video
                              if step + 1 < replay.steps else reference.video)
            velocity = replay.velocity(step, JointLatent(endpoint_video, actions[step + 1]),
                                       time_index=step + 1)
            actions[step] = actions[step + 1] - replay.step_deltas(step)[1] * velocity.action.float()
            action_queries += 1
        for correction in range(settings.fp_per_step):
            videos = [replay.initial.video]
            for step in range(replay.steps):
                native = replay.states[step]
                result = replay.query(step, JointLatent(videos[-1].to(native.video.dtype),
                                                        actions[step].to(native.action.dtype)))
                videos.append(result.video.detach())
                video_queries += 1
            updated = [None] * replay.steps + [target]
            for step in reversed(range(replay.steps)):
                velocity = replay.velocity(step, JointLatent(videos[step], actions[step]), time_index=step)
                updated[step] = updated[step + 1] - replay.step_deltas(step)[1] * velocity.action.float()
                action_queries += 1
            actions = updated
            reports.append({'correction': correction + 1,
                'native_video_forward_steps': replay.steps, 'action_step_corrections': replay.steps})
            print(json.dumps({'action_only_forward_backward_correction': reports[-1]}), flush=True)
        noise = JointLatent(replay.initial.video.detach().clone(),
                            actions[0].to(replay.initial.action.dtype))
        if not torch.isfinite(noise.action).all():
            raise InversionFailure('Action-only fixed-budget inverse returned nonfinite action points.')
        replay.release_prefix()
        reconstructed = replay.verify_complete(noise)
        error = reconstructed.action.float() - target
        if not torch.isfinite(error).all():
            raise InversionFailure('Complete native action-only reconstruction returned nonfinite actions.')
        metrics = {'solver': 'flowdagger-action-only-constrained-forward-backward.torch-port.v1',
            'reference': settings.reference, 'method_extension': 'fixed random video initial condition; forward video/backward action corrections',
            'noise_target': 'action', 'video_initial_unchanged': torch.equal(noise.video, replay.initial.video),
            'fp_corrections_per_action_step': settings.fp_per_step,
            'velocity_evaluations': action_queries + video_queries,
            'action_velocity_evaluations': action_queries, 'native_video_forward_step_evaluations': video_queries,
            'native_reference_velocity_evaluations': replay.steps,
            'native_reconstruction_velocity_evaluations': replay.steps,
            'per_sweep': reports, 'action_mse': float(error.square().mean()),
            'action_mse_per_chunk': error.square().mean(dim=(-2, -1)).cpu().tolist(),
            'action_max_error': float(error.abs().max()),
            'video_future_mse': float((reconstructed.video[:, :, 1:].float() - reference.video[:, :, 1:].float()).square().mean()),
            'finite_label': True, 'elapsed_seconds': time.perf_counter() - started,
            'native_execution': replay.diagnostics()}
        print(json.dumps({'action_only_native_reconstruction': metrics}), flush=True)
        return JointInverseResult(noise, metrics['action_max_error'], action_queries + video_queries + 1,
            metrics['solver'], asdict(settings), metrics)

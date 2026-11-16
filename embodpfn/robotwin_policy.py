"""RoboTwin's native policy plugin; delegate the complete UniWAM policy.

Use the original policy/uniwam/deploy_policy.yml and the benchmark's original
script/eval_policy.py. Only policy_name selects this method and embodpfn_*
arguments configure its explicit support/PFN paths. No YAML copy is created.
"""
import atexit
import hashlib
import json
import random
import traceback
from pathlib import Path

from embodpfn.adaptation import ContextInferenceFailure
from embodpfn.pipeline import NoiseContextSession
from embodpfn.runtime import attach_context_adapter
from embodpfn.upstream import require_declared_checkpoint


def _build_context(policy, session, bundle, output):
    import numpy as np
    import torch
    from uniwam import reset_model

    bundle = Path(bundle)
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    native_ids = manifest['native_dataset_episode_ids']
    coverage = manifest.get('context_coverage', {})
    expected_pairs = [(episode['episode'], frame) for episode in coverage.get('episodes', [])
                      for frame in episode['decision_frames']]
    if manifest.get('history_contract') != 'native-history-indices-and-raw-qpos.v1':
        raise ValueError('Expert context must include the original measured qpos history.')
    if (manifest['split'] != 'clean50'
            or coverage.get('version') != 'all-native-robotwin-replanning-windows.v1'
            or len(manifest['rows']) != coverage.get('expected_rows')
            or len(native_ids) != len(set(native_ids))
            or [episode['episode'] for episode in coverage['episodes']] != native_ids
            or [(row['episode'], row['condition_frame']) for row in manifest['rows']] != expected_pairs):
        raise ValueError('Expected every native decision window in every accepted clean50 episode.')
    session.adapter.support_context = {'coverage': coverage, 'task': manifest['task'],
        'expert_manifest_sha256': hashlib.sha256((bundle / 'MANIFEST.json').read_bytes()).hexdigest()}
    for index, row in enumerate(manifest['rows']):
        reset_model(policy)
        with np.load(bundle / row['file'], allow_pickle=False) as sample:
            observation = {'image': torch.from_numpy(sample['image']),
                           'joint_action': {'vector': sample['state'].copy()}}
            actions = torch.from_numpy(sample['actions'].copy()).unsqueeze(0)
            history = sample['history'].copy()
        if history.shape != (policy.model.config.action_chunk_size, policy.model.config.action_dim):
            raise ValueError('Preserve the complete original action-source history.')
        for qpos in history:
            policy.record_executed_qpos({'joint_action': {'vector': qpos}})
        policy.update_obs(observation)
        policy.set_instruction(row['instruction'])
        correction = session.add_expert_call(policy.get_action, actions)
        print(json.dumps({'context_episode': row['episode'], 'completed': index + 1,
                          'total': len(manifest['rows']),
                          'inverse_reconstruction_max_error': correction.reconstruction_error,
                          'reconstruction_metrics': correction.reconstruction_metrics}), flush=True)
    session.adapter.save_support(output)
    reset_model(policy)


def get_model(usr_args):
    from uniwam import get_model as official_get_model

    require_declared_checkpoint('uniwam', usr_args['ckpt_setting'], usr_args.get('embodpfn_policy_receipt'))

    # Everything, including inference mode/steps and normalization, is read by
    # the original loader from the original upstream arguments/configuration.
    policy = official_get_model(usr_args)
    from embodpfn.flowdagger import FlowDAggerSettings, ActionOnlyFlowDAggerSettings, reference_recipe
    from embodpfn.upstream_context_rule import context_rule
    support = Path(usr_args['embodpfn_support'])
    expected = json.loads(Path(usr_args['embodpfn_inverse_reference']).read_text())
    actual = reference_recipe(Path(__file__).resolve().parents[1])
    from embodpfn.official_recipe import robotwin_recipe
    import uniwam
    recipe = robotwin_recipe(Path(uniwam.__file__).parent)
    if any(usr_args.get(name) != value for name, value in recipe['arguments'].items()):
        raise RuntimeError('Invoke the complete upstream main recipe; internal legacy defaults are not a substitute.')
    actual['target_native_recipe'] = recipe
    profile = Path(policy.checkpoint_path).parent / 'config.json'
    actual['target_native_profile_sha256'] = hashlib.sha256(profile.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError('Staged FlowDAgger reference differs from the source used at runtime.')
    noise_target = usr_args.get('embodpfn_noise_target', 'joint')
    if noise_target not in ('joint', 'action'):
        raise ValueError('Declare joint video/action or action-only noise prediction.')
    settings = (ActionOnlyFlowDAggerSettings if noise_target == 'action' else FlowDAggerSettings)(actual)
    head_parent = usr_args.get('embodpfn_head_parent_receipt')
    if head_parent and (noise_target != 'action' or not usr_args.get('embodpfn_head_feature_manifest') or not usr_args.get('embodpfn_head_query_trace')):
        raise ValueError('Bind complete head-readout provenance/trace for original action-epsilon evaluation.')
    runtime = ('uniwam-robotwin-action-head' if head_parent else
        'uniwam-robotwin-action-only' if noise_target == 'action' else 'uniwam-robotwin')
    session = NoiseContextSession(
        policy.model, runtime=runtime, inversion_settings=settings,
        pfn_python=usr_args['embodpfn_pfn_python'],
        pfn_checkpoint=usr_args['embodpfn_pfn_checkpoint'],
        pfn_device=usr_args['embodpfn_pfn_device'],
        pfn_devices=usr_args.get('embodpfn_pfn_devices'),
        checkpoint_receipt=usr_args.get('embodpfn_policy_receipt'),
        context_rule=context_rule(Path(__file__).resolve().parents[1]),
        pfn_kv_cache={'none': False, 'kv': True, 'repr': 'repr'}[
            usr_args.get('embodpfn_pfn_cache', 'none')],
        pfn_execution=usr_args.get('embodpfn_pfn_execution', 'native'))
    if head_parent:
        session.adapter.backend.head_trace = Path(usr_args['embodpfn_head_query_trace'])
    attachment = None
    prepared = False

    def prepare_context():
        nonlocal attachment, prepared
        import numpy as np
        if prepared:
            return
        if head_parent:
            import torch
            from embodpfn.action_head_context import prepare_feature_context
            device = next(policy.model.parameters()).device
            with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
                prepare_feature_context(policy, session, usr_args['embodpfn_expert_bundle'], head_parent,
                    support, usr_args['embodpfn_head_feature_manifest'])
        elif support.exists():
            with np.load(support, allow_pickle=False) as artifact:
                recorded = json.loads(str(artifact['manifest'].item()))
            if recorded['inversion_recipe_id'] != settings.recipe_id:
                raise ValueError('Do not reuse labels from a retired solver as reference labels.')
            support_context = recorded.get('context_coverage')
            if (support_context is None or support_context['coverage'].get('version') !=
                    'all-native-robotwin-replanning-windows.v1'
                    or recorded['support_rows'] != support_context['coverage']['expected_rows']):
                raise ValueError('A sampled single-chunk support set is not a full-trajectory context.')
            session.adapter.support_context = support_context
            session.load_support(support)
        else:
            bundle = usr_args['embodpfn_expert_bundle']
            manifest = json.loads((Path(bundle) / 'MANIFEST.json').read_text())
            if manifest['task'] != usr_args['task_name']:
                raise ValueError('Expert context belongs to another official task.')
            import torch
            device = next(policy.model.parameters()).device
            # Offline context construction must not consume the unchanged
            # evaluator's random instruction stream or simulator RNG state.
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
                    _build_context(policy, session, bundle, support)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
            # Our offline inverse temporarily retains reference activations.
            # Return their now-unused allocator blocks before the separate
            # official PFN process needs device memory. Live tensors stay put.
            before = torch.cuda.memory_reserved(device)
            torch.cuda.empty_cache()
            print(json.dumps({'offline_inverse_allocator_release': True,
                'reserved_bytes_before': before,
                'reserved_bytes_after': torch.cuda.memory_reserved(device)}), flush=True)
        session.adapter.prepare_context()
        attachment = attach_context_adapter(session.adapter)
        attachment.__enter__()
        prepared = True

    def close():
        try:
            if attachment is not None:
                attachment.__exit__(None, None, None)
        finally:
            session.close()

    atexit.register(close)
    policy._embodpfn_context_session = session
    # Let the unchanged evaluator initialize its real scene first. A native
    # renderer failure must not consume hours of our offline inverse work.
    # The original eval sets instruction/observation after this callback.
    policy._embodpfn_prepare_context = prepare_context
    return policy


def _eval_with_native_action_evidence(official_eval, TASK_ENV, model, observation):
    """Observe existing CPU returns/feedback; delegate the original eval once."""
    from embodpfn.backends.uniwam import _observe_method
    backend = model._embodpfn_context_session.adapter.backend
    head_trace = getattr(backend, 'head_trace', None)
    if head_trace is None:
        return official_eval(TASK_ENV, model, observation)
    destination = head_trace.with_name(head_trace.stem + '-native-actions.jsonl')
    try:
        evidence = {
            'kind': 'original-native-CPU-action-return-and-execution-feedback.v1',
            'official_trial': TASK_ENV.test_num + 1,
            'native_episode_index': TASK_ENV.ep_num,
            'native_action_step_before': TASK_ENV.take_action_cnt,
            'query_before': backend.query_count,
            'existing_observation_state': observation['joint_action']['vector'].tolist(),
            'executed_feedback': [],
        }
    except Exception as error:
        raise ContextInferenceFailure('Native CPU action evidence could not be bound.') from error

    def observe_actions(original, args, kwargs):
        returned = original(*args, **kwargs)
        # UniWAMPolicy.get_action already made actions_real on CPU with NumPy.
        # Keep its object/bytes/order; never rerun or fetch another model output.
        try:
            evidence.update(query=backend.query_count,
                native_returned_actions=returned.tolist(),
                native_returned_actions_shape=list(returned.shape),
                native_returned_actions_dtype=str(returned.dtype),
                instruction=model.current_instruction)
        except Exception as error:
            raise ContextInferenceFailure('Native CPU action return could not be preserved.') from error
        return returned

    def observe_feedback(original, args, kwargs):
        result = original(*args, **kwargs)
        # This same observation was already acquired and recorded by native eval.
        # The gripper fields are the author's bookkeeping values, not contact force.
        try:
            feedback = args[0] if args else kwargs['observation']
            evidence['executed_feedback'].append({
                'query': backend.query_count,
                'native_action_step_after': TASK_ENV.take_action_cnt,
                'existing_joint_action_vector': feedback['joint_action']['vector'].tolist(),
            })
        except Exception as error:
            raise ContextInferenceFailure('Native existing feedback could not be preserved.') from error
        return result

    with _observe_method(model, 'get_action', observe_actions):
        with _observe_method(model, 'record_executed_qpos', observe_feedback):
            result = official_eval(TASK_ENV, model, observation)
    try:
        evidence.update(native_action_step_after=TASK_ENV.take_action_cnt,
            native_eval_success=bool(TASK_ENV.eval_success),
            executed_prefix_length=len(evidence['executed_feedback']))
        with destination.open('a') as stream:
            stream.write(json.dumps(evidence) + '\n')
    except Exception as error:
        raise ContextInferenceFailure('Complete native CPU action evidence could not be written.') from error
    return result


def eval(TASK_ENV, model, observation):
    from uniwam import eval as official_eval
    try:
        model._embodpfn_prepare_context()
    except Exception as error:
        traceback.print_exc()
        raise SystemExit('Context preparation failed; this is not a benchmark result.') from error
    try:
        return _eval_with_native_action_evidence(official_eval, TASK_ENV, model, observation)
    except ContextInferenceFailure as error:
        # The native evaluator catches ordinary episode exceptions. A broken
        # method integration must terminate the run instead of producing a
        # misleading full-run score. Simulator/controller errors stay native.
        traceback.print_exc()
        raise SystemExit('PFN integration failed; this is not a benchmark result.') from error


def reset_model(model):
    from uniwam import reset_model as official_reset
    return official_reset(model)

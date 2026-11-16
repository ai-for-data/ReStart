"""Feature-only replay of complete, source-bound original inverse labels."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import random

import numpy as np

from embodpfn.backends.uniwam import _observe_method
from embodpfn.backends.uniwam_action_head import READOUT_ID, READOUT_FEATURES, array_sha
from embodpfn.interfaces import InvertedCorrection, PolicyCondition
from embodpfn.native_history_condition import HISTORY_READOUT_ID

LABEL_SOURCE_FILES = {'embodpfn/backends/uniwam.py', 'embodpfn/backends/native_steps.py',
    'embodpfn/backends/condition_cache.py', 'embodpfn/flowdagger.py', 'embodpfn/interfaces.py',
    'embodpfn/adaptation.py', 'embodpfn/native_history_condition.py', 'embodpfn/noise_basis.py',
    'embodpfn/robotwin_experts.py', 'embodpfn/official_recipe.py', 'embodpfn/upstream.py',
    'embodpfn/upstream_context_rule.py'}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parent_arrays(receipt_path, expert_manifest, backend, context_rule):
    receipt = json.loads(Path(receipt_path).read_text())
    parent = Path(receipt['local_support'])
    if ((parent.stat().st_size, checksum(parent)) != (receipt['artifact']['bytes'], receipt['artifact']['sha256'])
            or checksum(expert_manifest) != receipt['expert_manifest']['sha256']):
        raise ValueError('Reuse only the exact complete original label artifact and original expert input chain.')
    root = Path(__file__).resolve().parents[1]
    if set(receipt['required_unchanged_source_files']) != LABEL_SOURCE_FILES:
        raise ValueError('Bind every unchanged label-controller/codec/data source explicitly.')
    for name, expected in receipt['required_unchanged_source_files'].items():
        if checksum(root / name) != expected:
            raise ValueError('The label solver/processor/source changed: ' + name)
    with np.load(parent, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest'].item()))
        values = {name: archive[name].copy() for name in ('conditions', 'noise', 'native_inverse_initial_points')}
    experts = json.loads(Path(expert_manifest).read_text())
    count = len(experts['rows'])
    if (receipt['kind'] != 'complete-native-action-label-source-for-head-readout.v1'
            or experts['task'] != receipt['task'] or experts['split'] != 'clean50'
            or manifest['readout_id'] != HISTORY_READOUT_ID
            or manifest['checkpoint_sha256'] != backend.asset['weight_sha256']
            or manifest['target_codec'] != backend.target_codec
            or manifest['inversion_recipe_id'] != backend.settings.recipe_id
            or manifest['pfn_context_admission'] != context_rule
            or manifest['support_rows'] != count or count != receipt['raw_rows']
            or values['conditions'].shape != (count, 750) or values['noise'].shape != (count, 224)
            or values['native_inverse_initial_points'].shape != (count, 46304)):
        raise ValueError('Require all current native-history4-step fixed5 raw labels; no sampled or alternate source.')
    coverage = manifest['context_coverage']
    if (coverage != {'coverage': experts['context_coverage'], 'task': experts['task'],
            'expert_manifest_sha256': checksum(expert_manifest)}
            or coverage['coverage'].get('native_tail_contexts') is not True
            or coverage['coverage']['expected_rows'] != count
            or len(experts['native_dataset_episode_ids']) != 50):
        raise ValueError('Preserve every original native replan, including terminal windows, in all50 experts.')
    pairs = [(episode['episode'], frame) for episode in coverage['coverage']['episodes']
             for frame in episode['decision_frames']]
    if [(row['episode'], row['condition_frame']) for row in experts['rows']] != pairs:
        raise ValueError('The original expert rows are missing or reordered.')
    metrics = manifest['reconstruction_metrics']
    if (len(metrics) != count or any(row['reference'] != backend.settings.reference
            or row.get('noise_target') != 'action' or row.get('finite_label') is not True
            or row.get('video_initial_unchanged') is not True or len(row['action_mse_per_chunk']) != 1
            or row.get('fp_corrections_per_action_step') != 5
            or row.get('native_reference_velocity_evaluations') != 4
            or row.get('native_reconstruction_velocity_evaluations') != 4
            or row.get('epsilon_reconstruction_velocity_evaluations') != 4
            for row in metrics)):
        raise ValueError('Keep every original finite native reconstruction diagnostic and its unchanged solver reference.')
    H = np.asarray([row for item in metrics for row in item['native_action_source']], dtype=np.float32)
    if (H.shape != (count, 16, 14) or not np.array_equal(H.reshape(count, 224), values['conditions'][:, -224:])
            or not all(np.isfinite(value).all() for value in values.values())):
        raise ValueError('The archived complete H and all original raw coordinates must remain exact and finite.')
    return receipt, experts, manifest, values


def prepare_feature_context(policy, session, bundle, receipt_path, output, feature_manifest):
    """Only called from the complete original evaluation lifecycle."""
    import torch
    from uniwam import reset_model
    bundle, output, feature_manifest = Path(bundle), Path(output), Path(feature_manifest)
    adapter, backend = session.adapter, session.adapter.backend
    if adapter.corrections or getattr(backend.model, backend.entry_name) != backend.official_entry:
        raise RuntimeError('Prepare one empty new-readout context before any online sampler attachment.')
    receipt, experts, original, arrays = parent_arrays(receipt_path, bundle / 'MANIFEST.json', backend, adapter.context_rule)
    binding = {'kind': 'native-H-anchor-action-head-feature-derivation.v1',
        'parent_record': receipt['parent_record'], 'parent_support': receipt['artifact'],
        'parent_expert_manifest': receipt['expert_manifest'], 'parent_expert_archive': receipt['expert_input_archive'],
        'label_source_method_sha256': receipt['label_source_method_sha256'],
        'readout_id': READOUT_ID, 'features': READOUT_FEATURES,
        'head_source_sha256': checksum(Path(__file__).with_name('backends') / 'uniwam_action_head.py'),
        'feature_builder_sha256': checksum(__file__),
        'context_coverage': original['context_coverage'],
        'epsilon_or_inverse_diagnostics_changed': False, 'H_constructed_or_normalized': False}
    if output.exists() or feature_manifest.exists():
        if not output.is_file() or not feature_manifest.is_file():
            raise FileExistsError('Preserve an incomplete feature replay; never silently restart or reinvert it.')
        proof = json.loads(feature_manifest.read_text())
        if proof.get('status') != 'complete' or proof.get('binding') != binding or proof.get('support_sha256') != checksum(output):
            raise ValueError('Only a complete same-source/same-label feature artifact may be reused.')
        validate_derived_context(output, Path(receipt['local_support']), proof, binding)
        adapter.support_context = original['context_coverage']
        session.load_support(output)
        return proof
    proof = {'status': 'in-progress', 'binding': binding, 'rows': [], 'raw_rows': len(experts['rows']),
        'label_inversion_run': False, 'standalone_feature_job': False}
    feature_manifest.write_text(json.dumps(proof, indent=2) + '\n')
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        for index, row in enumerate(experts['rows']):
            reset_model(policy)
            source = bundle / row['file']
            with np.load(source, allow_pickle=False) as sample:
                observation = {'image': torch.from_numpy(sample['image']),
                    'joint_action': {'vector': sample['state'].copy()}}
                history = sample['history'].copy()
            if history.shape != (16, 14):
                raise ValueError('Keep the original complete native expert history.')
            for qpos in history:
                policy.record_executed_qpos({'joint_action': {'vector': qpos}})
            policy.update_obs(observation)
            policy.set_instruction(row['instruction'])
            captured = None

            class FeatureCaptured(Exception):
                pass

            def collect(original_entry, args, kwargs):
                nonlocal captured
                inputs = dict(inspect.signature(original_entry).bind(*args, **kwargs).arguments)
                captured = backend.extract_h_anchor(inputs, arrays['native_inverse_initial_points'][index:index + 1, :46080])
                raise FeatureCaptured()

            with _observe_method(backend.model, backend.entry_name, collect):
                try:
                    policy.get_action()
                except FeatureCaptured:
                    pass
                else:
                    raise RuntimeError('The original expert call did not reach exactly one declared head readout.')
            if (captured is None or captured['features'].shape != (1, READOUT_FEATURES)
                    or not np.array_equal(captured['features'][:, 1024:1038], arrays['conditions'][index:index + 1, 512:526])
                    or not np.array_equal(captured['features'][:, -224:], arrays['conditions'][index:index + 1, -224:])):
                raise ValueError('The new readout must retain that exact original expert state and complete cast H.')
            adapter.corrections.append(InvertedCorrection(
                PolicyCondition(None, captured['features'], READOUT_ID, original['checkpoint_sha256']),
                arrays['noise'][index:index + 1].copy(), original['reconstruction_max_errors'][index],
                original['inversion_recipe_id'], deepcopy(original['reconstruction_metrics'][index]),
                arrays['native_inverse_initial_points'][index:index + 1].copy()))
            proof['rows'].append(dict(captured['proof'], support_index=index, expert_episode=row['episode'],
                condition_frame=row['condition_frame'], original_expert_file_sha256=checksum(source)))
            feature_manifest.write_text(json.dumps(proof, indent=2) + '\n')
            print(json.dumps({'head_feature_replay_completed': index + 1, 'total': len(experts['rows']),
                'label_inversion_run': False, 'features': READOUT_FEATURES}), flush=True)
        adapter.support_context = original['context_coverage']
        intermediate = output.with_name(output.stem + '-unbound-derived.npz')
        adapter.save_support(intermediate)
        with np.load(intermediate, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        derived = json.loads(str(values['manifest'].item()))
        derived['condition_derivation'] = binding
        values['manifest'] = np.asarray(json.dumps(derived))
        temporary = output.with_name(output.stem + '-publishing.npz')
        np.savez(temporary, **values)
        proof.update(status='complete', support_sha256=checksum(temporary))
        validate_derived_context(temporary, Path(receipt['local_support']), proof, binding)
        temporary.replace(output)
        intermediate.unlink()
        feature_manifest.write_text(json.dumps(proof, indent=2) + '\n')
    except BaseException:
        proof['status'] = 'failed; no fallback inversion or partial-context admission'
        feature_manifest.write_text(json.dumps(proof, indent=2) + '\n')
        raise
    finally:
        reset_model(policy)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return proof


def validate_derived_context(derived_path, parent_path, proof, binding):
    with np.load(derived_path, allow_pickle=False) as new, np.load(parent_path, allow_pickle=False) as old:
        original = json.loads(str(old['manifest'].item()))
        updated = json.loads(str(new['manifest'].item()))
        if (updated.pop('condition_derivation') != binding or updated.pop('readout_id') != READOUT_ID
                or new['conditions'].shape != (len(old['conditions']), READOUT_FEATURES)
                or not np.isfinite(new['conditions']).all()
                or not np.array_equal(new['conditions'][:, -224:], old['conditions'][:, -224:])
                or not np.array_equal(new['conditions'][:, 1024:1038], old['conditions'][:, 512:526])
                or any(not np.array_equal(new[name], old[name]) for name in ('noise', 'native_inverse_initial_points'))):
            raise ValueError('A feature-only artifact changed labels/points/state/H or its declared native readout.')
        original.pop('readout_id')
        if updated != original or len(proof['rows']) != len(new['conditions']):
            raise ValueError('All original diagnostics/admission/source coverage must remain unchanged.')
        for index, row in enumerate(proof['rows']):
            if (row['support_index'] != index or row['CPU_CUDA_rng_restored'] is not True
                    or row['actual_auxiliary_action_initial_equals_H'] is not True
                    or row['readout_id'] != READOUT_ID or row['condition_features'] != READOUT_FEATURES
                    or row['native_head_input_shape'] != [1, 21, 1024] or row['native_action_positions'] != [1, 17]
                    or row['auxiliary_first_head_prefixes'] != 1 or row['anchor_epsilon'] != 0 or row['first_native_time'] != 1.0
                    or row['H_sha256'] != array_sha(new['conditions'][index:index + 1, -224:])
                    or row['future_V0_sha256'] != array_sha(old['native_inverse_initial_points'][index:index + 1, :46080])):
                raise ValueError('Each actual feature row must bind its exact original V0/H and restored RNG.')


def validate_head_trace(path):
    """Read real completed-entry evidence; never run a model or synthetic query."""
    import torch
    started = completed = 0
    pending = None
    total_auxiliary = total_callback = total_main = 0.0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'native-H-anchor-action-head-query.v1':
                raise ValueError('The query trace belongs to another method boundary.')
            if row['event'] == 'query-start':
                features = np.asarray(row['features'], dtype=np.float32)
                if (pending is not None or row['query'] != started + 1 or features.shape != (1, READOUT_FEATURES)
                        or row['readout_id'] != READOUT_ID or row['condition_features'] != READOUT_FEATURES
                        or row['native_head_input_shape'] != [1, 21, 1024] or row['native_action_positions'] != [1, 17]
                        or not np.isfinite(features).all() or row['CPU_CUDA_rng_restored'] is not True
                        or row['auxiliary_first_head_prefixes'] != 1 or row['anchor_epsilon'] != 0
                        or row['actual_auxiliary_action_initial_equals_H'] is not True
                        or row['first_native_time'] != 1.0 or row['H_sha256'] != array_sha(features[:, -224:])):
                    raise ValueError('Require one real H-anchor first-head readout with complete unchanged H.')
                started += 1
                pending = row
            elif row['event'] == 'query-completed':
                predicted = np.asarray(row['predicted_epsilon'], dtype=np.float32)
                applied = np.asarray(row['applied_epsilon'], dtype=np.float32)
                original = np.asarray(row['original_random_epsilon'], dtype=np.float32)
                if (pending is None or row['query'] != pending['query']
                        or any(value.shape != (1, 224) or not np.isfinite(value).all() for value in (predicted, applied, original))
                        or row['condition_features'] != READOUT_FEATURES or row['main_native_steps'] != 4
                        or row['auxiliary_main_V0_equal'] is not True or row['auxiliary_main_H_equal'] is not True
                        or row['main_future_V0_sha256'] != pending['future_V0_sha256']
                        or row['main_full_V0_sha256'] != pending['full_V0_sha256']
                        or row['main_H_sha256'] != pending['H_sha256']
                        or row['PFN_callback_preserved_CPU_CUDA_RNG'] is not True
                        or row['CPU_CUDA_rng_restored'] is not True
                        or row['original_history_and_random_factories_retained'] is not True
                        or row['applied_native_dtype'] != 'torch.bfloat16'):
                    raise ValueError('Only the real same-V0/H original4-step call and complete epsilon may finish a query.')
                # Public Torch cast on actual archived predictions, not a
                # custom rounding implementation or a numerical tolerance.
                expected = torch.from_numpy(predicted).to(torch.bfloat16).float().numpy()
                if not np.array_equal(applied, expected):
                    raise ValueError('The actual injected epsilon differs from the original native dtype cast.')
                durations = [row[key] for key in ('auxiliary_wall_seconds', 'PFN_callback_wall_seconds', 'main_native_wall_seconds')]
                if any(not np.isfinite(value) or value < 0 for value in durations):
                    raise ValueError('Keep finite separately named auxiliary/callback/native timing scopes.')
                total_auxiliary += durations[0]
                total_callback += durations[1]
                total_main += durations[2]
                completed += 1
                pending = None
            else:
                raise ValueError('Unexpected query trace event.')
    if not completed or pending is not None or started != completed:
        raise ValueError('A running or incomplete query stream is not full-entry evidence.')
    return {'queries': completed, 'features': READOUT_FEATURES, 'epsilon_dimensions': 224,
        'auxiliary_original_prefixes': completed, 'main_original_steps_per_query': 4,
        'auxiliary_wall_seconds_total': total_auxiliary, 'PFN_callback_wall_seconds_total': total_callback,
        'main_native_wall_seconds_total': total_main, 'all_real_V0_H_RNG_and_native_cast_checks_passed': True}

"""Prepare our context rows with UniWAM's original data helpers.

Run in a separate process with the official UniWAM repository as cwd. This
keeps its root data/utils imports out of the separately deployed RoboTwin
policy's models/utils namespace. No upstream file or configuration is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--converter-config', type=Path, required=True)
    parser.add_argument('--vlm-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prepare-raw', action='store_true')
    parser.add_argument('--wan-path')
    args = parser.parse_args()

    import numpy as np
    import yaml
    from accelerate.utils import set_seed
    from data.robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
    from data.robotwin2.robotwin_data_convert.robotwin_converter import RobotWinConverter, T5EmbeddingProcessor
    from data.utils.image_utils import get_video_frame_count, load_video_frames

    set_seed(args.seed)
    config = yaml.safe_load(args.policy_config.read_text())
    common = config['common']
    converter = RobotWinConverter(str(args.converter_config))
    conversion_results = []
    if args.prepare_raw:
        # Compose the native per-episode conversion/cache APIs on this device.
        # The bulk converter's eight-device deployment config stays unchanged.
        structure = converter.scan_dataset(converter.config['source_root'])
        episodes = sorted(structure['clean'][args.task])
        if len(episodes) != 50:
            raise RuntimeError('The raw official clean50 archive must contain all 50 episodes.')
        target = args.dataset / 'clean' / args.task
        encoder = T5EmbeddingProcessor(args.wan_path, device='cuda:0')
        for index, episode in enumerate(episodes):
            episode = Path(episode)
            instruction = episode.parent.parent / 'instructions' / (episode.stem + '.json')
            if not instruction.is_file():
                raise FileNotFoundError(instruction)
            converted = converter.process_episode(str(episode), str(instruction), target, index)
            encoded = False
            if converted:
                encoded = encoder.process_meta_file(str(target / 'metas' / f'{index}.txt'),
                                                     str(target / 'umt5_wan' / f'{index}.pt'))
            # Preserve the native converter's accepted/rejected episode rule.
            conversion_results.append({'source_episode': str(episode), 'episode_id': str(index),
                                       'native_conversion_accepted': bool(converted),
                                       'native_t5_encoding_accepted': bool(encoded)})
    dataset = RobotWinTaskDataset(
        dataset_dir=str(args.dataset), data_mode='clean', task_mode='single', task_name=args.task,
        global_downsample_rate=common['global_downsample_rate'],
        video_action_freq_ratio=common['video_action_freq_ratio'],
        num_video_frames=common['num_video_frames'],
        video_size=(common['video_height'], common['video_width']),
        vlm_checkpoint_path=args.vlm_path)
    episodes = sorted(dataset.episode_files, key=lambda row: row['episode_name'])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'task': args.task, 'split': 'clean50', 'seed': args.seed,
                'selection': 'every native action-chunk execution decision in every native-valid expert episode',
                'raw_episode_count': len(conversion_results) if args.prepare_raw else None,
                'native_conversion_results': conversion_results,
                'native_dataset_episode_count': len(episodes),
                'native_dataset_episode_ids': [episode['episode_name'] for episode in episodes],
                'observation_and_action_loading': 'unmodified UniWAM data helpers',
                'policy_config_sha256': hashlib.sha256(args.policy_config.read_bytes()).hexdigest(),
                'history_contract': 'native-history-indices-and-raw-qpos.v1',
                'rows': []}
    coverage = {'version': 'all-native-robotwin-replanning-windows.v1',
        'native_action_horizon': dataset.action_chunk_size,
        'native_downsample_rate': dataset.global_downsample_rate,
        'condition_frame_stride': dataset.action_chunk_size * dataset.global_downsample_rate,
        'index_source': 'unmodified RobotWinTaskDataset._calculate_sampling_indices',
        'native_tail_contexts': True,
        'native_tail_index_source': 'original short-episode sampling on the remaining suffix, shifted to absolute episode indices',
        'random_subsampling': False, 'episodes': [], 'expected_rows': 0}
    manifest['context_coverage'] = coverage
    for episode in episodes:
        frames = get_video_frame_count(episode['video_path'])
        if frames < 2:
            raise RuntimeError('Invalid expert episode; never silently omit a demonstration.')
        span = coverage['condition_frame_stride']
        # Every original replan before the final observation is a condition,
        # including the final incomplete chunk. Keep the original sampler
        # for full windows. Its short-episode branch owns terminal clipping;
        # do not construct padded targets or enlarge its randint interval.
        wanted = set(range(0, frames - 1, span))
        full_windows = {condition for condition in wanted if condition + span < frames}
        native_indices = {}
        while len(native_indices) != len(full_windows):
            condition, _, action_indices = dataset._calculate_sampling_indices(frames)
            if condition in full_windows:
                native_indices[condition] = action_indices
        for condition in sorted(wanted - full_windows):
            remaining = frames - condition
            if not 1 < remaining <= span:
                raise ValueError('Only the original final short suffix may use native terminal handling.')
            local_condition, _, local_actions = dataset._calculate_sampling_indices(remaining)
            if local_condition != 0:
                raise RuntimeError('The original short-episode sampler must start at its first observation.')
            native_indices[condition] = [condition + index for index in local_actions]
        decisions = sorted(native_indices)
        coverage['episodes'].append({'episode': episode['episode_name'],
            'frames': frames, 'decision_frames': decisions,
            'native_tail_decision_frames': sorted(wanted - full_windows)})
        coverage['expected_rows'] += len(decisions)
        for condition in decisions:
            action_indices = native_indices[condition]
            image = load_video_frames(episode['video_path'], [condition], dataset.video_size)
            history_indices = dataset._calculate_history_action_indices(condition)
            state, actions, history = dataset._load_robot_data(
                episode['qpos_path'], action_indices, condition, history_indices)
            _, instruction_index = dataset._load_language_embedding(episode['lang_path'])
            text = dataset._load_text_instruction(args.task, episode['episode_name'], instruction_index, split='clean')
            if not text.startswith(converter.meta_prefix):
                raise RuntimeError('Expected the original converter scene prefix in expert metadata.')
            instruction = text[len(converter.meta_prefix):]
            if not instruction:
                raise RuntimeError('An expert instruction must be nonempty.')
            filename = f"{len(manifest['rows']):06d}.npz"
            np.savez(args.output / filename, image=image.numpy(), state=state.numpy(),
                     actions=actions.numpy(), history=history.numpy())
            manifest['rows'].append({'file': filename, 'episode': episode['episode_name'],
                'condition_frame': condition, 'episode_length': frames, 'action_indices': action_indices,
                'history_indices': history_indices, 'instruction_index': instruction_index,
                'instruction': instruction})
    expected_pairs = [(episode['episode'], frame) for episode in coverage['episodes']
                      for frame in episode['decision_frames']]
    if [(row['episode'], row['condition_frame']) for row in manifest['rows']] != expected_pairs:
        raise ValueError('Every original native decision point must occur exactly once.')
    (args.output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

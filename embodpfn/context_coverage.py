"""Our exhaustive context selection; native action/observation codecs stay upstream."""

VERSION = 'all-native-replanning-action-windows.v1'
BANK_VERSION = 'all-native-pretrain300-replanning-bank.v1'


def complete_window_coverage(episode_ids, episode_lengths, horizon, *, condition_stride,
                             final_observation_required=False, action_start_offset=0,
                             native_terminal_padding=False):
    ids, lengths = list(episode_ids), [int(n) for n in episode_lengths]
    horizon = int(horizon)
    condition_stride = int(condition_stride)
    action_start_offset = int(action_start_offset)
    minimum_length = (1 if native_terminal_padding else horizon) + action_start_offset + int(final_observation_required)
    if (not isinstance(native_terminal_padding, bool)
            or not ids or len(ids) != len(set(ids)) or len(ids) != len(lengths)
            or horizon <= 0 or condition_stride <= 0 or action_start_offset < 0
            or any(n < minimum_length for n in lengths)):
        raise ValueError('Every original trajectory requires a valid native action target.')
    def stop(length):
        return (length - action_start_offset - int(final_observation_required)
                if native_terminal_padding else
                length - action_start_offset - horizon + 1 - int(final_observation_required))
    coverage = {'version': VERSION, 'episode_ids': ids, 'episode_lengths': lengths,
        'native_action_horizon': horizon, 'condition_frame_stride': condition_stride,
        'future_action_start_offset': action_start_offset,
        'boundary': ('all native replanning conditions with at least one real target; terminal values remain original-reader outputs'
                     if native_terminal_padding else 'all complete native future windows; no invented terminal labels'),
        'expected_rows': sum(len(range(0, stop(n), condition_stride)) for n in lengths),
        'random_subsampling': False}
    if native_terminal_padding:
        coverage['native_terminal_padding'] = True
    if final_observation_required:
        coverage['final_observation_required'] = True
    return coverage


def all_complete_windows(coverage):
    expected = complete_window_coverage(coverage['episode_ids'], coverage['episode_lengths'],
        coverage['native_action_horizon'], condition_stride=coverage['condition_frame_stride'],
        final_observation_required=coverage.get('final_observation_required', False),
        action_start_offset=coverage['future_action_start_offset'],
        native_terminal_padding=coverage.get('native_terminal_padding', False))
    if coverage != expected:
        raise ValueError('Require the declared exhaustive original trajectory coverage.')
    for episode, length in zip(coverage['episode_ids'], coverage['episode_lengths'], strict=True):
        stop = length - coverage['future_action_start_offset'] - int(coverage.get('final_observation_required', False))
        if not coverage.get('native_terminal_padding', False):
            stop -= coverage['native_action_horizon'] - 1
        for frame in range(0, stop, coverage['condition_frame_stride']):
            yield episode, length, frame


def require_complete_window_rows(coverage, rows, *, frame_key='timestep'):
    if len(rows) != coverage['expected_rows']:
        raise ValueError('Incomplete full-trajectory context; no row cap or random fallback.')
    for row, (episode, length, frame) in zip(rows, all_complete_windows(coverage), strict=True):
        if (row['episode'], row['episode_length'], row[frame_key]) != (episode, length, frame):
            raise ValueError('Context must cover every original valid timestep exactly once, in order.')
    return coverage['expected_rows']


def require_complete_support_context(context):
    """Check exhaustive raw-label lineage, independently of admission or PFN fit."""
    if context.get('version') == VERSION:
        for _ in all_complete_windows(context):
            pass
        return context['expected_rows']
    if context.get('kind') == BANK_VERSION:
        tasks = context['corpus']['tasks']
        shards = context['source_shards']
        if (len(tasks) != 300 or len(set(tasks)) != 300
                or [row['task'] for row in shards] != tasks
                or (context['corpus']['split'], context['corpus']['source'],
                    context['corpus']['task_set']) != ('pretrain', 'human', 'pretrain300')):
            raise ValueError('Require every original human/pretrain300 task exactly once.')
        expected = 0
        for shard in shards:
            for _ in all_complete_windows(shard['coverage']):
                pass
            expected += shard['coverage']['expected_rows']
        if expected != context['expected_rows']:
            raise ValueError('The complete native bank row count differs from its trajectory coverage.')
        return expected
    for _ in all_complete_windows(context['coverage']):
        pass
    return context['coverage']['expected_rows']

"""Read the pinned upstream's main evaluation recipe without adding defaults."""
import hashlib
from pathlib import Path
import re
import ast
from collections import OrderedDict


def robocasa365_registry(simulator_directory):
    """Read the original declarative registry without importing a simulator."""
    path = Path(simulator_directory) / 'robocasa/utils/dataset_registry.py'
    names = {'ATOMIC_TASK_DATASETS', 'COMPOSITE_TASK_DATASETS', 'PRETRAINING_TASKS',
             'TARGET_TASKS', 'LIFELONG_LEARNING_TASKS', 'MG_DATASET_TASKS', 'TASK_SET_REGISTRY'}
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    if len(nodes) != len(names):
        raise RuntimeError('The original complete dataset registry changed.')
    namespace = {'__builtins__': {}, 'OrderedDict': OrderedDict, 'dict': dict, 'list': list}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in
            ('ATOMIC_TASK_DATASETS', 'COMPOSITE_TASK_DATASETS', 'TASK_SET_REGISTRY')}


def robocasa365_recipe(policy_directory, simulator_directory):
    directory = Path(policy_directory)
    entry = directory / 'eval_robocasa365/entry.py'
    readme = directory / 'eval_robocasa365/README.md'
    registry_path = Path(simulator_directory) / 'robocasa/utils/dataset_registry.py'
    tree = ast.parse(entry.read_text())
    names = {'--split', '--task-set', '--num-trials', '--replan-steps', '--obs-history',
             '--obs-interval', '--horizon', '--seed', '--crop-ratio', '--max-tasks'}
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in names):
            values = [kw.value for kw in node.keywords if kw.arg == 'default']
            if len(values) != 1:
                raise RuntimeError('Read all original scientific defaults; no replacements.')
            defaults[node.args[0].value] = ast.literal_eval(values[0])
    registry = robocasa365_registry(simulator_directory)
    if set(defaults) != names:
        raise RuntimeError('The complete original native evaluation defaults are required.')
    tasks = registry['TASK_SET_REGISTRY'][defaults['--task-set']]
    if len(tasks) != 50 or len(set(tasks)) != 50 or defaults['--max-tasks'] is not None:
        raise RuntimeError('Require the complete original fifty-task evaluation.')
    return {'native_entry': 'eval_robocasa365/entry.py', 'native_defaults': defaults,
        'tasks': tasks, 'total_episodes': len(tasks) * defaults['--num-trials'],
        'source_files_sha256': {
            'eval_robocasa365/entry.py': hashlib.sha256(entry.read_bytes()).hexdigest(),
            'eval_robocasa365/README.md': hashlib.sha256(readme.read_bytes()).hexdigest(),
            'robocasa/utils/dataset_registry.py': hashlib.sha256(registry_path.read_bytes()).hexdigest()},
        'native_reference_report': {'successes': 1432, 'episodes': 2500,
            'success_rate_percent': 57.28, 'source': 'Original eval README reference configuration'},
        'paper_headline_is_separate': True}


def robotwin_recipe(policy_directory):
    import yaml

    directory = Path(policy_directory)
    example = directory / 'paths_config.example.yml'
    entry = directory / 'eval.sh'
    values = yaml.safe_load(example.read_text())
    # These flags are selected by the original shell entry itself. Paths,
    # selected task/condition and the added PFN plugin are supplied separately.
    flags = set(re.findall(r'--([a-z_]+) "\$[A-Z_]+"', entry.read_text()))
    deployment = {'task_name', 'task_config', 'ckpt_setting', 'wan_path',
                  'vlm_path', 'policy_name', 'log_dir'}
    arguments = {key: value for key, value in values.items()
                 if key in flags and key not in deployment}
    if not arguments or arguments.get('inference_mode') != 'history_flow':
        raise RuntimeError('The main upstream history-flow recipe is required; no legacy fallback.')
    return {'source_files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (example, entry)},
            'arguments': arguments}

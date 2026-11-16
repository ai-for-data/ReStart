"""Read FlowDAgger's existing BC-buffer admission default, without tuning it."""
import ast
import hashlib
from pathlib import Path

from embodpfn.flowdagger import ENTRY_SOURCE, REFERENCE_DIRECTORY, reference_recipe


def context_rule(root):
    reference = reference_recipe(root)
    source = Path(root) / REFERENCE_DIRECTORY
    threshold = None
    for node in ast.walk(ast.parse((source / ENTRY_SOURCE).read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == '--inversion_mse_threshold'):
            threshold = ast.literal_eval(next(x.value for x in node.keywords if x.arg == 'default'))
    if threshold is None:
        raise RuntimeError('The author entry does not declare the required context admission rule.')
    filter_source = 'flowdagger_pi05/train_utils.py'
    return {'source_revision': reference['source_revision'], 'entry': ENTRY_SOURCE,
            'filter_source': filter_source,
            'filter_source_sha256': hashlib.sha256((source / filter_source).read_bytes()).hexdigest(),
            'threshold': float(threshold), 'rule': 'exclude chunk if reconstruction MSE > threshold',
            'upstream_coordinates': 'first seven executable pi0.5 model-action dimensions',
            'backend_coordinates': 'all fourteen executable native UniWAM RoboTwin action dimensions',
            'threshold_tuned': False}

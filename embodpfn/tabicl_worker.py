"""Explicit file boundary for the official TabICLv2 inference environment."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--kv-cache', choices=['none', 'kv', 'repr'], default='none')
    parser.add_argument('--execution', choices=['native', 'shared', 'shared-batched', 'pr87'], default='native')
    parser.add_argument('--tabicl-source', type=Path)
    args = parser.parse_args()
    if args.execution == 'pr87':
        if args.kv_cache != 'none':
            raise ValueError('Original PR87 multi-target execution requires cache=none.')
        if args.tabicl_source is None:
            raise ValueError('Select the entire pinned original PR87 checkout explicitly.')
        source = args.tabicl_source.resolve()
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
        if revision != 'd587fdcedcde83c64f3dd15fdfdd4fd165d38da4' or dirty:
            raise ValueError('Preserve the entire original PR87 source, without cherry-picks or edits.')
        sys.path.insert(0, str(source / 'src'))
        import tabicl
        if not Path(tabicl.__file__).resolve().is_relative_to(source):
            raise ValueError('The selected native PR87 source was not imported.')
    elif args.tabicl_source is not None:
        raise ValueError('Do not change native source implicitly for an existing execution mode.')
    # Keep official diagnostics visible on stderr. Stdout carries only this
    # process boundary's replies; the native estimator is otherwise unchanged.
    with redirect_stdout(sys.stderr):
        from embodpfn.tabicl_noise import TabICLNoisePredictor
        predictor = TabICLNoisePredictor(checkpoint=args.checkpoint, device=args.device,
                                        kv_cache={'none': False, 'kv': True, 'repr': 'repr'}[args.kv_cache],
                                        execution=args.execution)
    if args.execution == 'pr87':
        regressor_module = Path(sys.modules[type(predictor.estimator).__module__].__file__).resolve()
        if (type(predictor.estimator) is not tabicl.TabICLRegressor
                or not regressor_module.is_relative_to(source)):
            raise ValueError('Use the actual original PR87 TabICLRegressor directly.')
        parameters = predictor.estimator.get_params(deep=False)
        print(json.dumps({'original_TabICL_source': str(source), 'device': args.device,
            'revision': revision, 'pristine': not dirty,
            'tree': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
            'module': str(Path(tabicl.__file__).resolve()), 'regressor_module': str(regressor_module),
            'estimator': type(predictor.estimator).__module__ + '.' + type(predictor.estimator).__name__,
            'native_parameters': {key: parameters[key] for key in (
                'n_estimators', 'batch_size', 'kv_cache', 'use_amp', 'use_fa3', 'offload_mode')}
        }), file=sys.stderr, flush=True)
    last_execution = None
    for line in sys.stdin:
        closing = False
        try:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                if message['command'] == 'context':
                    with np.load(message['support'], allow_pickle=False) as support:
                        predictor.update_context(support['conditions'], support['noise'])
                    print(json.dumps({'pfn_context_ready': True, 'execution': args.execution,
                        'device': args.device, 'cache': args.kv_cache,
                        'context_rows': len(predictor.support_conditions),
                        'features': predictor.condition_dim, 'outputs': predictor.support_noise.shape[1],
                        'native_batch_size': (None if args.execution in ('shared-batched', 'pr87') else 8),
                        'ensembles_per_output': 8}), file=sys.stderr, flush=True)
                    last_execution = None
                elif message['command'] == 'predict':
                    query = np.load(message['query'], allow_pickle=False)
                    np.save(message['output'], predictor.predict_initial_noise(query), allow_pickle=False)
                    execution = getattr(predictor.estimator, 'last_query_preprocessing_', None)
                    if execution != last_execution:
                        print(json.dumps({'pfn_execution': execution}), file=sys.stderr, flush=True)
                        last_execution = execution
                elif message['command'] == 'close':
                    closing = True
                else:
                    raise ValueError('Unknown PFN boundary command.')
            result = {'status': 'ok'}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            result = {'status': 'error', 'error_type': type(error).__name__, 'message': str(error)}
        print(json.dumps(result), flush=True)
        if closing:
            return


if __name__ == '__main__':
    main()

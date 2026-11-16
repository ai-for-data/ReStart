"""Produce real expert inverse labels through the existing native-policy method.

This is the offline data-production stage. It neither fits the PFN nor invokes
or replaces any part of the original robot evaluation.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arguments', type=Path, required=True)
    args = parser.parse_args()
    arguments = json.loads(args.arguments.read_text())
    # Same policy import directory used by the original eval_policy.py.
    sys.path.append('./policy')
    import torch
    from embodpfn.robotwin_policy import get_model, _build_context

    output = Path(arguments['embodpfn_support'])
    if output.exists():
        raise FileExistsError('Fresh support production cannot overwrite prior labels.')
    policy = get_model(arguments)
    session = policy._embodpfn_context_session
    device = next(policy.model.parameters()).device
    try:
        with torch.random.fork_rng(devices=[device.index if device.index is not None
                                           else torch.cuda.current_device()]):
            _build_context(policy, session, arguments['embodpfn_expert_bundle'], output)
    finally:
        session.close()
    print(json.dumps({'offline_support_complete': True, 'output': str(output),
                      'robot_evaluation_run': False, 'pfn_fitted': False}), flush=True)


if __name__ == '__main__':
    main()

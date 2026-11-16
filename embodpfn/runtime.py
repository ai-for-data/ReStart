"""Explicit, scoped attachment at the native policy sampling entry point.

Construct the server with its upstream loader first. The caller then enters
this context around the original serving/evaluation loop. No sitecustomize,
import interception, processor replacement or automatic startup is involved.
"""
from contextlib import contextmanager
from functools import wraps
import inspect


def _prepared_inputs(name, signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    if name == 'generate' and set(bound.arguments) == {'batch'}:
        return bound.arguments['batch']
    if name == 'inference_step':
        return dict(bound.arguments)
    if name in ('predict', 'infer_online_action_chunk'):
        return dict(bound.arguments)
    raise RuntimeError('Unexpected native policy sampling contract.')


def collect_expert_call(adapter, official_call, expert_actions):
    """Let the original policy prepare inputs, then invert its expert target.

    The offline call ends at our sampling boundary. No fabricated action is
    returned to the policy, and no observation/text processor is duplicated.
    """
    backend = adapter.backend
    model = getattr(backend, 'entry_owner', backend.model)
    name, original = backend.entry_name, backend.official_entry
    if getattr(model, name) != original:
        raise RuntimeError('Collect expert context before attaching online adaptation.')
    signature = inspect.signature(original)
    had_local, previous = name in model.__dict__, model.__dict__.get(name)
    correction = None

    class ExpertCollected(Exception):
        pass

    @wraps(original)
    def collect(*args, **kwargs):
        nonlocal correction
        correction = adapter.add_demonstration(
            _prepared_inputs(name, signature, args, kwargs), expert_actions)
        raise ExpertCollected()

    setattr(model, name, collect)
    try:
        try:
            official_call()
        except ExpertCollected:
            return correction
        raise RuntimeError('The official expert call did not reach its native sampling entry.')
    finally:
        if had_local:
            setattr(model, name, previous)
        else:
            delattr(model, name)


@contextmanager
def attach_context_adapter(adapter):
    backend = adapter.backend
    model = getattr(backend, 'entry_owner', backend.model)
    name = backend.entry_name
    original = backend.official_entry
    if getattr(model, name) != original:
        raise RuntimeError('The native entry point is already wrapped; refuse a second policy change.')
    signature = inspect.signature(original)
    had_local = name in model.__dict__
    previous_local = model.__dict__.get(name)

    @wraps(original)
    def conditioned_entry(*args, **kwargs):
        from embodpfn.adaptation import ContextInferenceFailure
        try:
            return adapter.predict(_prepared_inputs(name, signature, args, kwargs))
        except ContextInferenceFailure as error:
            # Native RPC servers catch ordinary exceptions. Integration errors
            # must stop this method, not silently become robot task failures.
            raise SystemExit('PFN integration failed; this is not a benchmark result.') from error

    setattr(model, name, conditioned_entry)
    try:
        yield adapter
    finally:
        if had_local:
            setattr(model, name, previous_local)
        else:
            delattr(model, name)

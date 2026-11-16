"""Exact memoization of frozen condition encoders during our inverse only.

The native sampler, time grid, denoisers and decoder remain unchanged.
Every exported inverse is checked by a fresh, uncached native forward.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten, tree_map


def _snapshot(value):
    return tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x, value)


def _same(first, second):
    a, spec_a = tree_flatten(first)
    b, spec_b = tree_flatten(second)
    if spec_a != spec_b:
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if (not isinstance(y, torch.Tensor) or x.shape != y.shape or
                    x.dtype != y.dtype or x.device != y.device or not torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


@dataclass
class _Value:
    arguments: object
    output: object


class FrozenConditionCache:
    """One observation, frozen weights, exact inputs and exact native outputs.

    Cache hits return independent tensor copies because the native joint
    denoiser evolves its understanding tokens. Verification runs the original
    encoders and requires bitwise equality, without substituting their outputs.
    """

    def __init__(self, model):
        if model.training:
            raise RuntimeError('Inverse condition caching requires the original evaluation mode.')
        self.model = model
        self.values = {}
        self.calls = {}
        self.hits = {}
        self.verified = {}

    @contextmanager
    def install(self, *, verify=False):
        entries = (
            ('observed_video', self.model.video_model, 'encode_video'),
            ('understanding', self.model.und_module, 'extract_und_features'),
            ('t5_context', self.model.video_module, 'preprocess_t5_embeddings'),
        )
        with ExitStack() as stack:
            for key, owner, name in entries:
                original = getattr(owner, name)
                had_local = name in owner.__dict__
                previous = owner.__dict__.get(name)

                def restore(owner=owner, name=name, had_local=had_local, previous=previous):
                    if had_local:
                        setattr(owner, name, previous)
                    else:
                        delattr(owner, name)

                def call(*args, _key=key, _original=original, **kwargs):
                    arguments = (args, kwargs)
                    saved = self.values.get(_key)
                    if saved is not None and not _same(saved.arguments, arguments):
                        raise RuntimeError('Frozen inverse condition changed: ' + _key)
                    if saved is not None and not verify:
                        self.hits[_key] = self.hits.get(_key, 0) + 1
                        return _snapshot(saved.output)
                    with torch.no_grad():
                        output = _original(*args, **kwargs)
                    self.calls[_key] = self.calls.get(_key, 0) + 1
                    if saved is None:
                        if verify:
                            raise RuntimeError('No condition cache to verify: ' + _key)
                        self.values[_key] = _Value(_snapshot(arguments), _snapshot(output))
                    elif not _same(saved.output, output):
                        raise RuntimeError('Native condition encoder output changed: ' + _key)
                    else:
                        self.verified[_key] = self.verified.get(_key, 0) + 1
                    return output

                setattr(owner, name, call)
                stack.callback(restore)
            yield self

    def diagnostics(self):
        return {'scope': 'offline inverse only; no online sampler modification',
                'native_encoder_calls': dict(self.calls), 'exact_cache_hits': dict(self.hits),
                'fresh_native_output_verifications': dict(self.verified),
                'input_and_output_equality': 'bitwise; shape/dtype/device retained'}

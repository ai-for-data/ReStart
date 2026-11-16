"""Expert inversion → support context → official PFN → native policy loop.

The application constructs its model, processor, dataset and server/evaluator
with upstream code first. This module changes only initial-noise conditioning.
"""
from embodpfn.adaptation import ContextAdapter
from embodpfn.backends.uniwam import (UniWAMNoiseBackend, RoboTwinUniWAMNoiseBackend,
                                    RoboTwinUniWAMActionNoiseBackend, LiberoUniWAMNoiseBackend)
from embodpfn.runtime import attach_context_adapter, collect_expert_call
from embodpfn.tabicl_process import TabICLProcessPredictor


class NoiseContextSession:
    def __init__(self, official_model, *, runtime, inversion_settings, pfn_python,
                 pfn_checkpoint, pfn_device, checkpoint_receipt=None, context_rule=None,
                 pfn_kv_cache=False, pfn_execution='native', pfn_devices=None,
                 native_entry_owner=None, pfn_cuda_visible_devices=None, pfn_source=None):
        if runtime not in ('uniwam-robotwin', 'uniwam-robotwin-action-only', 'uniwam-robotwin-action-head'):
            raise ValueError('This release exposes only the declared UniWAM RoboTwin interfaces.')
        if context_rule is None:
            raise ValueError('Benchmark adaptation requires the declared author context admission rule.')
        from embodpfn.backends.uniwam_action_head import RoboTwinUniWAMHAnchorHeadBackend
        backend_type = {
            'uniwam-robotwin': RoboTwinUniWAMNoiseBackend,
            'uniwam-robotwin-action-only': RoboTwinUniWAMActionNoiseBackend,
            'uniwam-robotwin-action-head': RoboTwinUniWAMHAnchorHeadBackend,
        }[runtime]
        backend_arguments = dict(inversion_settings=inversion_settings, checkpoint_receipt=checkpoint_receipt)
        backend = backend_type(official_model, **backend_arguments)
        if pfn_devices:
            from embodpfn.tabicl_parallel import MultiDeviceTabICLProcessPredictor
            self.predictor = MultiDeviceTabICLProcessPredictor(python=pfn_python,
                checkpoint=pfn_checkpoint, devices=pfn_devices, kv_cache=pfn_kv_cache,
                execution=pfn_execution, cuda_visible_devices=pfn_cuda_visible_devices,
                tabicl_source=pfn_source)
        else:
            self.predictor = TabICLProcessPredictor(python=pfn_python, checkpoint=pfn_checkpoint,
                                                   device=pfn_device, kv_cache=pfn_kv_cache,
                                                   execution=pfn_execution,
                                                   cuda_visible_devices=pfn_cuda_visible_devices,
                                                   tabicl_source=pfn_source)
        self.adapter = ContextAdapter(backend, self.predictor, context_rule=context_rule)

    def build_support(self, official_expert_pairs, *, output):
        """Consume all supplied official (inference inputs, encoded expert chunk) pairs.

        The caller uses the native observation processor and action encoding.
        This interface performs no normalization, padding, slicing, task
        selection, reward filtering or transformation of expert trajectories.
        """
        for inputs, actions in official_expert_pairs:
            self.adapter.add_demonstration(inputs, actions)
        self.adapter.save_support(output)
        return self

    def load_support(self, path):
        self.adapter.load_support(path)
        return self

    def add_expert(self, official_inputs, official_actions):
        return self.adapter.add_demonstration(official_inputs, official_actions)

    def add_expert_call(self, official_call, official_actions):
        """Use the original policy wrapper to construct an expert's condition."""
        return collect_expert_call(self.adapter, official_call, official_actions)

    def run(self, official_run, *args, **kwargs):
        """Call the original serving/evaluation loop with its original arguments."""
        self.adapter.prepare_context()
        with attach_context_adapter(self.adapter):
            return official_run(*args, **kwargs)

    def close(self):
        self.predictor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

"""Partition independent native scalar-output estimators across GPU processes.

Each process uses the explicitly selected execution adapter around the official
TabICL estimator. Only output coordinates are partitioned: every estimator sees
all context rows and original input features, and retains all eight ensembles.
Native/shared/shared-batched execution is forwarded unchanged to every worker.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from embodpfn.tabicl_process import TabICLProcessPredictor


class MultiDeviceTabICLProcessPredictor:
    def __init__(self, *, python, checkpoint, devices, kv_cache=False, execution='native',
                 cuda_visible_devices=None, tabicl_source=None):
        if len(devices) < 2 or len(set(devices)) != len(devices):
            raise ValueError('Specify distinct allocated devices for coordinate parallelism.')
        self.workers = [TabICLProcessPredictor(python=python, checkpoint=checkpoint,
                                             device=device, kv_cache=kv_cache, execution=execution,
                                             cuda_visible_devices=cuda_visible_devices,
                                             tabicl_source=tabicl_source)
                        for device in devices]
        self.executor = ThreadPoolExecutor(max_workers=len(self.workers))
        self.has_context = False
        self._closed = False

    def update_context(self, conditions, inverted_noise):
        noise = np.asarray(inverted_noise)
        if noise.ndim < 2:
            raise ValueError('Expected one complete vector target per support row.')
        self.noise_shape = noise.shape[1:]
        flat = noise.reshape(len(noise), -1)
        if flat.shape[1] < len(self.workers):
            raise ValueError('The allocated output workers exceed the complete target width.')
        self.partitions = np.array_split(np.arange(flat.shape[1]), len(self.workers))
        self.has_context = False
        futures = [self.executor.submit(worker.update_context, conditions, flat[:, indices])
                   for worker, indices in zip(self.workers, self.partitions)]
        for future in futures:
            future.result()
        self.has_context = True
        return self

    def predict_initial_noise(self, conditions):
        if not self.has_context:
            raise RuntimeError('Provide the complete support context before prediction.')
        futures = [self.executor.submit(worker.predict_initial_noise, conditions)
                   for worker in self.workers]
        outputs = [future.result() for future in futures]
        return np.concatenate(outputs, axis=1).reshape(len(conditions), *self.noise_shape)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            # Python shuts executors down before user atexit callbacks. Close
            # the child processes directly, without scheduling new futures.
            for worker in self.workers:
                try:
                    worker.close()
                except Exception as error:
                    errors.append(error)
        finally:
            self.executor.shutdown(wait=True)
        if errors:
            raise errors[0]

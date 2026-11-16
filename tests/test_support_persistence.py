"""Artifact integrity only; no robot policy or benchmark substitute."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from embodpfn.adaptation import ContextAdapter
from embodpfn.interfaces import InvertedCorrection, PolicyCondition


class SupportPersistenceTests(unittest.TestCase):
    def test_resume_preserves_rows_and_rejects_another_checkpoint(self):
        backend = SimpleNamespace(asset={'weight_sha256': 'expected'}, readout_id='readout-v1', target_codec='direct-noise.v1')
        source = ContextAdapter(backend, None)
        condition = PolicyCondition(None, np.arange(12).reshape(3, 4), 'readout-v1', 'expected')
        source.corrections.append(InvertedCorrection(condition, np.arange(18).reshape(3, 2, 3), 1e-5, 'inverse-v1'))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'support.npz'
            source.save_support(path)
            restored = ContextAdapter(backend, None).load_support(path)
            np.testing.assert_array_equal(restored.corrections[0].condition.features, condition.features)
            np.testing.assert_array_equal(restored.corrections[0].initial_noise, source.corrections[0].initial_noise)
            with self.assertRaises(FileExistsError):
                source.save_support(path)
            wrong = SimpleNamespace(asset={'weight_sha256': 'other'}, readout_id='readout-v1', target_codec='direct-noise.v1')
            with self.assertRaises(ValueError):
                ContextAdapter(wrong, None).load_support(path)


if __name__ == '__main__':
    unittest.main()

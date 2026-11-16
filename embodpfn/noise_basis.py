"""Our joint-noise target coordinates, outside the official PFN and policy.

FlowDAgger Appendix A.3 motivates a basis for joint WAM noise. Here we keep
the complete centered support span (at most N-1 directions), rather than
silently choosing a fixed rank or discarding support-label variance.
"""
import json

import numpy as np


class CenteredSupportNoiseBasis:
    def fit_transform(self, noise):
        noise = np.asarray(noise, dtype=np.float32)
        if noise.ndim < 2 or len(noise) < 2 or not np.isfinite(noise).all():
            raise ValueError('The joint-noise basis requires at least two finite full-noise labels.')
        self.noise_shape = noise.shape[1:]
        flat = noise.reshape(len(noise), -1)
        self.mean = flat.mean(0, keepdims=True)
        centered = flat - self.mean
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self.components = vh[:min(len(noise) - 1, flat.shape[1])].copy()
        coordinates = centered @ self.components.T
        reconstructed = coordinates @ self.components + self.mean
        print(json.dumps({'joint_noise_basis': 'complete-centered-support-span',
            'rows': len(noise), 'noise_dimensions': flat.shape[1],
            'coordinate_dimensions': len(self.components),
            'support_roundtrip_max_error': float(np.abs(reconstructed - flat).max())}), flush=True)
        return coordinates

    def inverse_transform(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.components):
            raise ValueError('PFN output does not match the declared joint-noise basis.')
        noise = coordinates @ self.components + self.mean
        if not np.isfinite(noise).all():
            raise ValueError('The reconstructed joint noise is non-finite.')
        return noise.reshape(len(noise), *self.noise_shape)

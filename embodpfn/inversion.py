"""Shared result types for the pinned FlowDAgger backend port.

No custom solver, tolerance gate or alternative optimizer lives here.
"""
from dataclasses import dataclass

import torch


class InversionFailure(RuntimeError):
    """A failed inverse is never admitted to the support context."""


@dataclass(frozen=True)
class JointLatent:
    video: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True)
class JointInverseResult:
    noise: JointLatent
    reconstruction_max_error: float
    evaluations: int
    solver: str
    settings: dict
    reconstruction_metrics: dict

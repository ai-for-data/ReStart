"""Backend contract, not an implementation of a substitute robot sampler."""
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PolicyCondition:
    # Opaque output of the official processor/encoder, retained for inversion.
    official_condition: Any
    # Deterministic, explicitly versioned readout for the PFN only.
    features: np.ndarray
    readout_id: str
    policy_checkpoint_sha256: str


@dataclass(frozen=True)
class InvertedCorrection:
    condition: PolicyCondition
    initial_noise: np.ndarray
    reconstruction_error: float
    inversion_recipe_id: str
    reconstruction_metrics: dict = field(default_factory=dict)
    native_inverse_initial_point: np.ndarray | None = None


class FrozenPolicyBackend(Protocol):
    """An integration must implement these against its official policy code.

    No backend is declared working until tested through the full official eval
    entry point. Empty context is an error in an adapted run. Inversion must
    consume the exact official discretization and model-space target encoding.
    """

    target_codec: str

    # Inputs have already passed the official processor and normalization.
    # Backends must never duplicate these components.
    def capture(self, official_prepared_inputs: Any) -> PolicyCondition: ...

    def act_official(self, official_prepared_inputs: Any) -> Any: ...

    def act_conditioned(self, official_prepared_inputs: Any, predict_noise: Any) -> Any: ...

    def act_with_initial_noise(
        self, condition: PolicyCondition, initial_noise: np.ndarray
    ) -> Any: ...

    def invert_expert_action(
        self, condition: PolicyCondition, expert_action_chunk: Any
    ) -> InvertedCorrection: ...

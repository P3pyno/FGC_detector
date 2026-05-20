from __future__ import annotations
from abc import ABC, abstractmethod
import warnings
import torch


class VelocityAdapter(ABC):
    """Frozen surrogate velocity field adapter used as a geometric probe."""

    def __init__(self, probe_model_name: str = "unknown", frozen: bool = True):
        self.probe_model_name = probe_model_name
        self.frozen = frozen

    @abstractmethod
    def velocity(self, z: torch.Tensor, t: float | torch.Tensor, cond=None) -> torch.Tensor:
        raise NotImplementedError


class DummyLinearVelocityAdapter(VelocityAdapter):
    def velocity(self, z, t, cond=None):
        return -z


class DummyCurvedVelocityAdapter(VelocityAdapter):
    def velocity(self, z, t, cond=None):
        # Adds rotational term on first two channels for non-conservative behavior.
        v = -0.5 * z
        if z.shape[0] >= 2:
            a = z[0].clone()
            b = z[1].clone()
            v[0] = v[0] - (1.0 + 0.1 * float(t)) * b
            v[1] = v[1] + (1.0 + 0.1 * float(t)) * a
        return v


class DiffusersFlowVelocityAdapter(VelocityAdapter):
    def __init__(self, probe_model_name: str = "flux_surrogate", frozen: bool = True):
        super().__init__(probe_model_name, frozen)
        warnings.warn("Using dummy velocity adapter. This validates the geometry pipeline only, not real detector performance.")
        self.fallback = DummyCurvedVelocityAdapter(probe_model_name=probe_model_name, frozen=True)

    def velocity(self, z, t, cond=None):
        # TODO: Integrate Flux/SD3/SD3.5 internals for true flow-matching velocity evaluation.
        return self.fallback.velocity(z, t, cond=cond)

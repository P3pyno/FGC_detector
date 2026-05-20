import torch
from flowgeom.d1_curl import compute_d1_features
from flowgeom.velocity_adapter import VelocityAdapter, DummyCurvedVelocityAdapter

class ZeroV(VelocityAdapter):
    def velocity(self, z, t, cond=None):
        return torch.zeros_like(z)


def test_zero_field_near_zero_curl():
    z = torch.randn(3, 8, 8)
    lat = [z for _ in range(5)]
    ts = [1, 0.75, 0.5, 0.25, 0.0]
    _, d1, _ = compute_d1_features(lat, ts, ZeroV(), num_probes=8, eps=0.01)
    assert d1["curl_mean"] < 1e-3


def test_rotational_nonzero_curl():
    z = torch.randn(3, 8, 8)
    lat = [z for _ in range(5)]
    ts = [1, 0.75, 0.5, 0.25, 0.0]
    _, d1, _ = compute_d1_features(lat, ts, DummyCurvedVelocityAdapter(), num_probes=8, eps=0.01)
    assert d1["curl_mean"] > 1e-4

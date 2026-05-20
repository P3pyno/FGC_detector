import torch
from flowgeom.ode_inversion import invert_trajectory
from flowgeom.d2_trajectory import compute_d2_features
from flowgeom.velocity_adapter import DummyLinearVelocityAdapter, DummyCurvedVelocityAdapter


def test_straight_has_lower_curvature():
    z = torch.randn(3, 16, 16)
    lin = DummyLinearVelocityAdapter()
    cur = DummyCurvedVelocityAdapter()
    d2_lin = compute_d2_features(invert_trajectory(z, lin, num_steps=20))[1]
    d2_cur = compute_d2_features(invert_trajectory(z, cur, num_steps=20))[1]
    assert d2_lin["mean_curvature"] < d2_cur["mean_curvature"]

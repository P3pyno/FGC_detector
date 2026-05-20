import numpy as np
from flowgeom.coupling import compute_coupling_features


def test_coupling_no_nan():
    curl = {"t": [1, 0.5, 0.0], "curl": [0.1, 0.2, 0.3]}
    curv = {"t": [1, 0.5, 0.0], "curvature": [0.3, 0.2, 0.1]}
    _, feats = compute_coupling_features(curl, curv)
    assert all(np.isfinite(v) for v in feats.values())

from flowgeom.data import load_dataset
from flowgeom.feature_extraction import ExtractSettings, extract_features
from flowgeom.vae_adapter import VAEAdapter
from flowgeom.velocity_adapter import DummyCurvedVelocityAdapter


def test_pipeline_runs_random():
    ds = load_dataset(None, None, image_size=64)
    settings = ExtractSettings(num_steps=5, num_curl_probes=4, max_images=3)
    df = extract_features(ds, VAEAdapter(mode="identity"), DummyCurvedVelocityAdapter(), settings)
    assert len(df) == 3
    assert "T1_curl_score" in df.columns

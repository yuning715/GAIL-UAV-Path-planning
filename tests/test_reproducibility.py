import numpy as np

from src.data.missing_mask import random_missing_mask
from src.data.synthetic_expert_generator import generate_feature_trajectory


def test_expert_generation_seed_reproducible():
    a = generate_feature_trajectory(64, seed=3, track_id=1)["features"]
    b = generate_feature_trajectory(64, seed=3, track_id=1)["features"]
    assert np.allclose(a, b)


def test_missing_mask_seed_reproducible():
    a = random_missing_mask((50, 3), 0.1, seed=7)
    b = random_missing_mask((50, 3), 0.1, seed=7)
    assert np.array_equal(a, b)


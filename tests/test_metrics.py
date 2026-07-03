import numpy as np

from src.metrics.inpainting_metrics import mer, mser, tmse
from src.metrics.planning_metrics import score_interception


def test_inpainting_metric_formulas():
    real = np.array([0.0, 2.0, 4.0])
    pred = np.array([1.0, 2.0, 5.0])
    assert np.isclose(mer(pred, real), 2.0 / (3 * 4.0))
    assert np.isclose(tmse(pred, real), 2.0 / 3.0)
    assert np.isclose(mser(pred, real), 1.0 / 4.0)


def test_score_formula():
    assert score_interception(7, 2) == 5


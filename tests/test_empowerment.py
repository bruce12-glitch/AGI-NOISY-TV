import numpy as np

from aed.components.empowerment import EmpowermentEstimator


def test_independent_noise_has_low_empowerment():
    estimator = EmpowermentEstimator(normalized=True)
    rng = np.random.default_rng(4)
    for _ in range(1000):
        action = int(rng.integers(2))
        observation = int(rng.integers(8))
        estimator.observe("tv", action, observation)
    assert estimator.estimate("tv") < 0.05


def test_action_controlled_observation_has_high_empowerment():
    estimator = EmpowermentEstimator(normalized=True)
    for _ in range(100):
        for action in (0, 1):
            estimator.observe("puzzle", action, action)
    assert estimator.estimate("puzzle") > 0.95

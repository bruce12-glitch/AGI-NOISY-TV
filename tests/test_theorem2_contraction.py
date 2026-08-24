import numpy as np


def test_discounted_augmented_bellman_operator_is_contractive():
    gamma = 0.9
    rewards = np.array([1.0, -0.2, 0.5])

    def operator(values):
        # A toy finite-state operator with a state-independent bounded AED term.
        return rewards + gamma * np.roll(values, 1)

    x = np.array([0.2, 1.1, -0.7])
    y = np.array([-1.0, 0.1, 0.9])
    lhs = np.max(np.abs(operator(x) - operator(y)))
    rhs = gamma * np.max(np.abs(x - y))
    assert lhs <= rhs + 1e-12

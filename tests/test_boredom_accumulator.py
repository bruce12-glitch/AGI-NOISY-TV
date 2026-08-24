from aed.components.boredom_accumulator import BoredomAccumulator


def test_boredom_rises_below_threshold_and_crosses():
    accumulator = BoredomAccumulator(decay=1.0, maximum=1.0)
    values = [accumulator.step(utility=0.0, tau=0.3) for _ in range(4)]
    assert values == [0.3, 0.6, 0.8999999999999999, 1.2]
    assert accumulator.threshold_crossed
    assert accumulator.stagnation_time == 4


def test_boredom_decays_above_utility_threshold():
    accumulator = BoredomAccumulator(decay=0.9, maximum=10)
    accumulator.step(utility=0.0, tau=1.0)
    before = accumulator.value
    accumulator.step(utility=2.0, tau=1.0)
    assert accumulator.value < before

from aed.components.boredom_accumulator import BoredomAccumulator


def test_no_decay_escape_bound_is_reached_by_bmax_over_tau():
    b_max = 10.0
    tau = 0.3
    accumulator = BoredomAccumulator(decay=1.0, maximum=b_max)
    crossing_step = None
    for step in range(1, 100):
        accumulator.step(utility=0.0, tau=tau)
        if accumulator.threshold_crossed:
            crossing_step = step
            break
    assert crossing_step == 34  # ceil(10 / .3)
    assert crossing_step <= int(b_max / tau) + 1

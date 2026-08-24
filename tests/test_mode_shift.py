from aed.components.mode_shift import ModeShiftController


def test_mode_shift_is_finite_and_boosts_epsilon():
    shift = ModeShiftController(enabled=True, epsilon_boost=0.6, duration=4)
    assert shift.update(10, True)
    assert shift.trigger_count == 1
    assert shift.is_active(10)
    assert shift.exploration_epsilon(0.05, 10) == 0.65
    assert not shift.is_active(14)
    # A second crossing event is a valid future trigger; the accumulator
    # itself emits `True` only on a rising edge.
    assert shift.update(14, True)
    assert shift.trigger_count == 2


def test_disabled_mode_shift_does_not_trigger():
    shift = ModeShiftController(enabled=False)
    assert not shift.update(0, True)
    assert shift.trigger_count == 0

from aed.components.learning_progress import LearningProgress


def test_learning_progress_is_positive_when_error_falls():
    progress = LearningProgress(ema_decay=0.5)
    assert progress.update(1.0) == 0.0
    values = [progress.update(error) for error in (0.5, 0.25, 0.1)]
    assert all(value > 0 for value in values)


def test_learning_progress_is_near_zero_for_constant_error():
    progress = LearningProgress(ema_decay=0.5)
    values = [progress.update(0.5) for _ in range(20)]
    assert abs(values[-1]) < 1e-6

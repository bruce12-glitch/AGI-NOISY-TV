from aed.envs.stochastic_corridor import Room, StochasticCorridor


def test_corridor_regions_and_rewards():
    env = StochasticCorridor(seed=7)
    _, info = env.reset(seed=7)
    assert info["room"] == Room.NOISY_TV
    _, reward, *_ = env.step(env.WATCH_TV)
    assert reward == 0.0
    _, reward, _, _, info = env.step(env.MOVE_CORRIDOR)
    assert info["room"] == Room.PUZZLE
    assert reward == 0.0
    _, reward, _, _, info = env.step(env.WORK_PUZZLE)
    assert info["room"] == Room.PUZZLE
    assert reward == 1.0
    _, _, _, _, info = env.step(env.MOVE_CORRIDOR)
    assert info["room"] == Room.MASTERED
    _, _, _, _, info = env.step(env.MOVE_CORRIDOR)
    assert info["room"] == Room.PUZZLE


def test_tv_observations_are_stochastic_but_do_not_change_room():
    env = StochasticCorridor(tv_vocab=8, seed=1)
    env.reset(seed=1)
    observations = [env.step(env.WATCH_TV)[0] for _ in range(40)]
    assert env.state.room == Room.NOISY_TV
    assert len(set(observations)) > 1

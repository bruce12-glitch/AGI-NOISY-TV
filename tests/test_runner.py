from pathlib import Path

from aed.utils import load_config
from experiments.common import run_seed, write_rows


def test_runner_emits_stable_telemetry(tmp_path: Path):
    config = load_config("configs/corridor/aed.yaml")
    rows = run_seed(config, seed=0, steps=30)
    assert len(rows) == 30
    required = {"method", "seed", "room", "boredom", "mode_shift", "external_reward"}
    assert required.issubset(rows[0])
    target = write_rows(tmp_path / "telemetry.csv", rows)
    assert target.exists()
    assert target.read_text().splitlines()[0].startswith("method,seed,step")

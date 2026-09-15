from pathlib import Path

import pytest

from quantumflow.cli import _config_path, load_run_config


@pytest.mark.parametrize(
    ("experiment", "run_name"),
    [("resnets", "resnet_100"), ("xdiff", "default")],
)
def test_loads_existing_experiment_configuration(experiment: str, run_name: str) -> None:
    config = load_run_config(experiment, run_name, Path("experiments"))

    assert config["seed"] == 0


def test_selects_single_nonstandard_config_name() -> None:
    assert _config_path("resnets", Path("experiments")).name == "resnets.yaml"


def test_missing_run_lists_available_runs() -> None:
    with pytest.raises(KeyError, match="available"):
        load_run_config("xdiff", "missing-run", Path("experiments"))

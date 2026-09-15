"""Command-line entry points for reproducible QuantumFlow experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_path(experiment: str, experiments_dir: Path) -> Path:
    """Return the single configuration file for an experiment."""
    experiment_dir = experiments_dir / experiment
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    preferred = experiment_dir / "hyperparams.yaml"
    if preferred.is_file():
        return preferred

    configs = sorted(experiment_dir.glob("*.yaml"))
    if len(configs) == 1:
        return configs[0]
    if not configs:
        raise FileNotFoundError(f"No YAML configuration found in: {experiment_dir}")
    raise ValueError(f"Multiple YAML configurations found in: {experiment_dir}")


def load_run_config(experiment: str, run_name: str, experiments_dir: Path) -> dict[str, Any]:
    """Load a named run configuration and report invalid selections clearly."""
    config_path = _config_path(experiment, experiments_dir)
    config = YAML(typ="safe").load(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    if run_name not in config:
        available = ", ".join(config) or "(none)"
        raise KeyError(f"Run '{run_name}' is not defined in {config_path}; available: {available}")
    run_config = config[run_name]
    if not isinstance(run_config, dict):
        raise ValueError(f"Run '{run_name}' in {config_path} must be a mapping")
    return run_config


def _run_directory(experiment: str, run_name: str, output_dir: Path | None) -> Path:
    root = output_dir or PROJECT_ROOT / "outputs"
    run_dir = root / experiment / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def generate_dataset(experiment: str, run_name: str, output_dir: Path | None = None) -> None:
    """Build the dataset selected by a named experiment configuration."""
    import quantumflow

    config = load_run_config(experiment, run_name, PROJECT_ROOT / "experiments")
    run_dir = _run_directory(experiment, run_name, output_dir)
    dataset = quantumflow.instantiate(config, run_dir=run_dir)
    dataset.build()


def train(experiment: str, run_name: str, output_dir: Path | None = None) -> None:
    """Train a model selected by a named experiment configuration."""
    import tensorflow as tf

    import quantumflow

    config = load_run_config(experiment, run_name, PROJECT_ROOT / "experiments")
    run_dir = _run_directory(experiment, run_name, output_dir)
    dataset_train = quantumflow.instantiate(config["dataset_train"], run_dir=run_dir)
    dataset_train.build()
    dataset_validate = quantumflow.instantiate(config["dataset_validate"], run_dir=run_dir)
    dataset_validate.build()

    tf.keras.backend.clear_session()
    tf.random.set_seed(config["seed"])
    model = quantumflow.instantiate(config["model"], run_dir=run_dir, dataset=dataset_train)
    optimizer = quantumflow.instantiate(config["optimizer"])
    model.compile(
        optimizer=optimizer,
        loss=config["loss"],
        loss_weights=config.get("loss_weights"),
        metrics=config.get("metrics"),
    )

    checkpoint_path = config.get("load_checkpoint")
    if checkpoint_path:
        model.load_weights(run_dir / checkpoint_path)

    callbacks = []
    checkpoint = config.get("checkpoint")
    if checkpoint:
        checkpoint_config = dict(checkpoint)
        filename = checkpoint_config.pop("filename", "weights.{epoch:05d}.hdf5")
        checkpoint_config["filepath"] = run_dir / filename
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoint_config))
    if "tensorboard" in config:
        callbacks.append(
            quantumflow.instantiate(
                config["tensorboard"], log_dir=run_dir, learning_rate=optimizer.learning_rate
            )
        )

    model.fit(
        x=dataset_train.features,
        y=dataset_train.targets,
        callbacks=callbacks,
        validation_data=(dataset_validate.features, dataset_validate.targets),
        **config["fit"],
    )
    if config.get("save"):
        save_model_name = config.get("save_model", "self")
        save_model = model if save_model_name == "self" else getattr(model, save_model_name)
        save_model.save(run_dir / "saved_model", include_optimizer=False)


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("experiment", help="experiment directory under experiments/")
    parser.add_argument("run_name", help="named run in the experiment configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for generated artifacts (default: <repository>/outputs)",
    )
    return parser


def dataset_main() -> None:
    args = _parser("Generate a configured dataset.").parse_args()
    generate_dataset(args.experiment, args.run_name, args.output_dir)


def train_main() -> None:
    args = _parser("Train a configured model.").parse_args()
    train(args.experiment, args.run_name, args.output_dir)

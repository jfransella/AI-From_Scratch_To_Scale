import sys
import importlib.util
from pathlib import Path
import pytest

# Patch sys.path for Perceptron src
perceptron_src = (
    Path(__file__).parent.parent.parent / "models" / "01_perceptron" / "src"
)
sys.path.insert(0, str(perceptron_src))

config_mod = importlib.util.spec_from_file_location(
    "config", perceptron_src / "config.py"
)
config = importlib.util.module_from_spec(config_mod)
config_mod.loader.exec_module(config)


def test_get_training_config_valid():
    cfg = config.get_training_config("debug_small")
    assert hasattr(cfg, "experiment_name")
    assert cfg.experiment_name == "debug_small"
    assert hasattr(cfg, "learning_rate")
    assert cfg.learning_rate > 0


def test_get_training_config_invalid():
    with pytest.raises(ValueError):
        config.get_training_config("not_a_real_experiment")


def test_get_evaluation_config_valid():
    cfg = config.get_evaluation_config("debug_small")
    # EvaluationConfig may not have 'experiment_name', just check type
    assert hasattr(cfg, "compute_accuracy")
    assert hasattr(cfg, "output_path")


def test_get_model_config_valid():
    cfg = config.get_model_config("debug_small")
    assert isinstance(cfg, dict)
    assert "input_size" in cfg
    assert "learning_rate" in cfg


def test_get_model_config_invalid():
    with pytest.raises(ValueError):
        config.get_model_config("not_a_real_experiment")


def test_get_dataset_config_valid():
    cfg = config.get_dataset_config("debug_small")
    assert isinstance(cfg, dict)
    assert "dataset_name" in cfg
    assert "dataset_params" in cfg


def test_get_dataset_config_invalid():
    with pytest.raises(ValueError):
        config.get_dataset_config("not_a_real_experiment")


def test_print_config_summary(capsys):
    config.print_config_summary("debug_small")
    captured = capsys.readouterr()
    assert "debug_small" in captured.out


def test_get_complete_config():
    cfg = config.get_complete_config("debug_small")
    assert isinstance(cfg, dict)
    # Check for actual keys present in the returned config
    assert "dataset_config" in cfg
    assert "model_config" in cfg
    assert "training_config" in cfg

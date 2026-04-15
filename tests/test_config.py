"""Tests for YAML config loading."""

import pytest
from diffusion_sketch.config import load_config


class TestConfigLoading:
    def test_default_config_loads(self):
        cfg = load_config()
        assert "data" in cfg
        assert "diffusion" in cfg
        assert "model" in cfg
        assert "loss" in cfg
        assert "training" in cfg
        assert "paths" in cfg

    def test_attribute_access(self):
        cfg = load_config()
        assert cfg.training["epochs"] == 200
        assert cfg.diffusion["timesteps"] == 1000

    def test_overrides(self):
        cfg = load_config(overrides={"training.epochs": 5, "training.batch_size": 2})
        assert cfg.training["epochs"] == 5
        assert cfg.training["batch_size"] == 2

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_nested_override(self):
        cfg = load_config(overrides={"loss.lambda_l1": 42.0})
        assert cfg.loss["lambda_l1"] == 42.0

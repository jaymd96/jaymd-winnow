"""Tests for configuration dataclasses."""

import pytest

from jaymd_winnow.config import (
    ClusteringConfig,
    ModelConfig,
    PipelineConfig,
    ScreeningConfig,
    TargetConfig,
)


def test_defaults():
    config = PipelineConfig()
    assert config.min_history == 252
    assert len(config.targets) == 1
    assert config.targets[0].name == "target"
    assert config.targets[0].task == "regression"


def test_frozen():
    config = PipelineConfig()
    with pytest.raises(AttributeError):
        config.min_history = 100


def test_custom_targets():
    config = PipelineConfig(
        targets=[
            TargetConfig(name="returns", task="regression"),
            TargetConfig(name="direction", task="binary", model=ModelConfig(model_type="lightgbm")),
        ]
    )
    assert len(config.targets) == 2
    assert config.targets[1].model.model_type == "lightgbm"


def test_clustering_defaults():
    c = ClusteringConfig()
    assert c.max_clusters == 200
    assert c.method == "ward"


def test_screening_defaults():
    s = ScreeningConfig()
    assert s.threshold == 0.6
    assert s.n_bootstraps == 200


def test_model_config_defaults():
    m = ModelConfig()
    assert m.model_type == "elastic_net"
    assert m.cv_folds == 5
    assert len(m.confidence_levels) == 2

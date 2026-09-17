"""Unit tests — paper-locked configuration (Table 3)."""

from app.core.config import BASE_CONFIG, BIG_CONFIG, PAPER_TRAINING


def test_base_config_matches_paper_table3():
    assert BASE_CONFIG.name == "base"
    assert BASE_CONFIG.num_encoder_layers == 6
    assert BASE_CONFIG.num_decoder_layers == 6
    assert BASE_CONFIG.d_model == 512
    assert BASE_CONFIG.d_ff == 2048
    assert BASE_CONFIG.num_heads == 8
    assert BASE_CONFIG.d_k == 64
    assert BASE_CONFIG.d_v == 64
    assert BASE_CONFIG.dropout == 0.1
    assert BASE_CONFIG.label_smoothing == 0.1
    assert BASE_CONFIG.train_steps == 100_000


def test_big_config_matches_paper_table3():
    assert BIG_CONFIG.d_model == 1024
    assert BIG_CONFIG.d_ff == 4096
    assert BIG_CONFIG.num_heads == 16
    assert BIG_CONFIG.dropout == 0.3
    assert BIG_CONFIG.train_steps == 300_000


def test_training_regime_matches_paper_s5():
    assert PAPER_TRAINING.optimizer == "Adam"
    assert PAPER_TRAINING.beta1 == 0.9
    assert PAPER_TRAINING.beta2 == 0.98
    assert PAPER_TRAINING.epsilon == 1e-9
    assert PAPER_TRAINING.warmup_steps == 4000


def test_paper_configs_are_frozen():
    import pytest

    with pytest.raises((AttributeError, TypeError)):
        BASE_CONFIG.d_model = 1
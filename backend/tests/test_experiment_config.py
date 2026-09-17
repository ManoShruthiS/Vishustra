"""PHASE 7 — Experiment configuration tests (plan.md §14 grid, §15 record)."""

from app.core.config import PaperConfig
from app.experiments.config import ExperimentConfig, generate_grid


def default_cfg() -> ExperimentConfig:
    return ExperimentConfig(name="sweep", hypothesis="heads affect quality")


def test_config_round_trip():
    cfg = default_cfg()
    cfg2 = ExperimentConfig.from_dict(cfg.to_dict())
    assert cfg == cfg2


def test_as_paper_config_resolves_d_ff_and_d_k():
    cfg = ExperimentConfig(name="x", architecture={"d_model": 32, "num_heads": 4})
    paper = cfg.as_paper_config()
    assert isinstance(paper, PaperConfig)
    assert paper.d_ff == 64  # default 2 * d_model
    assert paper.d_k == 8    # default d_model // heads
    assert paper.num_heads == 4


def test_generate_grid_cartesian_product():
    cfg = default_cfg()
    grid = generate_grid(
        cfg,
        {
            "architecture": {"num_heads": [1, 2], "num_encoder_layers": [1, 2]},
            "training": {"epochs": [3, 6]},
        },
    )
    assert len(grid) == 2 * 2 * 2
    model_names = {g.model_name() for g in grid}
    assert len(model_names) == 4  # every head/layer combination present


def test_generate_grid_unknown_section_raises():
    try:
        generate_grid(default_cfg(), {"bogus": {"epochs": [1, 2]}})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown section")


def test_generate_grid_no_overrides_returns_single():
    assert len(generate_grid(default_cfg(), {})) == 1
    assert len(generate_grid(default_cfg(), {"training": {}})) == 1
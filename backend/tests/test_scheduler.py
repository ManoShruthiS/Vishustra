"""PHASE 5 — Noam scheduler tests (paper §5.3, eq. 3)."""

import math

from app.training import noam_lr

D_MODEL = 16


def test_noam_formula_warmup_region_is_linear():
    warmup = 100
    for step in (1, 10, 25, 50, 99):
        expected = D_MODEL ** -0.5 * step * warmup ** -1.5
        assert math.isclose(noam_lr(D_MODEL, step, warmup), expected, rel_tol=1e-9)


def test_noam_formula_decay_region():
    warmup = 100
    for step in (101, 500, 5000):
        expected = D_MODEL ** -0.5 * step ** -0.5
        assert math.isclose(noam_lr(D_MODEL, step, warmup), expected, rel_tol=1e-9)


def test_noam_peak_at_warmup():
    warmup = 4000
    peak = noam_lr(D_MODEL, warmup, warmup)
    after = noam_lr(D_MODEL, warmup * 2, warmup)
    assert peak == D_MODEL ** -0.5 * warmup ** -0.5
    assert after < peak


def test_noam_scheduler_tracks_optimizer_lr():
    from app.training.scheduler import NoamScheduler
    from torch import nn
    from torch.optim import SGD

    layer = nn.Linear(4, 4)
    opt = SGD(layer.parameters(), lr=1.0)
    sched = NoamScheduler(opt, d_model=16, warmup_steps=100)
    for _ in range(404):
        opt.step()
        sched.step()
        lr = opt.param_groups[0]["lr"]
        assert math.isclose(lr, noam_lr(16, sched.step_count, 100), rel_tol=1e-9)
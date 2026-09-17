"""PHASE 5 — label-smoothing loss tests (paper §5.4)."""

import math

import torch
from app.training import LabelSmoothingLoss
from app.transformer import log_softmax


def test_loss_matches_smoothed_closed_form():
    vocab = 6
    eps = 0.1
    logits = torch.randn(3, 5, vocab)
    targets = torch.randint(0, vocab, (3, 5))
    loss = LabelSmoothingLoss(vocab, smoothing=eps)(logits, targets)

    lp = log_softmax(logits, -1)
    nll_gt = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll_avg = -lp.mean(-1)
    expected = ((1 - eps) * nll_gt + eps * nll_avg).mean()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_ignored_positions_excluded():
    vocab = 6
    logits = torch.randn(2, 4, vocab) * 5
    targets = torch.tensor([[1, 2, -100, -100], [3, 4, 5, -100]])
    eps = 0.1
    loss = LabelSmoothingLoss(vocab, smoothing=eps, ignore_index=-100)(logits, targets)

    lp = log_softmax(logits, -1)
    safe = targets.clamp(min=0)
    nll_gt = -lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll_avg = -lp.mean(-1)
    valid = targets != -100
    expected = ((1 - eps) * nll_gt + eps * nll_avg)[valid].mean()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_minimum_loss_is_smoothed_target_entropy():
    """When softmax == the smoothed target exactly, loss == H(target)."""
    vocab = 6
    eps = 0.1
    p_y = 1 - eps + eps / vocab
    p_off = eps / vocab
    ratio = p_y / p_off
    logits = torch.zeros(vocab, vocab)
    logits.fill_diagonal_(math.log(ratio))
    targets = torch.arange(vocab).repeat(2, 1)
    loss = LabelSmoothingLoss(vocab, smoothing=eps)(logits[None].repeat(2, 1, 1), targets)
    expected = -((1 - eps) + eps / vocab) * math.log(p_y) - (eps * (vocab - 1) / vocab) * math.log(p_off)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-5)


def test_no_smoothing_is_plain_ce():
    vocab = 6
    logits = torch.randn(2, 3, vocab)
    targets = torch.randint(0, vocab, (2, 3))
    loss = LabelSmoothingLoss(vocab, smoothing=0.0)(logits, targets)
    lp = log_softmax(logits, -1)
    expected = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    assert torch.isclose(loss, expected, atol=1e-6)
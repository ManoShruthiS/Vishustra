"""Unit tests — positional encoding (paper §3.5, eq. 3/4)."""

import torch
from app.transformer import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    create_positional_encoding,
    positional_encoding,
)


def test_table_shape():
    pe = positional_encoding(20, 128)
    assert tuple(pe.shape) == (20, 128)


def test_position_zero_structure():
    """sin(0)=0 on even cols, cos(0)=1 on odd cols."""
    pe = positional_encoding(10, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6)


def test_matches_paper_formula():
    """PE(pos,2i)=sin(pos/10000^(2i/d)); PE(pos,2i+1)=cos(same angle)."""
    d_model = 16
    pe = positional_encoding(20, d_model)
    for pos in (3, 7, 9):
        for i in range(d_model // 2):
            angle = pos / (10000 ** ((2 * i) / d_model))
            assert torch.allclose(pe[pos, 2 * i], torch.sin(torch.tensor(angle)), atol=1e-6)
            assert torch.allclose(pe[pos, 2 * i + 1], torch.cos(torch.tensor(angle)), atol=1e-6)


def test_odd_d_model_truncates_cosine_column():
    pe = positional_encoding(8, 7)
    assert pe.shape == (8, 7)
    assert not torch.isnan(pe).any()


def test_sinusoidal_module_adds_pe_to_input():
    max_len, d_model, seq = 10, 8, 5
    mod = SinusoidalPositionalEncoding(max_len, d_model)
    x = torch.randn(3, seq, d_model)
    out = mod(x)
    assert torch.allclose(out, x + mod.pe[:seq], atol=1e-6)
    assert torch.allclose(mod.encoding_for(0), mod.pe[0], atol=1e-6)


def test_sinusoidal_module_not_trainable():
    mod = SinusoidalPositionalEncoding(10, 8)
    assert all(not p.requires_grad for p in mod.parameters())


def test_learned_module_is_trainable_and_adds_table():
    max_len, d_model, seq = 10, 8, 5
    mod = LearnedPositionalEncoding(max_len, d_model)
    assert mod.pe.shape == (max_len, d_model)
    assert mod.pe.requires_grad
    x = torch.randn(3, seq, d_model)
    out = mod(x)
    assert torch.allclose(out, x + mod.pe[:seq], atol=1e-6)


def test_factory_kinds():
    assert isinstance(create_positional_encoding("sinusoidal", 10, 8), SinusoidalPositionalEncoding)
    assert isinstance(create_positional_encoding("learned", 10, 8), LearnedPositionalEncoding)
    try:
        create_positional_encoding("nope", 10, 8)
        raised = False
    except ValueError:
        raised = True
    assert raised
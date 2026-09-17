"""PHASE 6 — Positional encoding waveform plotter tests (paper §3.5, eq. 3)."""

from app.visualization.positional_encoding_plot import plot_positional_encoding


def test_pe_plot_renders_file(tmp_path):
    out = plot_positional_encoding(32, 16, num_waves=6, out_path=tmp_path / "pe.png")
    assert out.stat().st_size > 0


def test_pe_plot_caps_waves_at_d_model(tmp_path):
    out = plot_positional_encoding(8, 4, num_waves=100, out_path=tmp_path / "pe_small.png")
    assert out.stat().st_size > 0
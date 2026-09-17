"""Visualization engine.

First milestone (plan.md §62): render scaled dot-product attention so the
paper's mathematics becomes visible.  Phase 6 (plan.md §66) expands this
into the Attention Observatory (§8): head/layer visualizations, attention
graphs, position-encoding waveforms and training graphs.
"""

from app.visualization.attention_graph import plot_attention_graph
from app.visualization.attention_plot import plot_attention_weights, plot_head_grid
from app.visualization.layer_visualization import plot_layer_strip
from app.visualization.positional_encoding_plot import plot_positional_encoding
from app.visualization.training_plot import plot_training_curve

__all__ = [
    "plot_attention_graph",
    "plot_attention_weights",
    "plot_head_grid",
    "plot_layer_strip",
    "plot_positional_encoding",
    "plot_training_curve",
]
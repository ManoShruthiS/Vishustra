"""Visualization engine.

First milestone (plan.md §62): render scaled dot-product attention so the
paper's mathematics becomes visible. Later phases extend this into the
Attention Observatory (§8) and representation views.
"""

from app.visualization.attention_plot import plot_attention_weights
from app.visualization.training_plot import plot_training_curve

__all__ = ["plot_attention_weights", "plot_training_curve"]
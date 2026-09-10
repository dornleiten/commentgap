"""Plotting helpers for vote-attention diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_vote_attention_curve(
    curve: pd.DataFrame,
    fit: dict,
    output_path: Path,
    *,
    show: bool = True,
):
    """Plot observed vote attention and the fitted sensitivity curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    fit_labels = {
        "spline": "Smoothed empirical spline (primary)",
        "power_law": f"Fitted power law (alpha={fit['models']['power_law']['parameter']:.3f})",
        "exponential": f"Fitted exponential (lambda={fit['models']['exponential']['parameter']:.5f})",
    }
    fit_colors = {"spline": "#167d8d", "power_law": "#cf6426", "exponential": "#8354aa"}
    for axis in axes:
        positive = curve.vote_fraction.gt(0)
        axis.scatter(
            curve.loc[positive, "rank"], curve.loc[positive, "vote_fraction"],
            s=10, alpha=0.45, color="black", label="Observed mean within-story vote fraction",
        )
        for model, label in fit_labels.items():
            axis.plot(
                curve["rank"], curve[f"{model}_fitted_vote_fraction"],
                color=fit_colors[model], linewidth=2, label=label,
            )
        axis.set(xlabel="Default visible display rank", ylabel="Mean within-story vote fraction")
    axes[0].set(xlim=(1, 40), title="First 40 positions")
    axes[1].set(xscale="log", yscale="log", title="Full development range (log scales)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Development vote attention: length-adjusted fitted curves")
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    return fig

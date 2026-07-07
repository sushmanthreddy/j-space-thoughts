"""Consistent, report-ready plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style() -> None:
    """Apply one accessible style across every executed notebook."""

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: plt.Figure, path: str | Path) -> Path:
    """Save a tight PNG and return its resolved path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight", facecolor="white")
    return target.resolve()


def validation_scatter(
    predicted: np.ndarray | list[float],
    measured: np.ndarray | list[float],
    *,
    correlation: float,
    n: int,
    title: str = "Attribution prediction vs. real ablation",
) -> tuple[plt.Figure, plt.Axes]:
    """F5 scatter with an OLS fit, identity line, Pearson r, and N."""

    set_style()
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(measured, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        raise ValueError("At least two finite points are required")

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    sns.regplot(
        x=x,
        y=y,
        ax=axis,
        scatter_kws={"s": 38, "alpha": 0.8, "edgecolor": "none"},
        line_kws={"color": "#B33A3A", "lw": 2},
        ci=None,
    )
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    axis.plot([lower, upper], [lower, upper], "--", color="0.45", lw=1, label="identity")
    axis.set(
        xlabel=r"Predicted $\Delta M=-\sum WRITE\,READ$",
        ylabel=r"Measured $\Delta M=M_{abl}-M_{clean}$",
        title=title,
    )
    axis.text(
        0.04,
        0.96,
        f"Pearson r = {correlation:.3f}\nN = {n}",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.legend(frameon=False, loc="lower right")
    return figure, axis

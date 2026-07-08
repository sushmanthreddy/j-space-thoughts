"""Pure, report-ready plots for the final READ evaluation.

Each ``plot_*`` function consumes already-computed rows and summaries, creates
one figure, and performs no model loading, evaluation, or file I/O.  The
``generate_final_figures`` convenience function is the sole orchestration layer
that writes the six fixed paper figures to a caller-provided directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PAPER_DPI = 180
PLOT_SEED = 1729
PAPER_RC: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": PAPER_DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}


def set_style() -> None:
    """Apply the fixed paper style for interactive notebook use."""

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update(PAPER_RC)


def save_figure(figure: plt.Figure, path: str | Path) -> Path:
    """Save one white-background PNG at the fixed paper resolution."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        target,
        dpi=PAPER_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    return target.resolve()


def _estimator_rows(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index a binary-evaluation AUC table and enforce unique estimators."""

    table = summary.get("auc_table")
    if isinstance(table, (str, bytes)) or not isinstance(table, Sequence):
        raise TypeError("binary summary must contain an auc_table sequence")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in table:
        if not isinstance(row, Mapping) or "estimator" not in row:
            raise TypeError("every AUC record must be a mapping with estimator")
        name = str(row["estimator"])
        if name in indexed:
            raise ValueError(f"duplicate AUC estimator {name!r}")
        indexed[name] = row
    return indexed


def _task_values(
    rows: Sequence[Mapping[str, Any]], task: str, field: str
) -> np.ndarray:
    """Extract a non-empty finite field vector for one task."""

    values = np.asarray(
        [row[field] for row in rows if row.get("task") == task], dtype=np.float64
    )
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{task}.{field} must be a non-empty finite vector")
    return values


def plot_causal_sanity(
    task_rows: Sequence[Mapping[str, Any]],
    causal_sanity: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot signed causal effects for matched used and idle concepts (F1)."""

    engine = _task_values(task_rows, "engine", "C")
    dashboard = _task_values(task_rows, "dashboard", "C")
    rng = np.random.default_rng(PLOT_SEED)
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(8.0, 5.8))
        values = [engine, dashboard]
        axis.boxplot(
            values,
            tick_labels=["Used\n(engine)", "Visible but idle\n(dashboard)"],
            showfliers=False,
        )
        for position, group_values, color in zip(
            (1, 2), values, ("#2166AC", "#EF8A62"), strict=True
        ):
            jitter = rng.normal(0.0, 0.045, size=group_values.size)
            axis.scatter(
                np.full(group_values.size, position) + jitter,
                group_values,
                s=20,
                alpha=0.55,
                color=color,
                edgecolor="none",
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_ylabel("Signed causal recovery C (unclipped)")
        axis.set_title(
            "Causal intervention sanity check\n"
            f"median C: used={float(causal_sanity['engine_C_median']):.3f}, "
            f"idle={float(causal_sanity['dashboard_C_median']):.3f}"
        )
        figure.tight_layout()
    return figure, axis


def plot_binary_auc(
    binary_summary: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot held-out binary AUC against the known-broken baseline (F2)."""

    by_name = _estimator_rows(binary_summary)
    order = ("READ_IG", "READ_local", "weight_norm_baseline")
    missing = set(order) - set(by_name)
    if missing:
        raise ValueError(f"binary summary is missing estimators {sorted(missing)}")
    records = [by_name[name] for name in order]
    labels = ["READ-IG", "READ-local", "Static capacity\n(control)"]
    colors = ["#2166AC", "#1B9E77", "#999999"]
    estimates = np.asarray([row["heldout_auc"] for row in records], dtype=float)
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(8.0, 5.8))
        axis.bar(labels, estimates, color=colors)
        for index, row in enumerate(records):
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            axis.vlines(index, low, high, color="black", linewidth=1.5)
            axis.hlines(
                (low, high),
                index - 0.08,
                index + 0.08,
                color="black",
                linewidth=1.5,
            )
        axis.axhline(
            0.70,
            color="#B2182B",
            linestyle="--",
            linewidth=1.6,
            label="preregistered AUC bar",
        )
        axis.axhline(0.50, color="black", linestyle=":", label="chance")
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("Held-out ROC AUC (group-bootstrap 95% CI)")
        axis.set_title("Binary use-versus-idle discrimination")
        axis.legend(loc="lower right", frameon=False)
        figure.tight_layout()
    return figure, axis


def plot_engine_only_graded_check(
    engine_rows: Sequence[Mapping[str, Any]],
    engine_summary: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot READ-IG against causal magnitude within engines only (F3)."""

    x = np.asarray([row["READ_IG"] for row in engine_rows], dtype=np.float64)
    y = np.asarray([row["abs_C"] for row in engine_rows], dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or np.any(x <= 0.0):
        raise ValueError("engine rows require aligned positive READ_IG and |C|")
    primary = engine_summary["correlations"]["READ_IG"]
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(8.6, 6.3))
        axis.scatter(
            x,
            y,
            s=52,
            alpha=0.72,
            color="#2166AC",
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_xscale("log")
        axis.set_xlabel("READ-IG (16-step midpoint; log scale)")
        axis.set_ylabel("Absolute causal recovery |C|")
        axis.set_title(
            "Graded-use check within causally used concepts\n"
            f"Spearman rho={float(primary['estimate']):.3f}, 95% CI "
            f"[{float(primary['ci95_low']):.3f}, "
            f"{float(primary['ci95_high']):.3f}]"
        )
        axis.text(
            0.02,
            0.04,
            f"N={len(x)} engines; "
            f"{int(engine_summary['n_dependency_groups'])} dependency groups\n"
            "CI spans zero: no positive graded-use evidence",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.9,
            },
        )
        figure.tight_layout()
    return figure, axis


def plot_hard_control_auc(
    binary_summary: Mapping[str, Any],
    hard_summary: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Compare the old and answer-type-matched control AUCs (F4)."""

    old = _estimator_rows(binary_summary)["READ_IG"]
    hard = hard_summary["hard_dashboard_auc"]
    labels = ["Arithmetic\nidle control", "Answer-type-matched\nidle control"]
    estimates = np.asarray([old["heldout_auc"], hard["estimate"]], dtype=float)
    lows = np.asarray([old["ci95_low"], hard["ci95_low"]], dtype=float)
    highs = np.asarray([old["ci95_high"], hard["ci95_high"]], dtype=float)
    errors = np.vstack((estimates - lows, highs - estimates))
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(8.4, 6.2))
        bars = axis.bar(
            labels,
            estimates,
            yerr=errors,
            capsize=7,
            color=["#999999", "#2166AC"],
            edgecolor="black",
            linewidth=0.6,
        )
        axis.axhline(
            0.70,
            color="#B2182B",
            linestyle="--",
            linewidth=1.8,
            label="0.70 evaluation bar",
        )
        axis.axhline(
            0.50,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="chance",
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Held-out ROC AUC (group-bootstrap 95% CI)")
        axis.set_title("READ-IG under a harder matched-answer control")
        axis.legend(loc="lower left", frameon=False)
        for bar, value in zip(bars, estimates, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
            )
        figure.tight_layout()
    return figure, axis


def plot_read_distributions(
    distribution_summary: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot raw READ-IG distributions for used and both idle classes (F5)."""

    raw = distribution_summary["raw_READ_IG"]
    diagnostics = distribution_summary["diagnostics"]
    order = ("engine", "old_dashboard", "hard_dashboard")
    values = [np.asarray(raw[name], dtype=np.float64) for name in order]
    if any(not value.size or np.any(value <= 0.0) for value in values):
        raise ValueError("log-scale READ distributions must be non-empty and positive")
    labels = [
        "Used\n(engine)",
        "Idle\n(arithmetic)",
        "Idle\n(answer-type matched)",
    ]
    colors = ["#2166AC", "#EF8A62", "#67A9CF"]
    positions = np.arange(1, 4)
    rng = np.random.default_rng(PLOT_SEED)
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(9.0, 6.5))
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.6},
        )
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.33)
        for position, group_values, color in zip(
            positions, values, colors, strict=True
        ):
            jitter = rng.normal(0.0, 0.055, size=group_values.size)
            axis.scatter(
                np.full(group_values.size, position) + jitter,
                group_values,
                s=34,
                alpha=0.72,
                color=color,
                edgecolor="white",
                linewidth=0.35,
            )
        axis.set_yscale("log")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Raw READ-IG (log scale)")
        axis.set_title(
            "READ-IG distributions\n"
            "both idle controls occupy an overlapping low-score band"
        )
        classes = diagnostics["classes"]
        axis.text(
            0.02,
            0.04,
            f"used min={float(classes['engine']['minimum']):.4f}\n"
            f"idle maxima: arithmetic="
            f"{float(classes['old_dashboard']['maximum']):.4f}, matched="
            f"{float(classes['hard_dashboard']['maximum']):.4f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.9,
            },
        )
        figure.tight_layout()
    return figure, axis


def plot_directional_agreement(
    task_rows: Sequence[Mapping[str, Any]],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the two signed directions used to form symmetric causal C (F6)."""

    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(7.2, 6.4))
        for task, label, color in (
            ("engine", "Used (engine)", "#2166AC"),
            ("dashboard", "Visible but idle (dashboard)", "#EF8A62"),
        ):
            x = _task_values(task_rows, task, "R_a_from_b")
            y = _task_values(task_rows, task, "R_b_from_a")
            axis.scatter(x, y, label=label, alpha=0.6, s=25, color=color)
        all_recoveries = np.asarray(
            [
                row[key]
                for row in task_rows
                for key in ("R_a_from_b", "R_b_from_a")
            ],
            dtype=np.float64,
        )
        low, high = np.quantile(all_recoveries, [0.01, 0.99])
        padding = max(0.05, 0.08 * float(high - low))
        bounds = (float(low - padding), float(high + padding))
        axis.plot(bounds, bounds, "k--", linewidth=1.2)
        axis.set_xlim(*bounds)
        axis.set_ylim(*bounds)
        axis.set_xlabel("Signed recovery A <- B")
        axis.set_ylabel("Signed recovery B <- A")
        axis.set_title("Directional agreement of symmetric causal recovery")
        axis.legend(frameon=False)
        figure.tight_layout()
    return figure, axis


def generate_final_figures(
    final_metrics: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Render and save the six fixed final figures, returning path metadata."""

    if final_metrics.get("status") != "COMPLETE":
        raise ValueError("final_metrics must be COMPLETE before plotting")
    task_rows = final_metrics["task_rows"]
    old_binary = final_metrics["old_binary"]
    engine_only = final_metrics["engine_only"]
    hard_control = final_metrics["hard_control"]
    distributions = final_metrics["distributions"]
    engine_rows = [row for row in task_rows if row.get("task") == "engine"]
    specifications = (
        (
            "f1_causal_sanity",
            "f1_causal_sanity.png",
            lambda: plot_causal_sanity(task_rows, old_binary["causal_sanity"]),
        ),
        (
            "f2_binary_auc_and_baseline",
            "f2_binary_auc_and_baseline.png",
            lambda: plot_binary_auc(old_binary),
        ),
        (
            "f3_engine_only_graded_check",
            "f3_engine_only_graded_check.png",
            lambda: plot_engine_only_graded_check(engine_rows, engine_only),
        ),
        (
            "f4_hard_dashboard_auc",
            "f4_hard_dashboard_auc.png",
            lambda: plot_hard_control_auc(old_binary, hard_control),
        ),
        (
            "f5_read_ig_distributions",
            "f5_read_ig_distributions.png",
            lambda: plot_read_distributions(distributions),
        ),
        (
            "f6_directional_agreement",
            "f6_directional_agreement.png",
            lambda: plot_directional_agreement(task_rows),
        ),
    )
    destination = Path(output_dir)
    metadata: dict[str, dict[str, Any]] = {}
    for key, filename, builder in specifications:
        figure, _ = builder()
        try:
            path = save_figure(figure, destination / filename)
        finally:
            plt.close(figure)
        metadata[key] = {
            "path": str(path),
            "filename": filename,
            "bytes": int(path.stat().st_size),
            "dpi": PAPER_DPI,
        }
    return metadata

"""Behavior metrics, causal-effect signs, correlations, regressions, and CIs."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def logit_difference(
    logits: torch.Tensor,
    target_token_id: int,
    foil_token_id: int,
    *,
    position: int = -1,
) -> torch.Tensor:
    """Return ``logit(target) - logit(foil)`` in fp32 for every batch item."""

    if logits.ndim != 3:
        raise ValueError(f"Expected [batch, seq, vocab], got {tuple(logits.shape)}")
    if target_token_id == foil_token_id:
        raise ValueError("Target and foil token IDs must differ")
    return (
        logits[:, position, int(target_token_id)].float()
        - logits[:, position, int(foil_token_id)].float()
    )


def signed_causal_delta(clean_metric: float, edited_metric: float) -> float:
    """Canonical effect ``M_edited - M_clean`` used throughout this project."""

    return float(edited_metric - clean_metric)


def support_damage(clean_metric: float, edited_metric: float) -> float:
    """Positive-is-damage companion ``M_clean - M_edited`` for readable plots."""

    return float(clean_metric - edited_metric)


def _finite_vectors(*arrays: Sequence[float]) -> list[np.ndarray]:
    converted = [np.asarray(array, dtype=float).reshape(-1) for array in arrays]
    if not converted or len({len(array) for array in converted}) != 1:
        raise ValueError("All arrays must be nonempty and have identical length")
    mask = np.logical_and.reduce([np.isfinite(array) for array in converted])
    filtered = [array[mask] for array in converted]
    if len(filtered[0]) < 3:
        raise ValueError("At least three finite paired observations are required")
    return filtered


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation after paired finite-value filtering."""

    x_array, y_array = _finite_vectors(x, y)
    if x_array.std() == 0 or y_array.std() == 0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def bootstrap_statistic(
    arrays: Sequence[Sequence[float]],
    statistic: Callable[..., float],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Paired nonparametric bootstrap with the resampling seed persisted."""

    finite = _finite_vectors(*arrays)
    n = len(finite[0])
    estimate = float(statistic(*finite))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selection = rng.integers(0, n, size=n)
        samples[index] = statistic(*(array[selection] for array in finite))
    valid = samples[np.isfinite(samples)]
    if len(valid) < max(100, n_bootstrap // 2):
        raise ValueError("Too few finite bootstrap replicates")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
    return {
        "n": n,
        "estimate": estimate,
        "ci_level": confidence,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def pearson_with_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Pearson ``r`` with a paired bootstrap confidence interval."""

    return bootstrap_statistic(
        [x, y],
        pearson_r,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def _residualize(outcome: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(outcome)), controls])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    return outcome - design @ coefficients


def partial_correlation(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[float] | Sequence[Sequence[float]],
) -> float:
    """Correlation of residuals after linear adjustment for controls."""

    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    control_array = np.asarray(controls, dtype=float)
    if control_array.ndim == 1:
        control_array = control_array[:, None]
    if control_array.ndim != 2 or not (
        len(x_array) == len(y_array) == len(control_array)
    ):
        raise ValueError("x, y, and control rows must align")
    mask = (
        np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(control_array).all(1)
    )
    if mask.sum() < control_array.shape[1] + 3:
        raise ValueError("Insufficient finite rows for partial correlation")
    x_residual = _residualize(x_array[mask], control_array[mask])
    y_residual = _residualize(y_array[mask], control_array[mask])
    return pearson_r(x_residual, y_residual)


def partial_correlation_with_ci(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[float] | Sequence[Sequence[float]],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Partial correlation with paired row-bootstrap confidence interval."""

    control_array = np.asarray(controls, dtype=float)
    if control_array.ndim == 1:
        arrays: list[Sequence[float]] = [x, y, control_array]

        def statistic(a, b, c):
            return partial_correlation(a, b, c)

    else:
        arrays = [
            x,
            y,
            *[control_array[:, column] for column in range(control_array.shape[1])],
        ]

        def statistic(a, b, *columns):
            return partial_correlation(a, b, np.column_stack(columns))

    return bootstrap_statistic(
        arrays,
        statistic,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def standardized_regression(
    causal: Sequence[float],
    write: Sequence[float],
    read: Sequence[float],
    *,
    interaction: bool = False,
) -> dict[str, Any]:
    """Fit standardized ``CAUSAL ~ WRITE + READ [ + WRITE:READ ]`` by OLS."""

    causal_array, write_array, read_array = _finite_vectors(causal, write, read)

    def zscore(array: np.ndarray) -> np.ndarray:
        std = array.std(ddof=0)
        if std == 0:
            raise ValueError("Cannot standardize a constant variable")
        return (array - array.mean()) / std

    y = zscore(causal_array)
    write_z = zscore(write_array)
    read_z = zscore(read_array)
    columns = [np.ones(len(y)), write_z, read_z]
    names = ["intercept", "write", "read"]
    if interaction:
        columns.append(write_z * read_z)
        names.append("write_x_read")
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_total = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - float(np.square(residual).sum()) / ss_total
    return {
        "n": len(y),
        "coefficients": {
            name: float(value) for name, value in zip(names, coefficients, strict=True)
        },
        "r_squared": r_squared,
        "interaction": interaction,
    }


def standardized_regression_with_ci(
    causal: Sequence[float],
    write: Sequence[float],
    read: Sequence[float],
    *,
    interaction: bool = False,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Standardized OLS coefficients with paired row-bootstrap intervals."""

    causal_array, write_array, read_array = _finite_vectors(causal, write, read)
    estimate = standardized_regression(
        causal_array,
        write_array,
        read_array,
        interaction=interaction,
    )
    coefficient_names = list(estimate["coefficients"])
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in coefficient_names}
    r_squared_samples: list[float] = []
    for _ in range(n_bootstrap):
        selection = rng.integers(0, len(causal_array), size=len(causal_array))
        try:
            fitted = standardized_regression(
                causal_array[selection],
                write_array[selection],
                read_array[selection],
                interaction=interaction,
            )
        except ValueError:
            continue
        for name in coefficient_names:
            value = float(fitted["coefficients"][name])
            if np.isfinite(value):
                samples[name].append(value)
        if np.isfinite(fitted["r_squared"]):
            r_squared_samples.append(float(fitted["r_squared"]))

    minimum_valid = max(100, n_bootstrap // 2)
    if min(len(values) for values in samples.values()) < minimum_valid:
        raise ValueError("Too few finite bootstrap regression replicates")
    alpha = (1.0 - confidence) / 2.0
    estimate["coefficient_intervals"] = {
        name: {
            "ci_level": confidence,
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1.0 - alpha)),
        }
        for name, values in samples.items()
    }
    estimate["r_squared_interval"] = {
        "ci_level": confidence,
        "ci_low": float(np.quantile(r_squared_samples, alpha)),
        "ci_high": float(np.quantile(r_squared_samples, 1.0 - alpha)),
    }
    estimate["n_bootstrap"] = n_bootstrap
    estimate["seed"] = seed
    return estimate


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist metrics without allowing NaN/Infinity serialization."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, target)

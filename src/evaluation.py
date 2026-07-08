"""Model-free statistics and validation for final READ evaluation and stress tests.

The functions in this module operate only on already-computed scalar values.
They do not import model, activation-patching, causal-intervention, or READ
implementation code.  Group bootstrap routines resample complete dependency
groups with replacement and use a deterministic ``numpy`` generator.

Public statistical summaries are composed only of JSON-serializable Python
objects.  Bootstrap samples are returned separately as NumPy arrays so callers
can make plots without inflating persisted metrics files.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 1729
DEFAULT_CONFIDENCE = 0.95
DEFAULT_FOLD_ORDER = (0, 1, 2, 3, 4)
DEFAULT_SCORE_KEYS = ("READ_IG", "READ_local", "weight_norm_baseline")


def _finite_float_or_none(value: Any) -> float | None:
    """Return a finite built-in float, or ``None`` for an undefined result."""

    result = float(value)
    return result if math.isfinite(result) else None


def _validate_bootstrap_options(
    *, n_draws: int, seed: int, confidence: float
) -> tuple[int, int, float]:
    """Normalize and validate common deterministic-bootstrap options."""

    if isinstance(n_draws, bool) or int(n_draws) != n_draws or int(n_draws) < 1:
        raise ValueError("n_draws must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    return int(n_draws), int(seed), confidence


def validate_finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str = "values",
    min_size: int = 1,
) -> np.ndarray:
    """Return a copied one-dimensional float array after strict validation.

    Boolean arrays are rejected because silently treating a flag as a numeric
    measurement is usually a schema error.  The returned array is independent
    of the caller's storage and is therefore safe to pass to resampling code.
    """

    if isinstance(min_size, bool) or int(min_size) != min_size or min_size < 0:
        raise ValueError("min_size must be a non-negative integer")
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    if raw.size < int(min_size):
        raise ValueError(f"{name} must contain at least {min_size} values")
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain measurements, not booleans")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain only numeric values") from error
    invalid = np.flatnonzero(~np.isfinite(array))
    if invalid.size:
        preview = invalid[:8].astype(int).tolist()
        raise ValueError(f"{name} contains non-finite values at indices {preview}")
    return array


def validate_aligned_vectors(
    vectors: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    finite_fields: Collection[str] = (),
    min_size: int = 1,
) -> dict[str, Any]:
    """Validate aligned one-dimensional vectors and return a schema summary.

    Fields named in ``finite_fields`` additionally undergo strict numeric and
    finiteness validation.  Other fields may hold group IDs, labels, or other
    scalar metadata, but every vector must be one-dimensional and equally long.
    """

    if not isinstance(vectors, Mapping) or not vectors:
        raise ValueError("vectors must be a non-empty mapping")
    finite_names = {str(field) for field in finite_fields}
    unknown = finite_names.difference(str(key) for key in vectors)
    if unknown:
        raise ValueError(f"finite_fields are absent from vectors: {sorted(unknown)}")

    lengths: dict[str, int] = {}
    for raw_name, values in vectors.items():
        name = str(raw_name)
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
        lengths[name] = int(array.size)
        if name in finite_names:
            validate_finite_vector(array, name=name, min_size=min_size)

    distinct_lengths = set(lengths.values())
    if len(distinct_lengths) != 1:
        raise ValueError(f"vectors are not aligned: {lengths}")
    n_rows = next(iter(distinct_lengths))
    if n_rows < int(min_size):
        raise ValueError(f"vectors must contain at least {min_size} rows")
    return {
        "status": "PASS",
        "n_rows": int(n_rows),
        "fields": sorted(lengths),
        "finite_fields": sorted(finite_names),
    }


def validate_record_schema(
    records: Sequence[Mapping[str, Any]],
    *,
    required_fields: Collection[str],
    finite_fields: Collection[str] = (),
    allowed_values: Mapping[str, Collection[Any]] | None = None,
    unique_by: Sequence[str] = (),
    schema_name: str = "records",
    allow_extra_fields: bool = True,
) -> dict[str, Any]:
    """Validate a flat record schema and return a JSON-serializable audit.

    ``finite_fields`` must contain scalar numeric values.  ``allowed_values``
    provides optional categorical domains.  When ``unique_by`` is non-empty,
    its ordered field tuple must uniquely identify every row.  Extra fields are
    permitted by default; setting ``allow_extra_fields=False`` limits each row
    to the union of fields declared by these arguments.

    The function raises on the first schema class that fails and never drops or
    repairs a row, preventing accidental post-hoc filtering of held-out data.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(f"{schema_name} must be a sequence of mappings")
    required = {str(field) for field in required_fields}
    finite = {str(field) for field in finite_fields}
    domains = {
        str(field): tuple(values)
        for field, values in (allowed_values or {}).items()
    }
    unique_fields = tuple(str(field) for field in unique_by)
    declared = required | finite | set(domains) | set(unique_fields)

    if not required:
        raise ValueError("required_fields must not be empty")
    missing_declarations = (finite | set(domains) | set(unique_fields)) - required
    if missing_declarations:
        raise ValueError(
            "finite, categorical, and unique fields must also be required: "
            f"{sorted(missing_declarations)}"
        )

    seen_keys: set[tuple[Any, ...]] = set()
    observed_fields: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"{schema_name}[{index}] is not a mapping")
        row_fields = {str(field) for field in record}
        observed_fields.update(row_fields)
        missing = required - row_fields
        if missing:
            raise ValueError(
                f"{schema_name}[{index}] is missing required fields {sorted(missing)}"
            )
        if not allow_extra_fields:
            extras = row_fields - declared
            if extras:
                raise ValueError(
                    f"{schema_name}[{index}] has unexpected fields {sorted(extras)}"
                )

        for field in finite:
            value = record[field]
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{schema_name}[{index}].{field} is boolean, not numeric"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"{schema_name}[{index}].{field} is not numeric"
                ) from error
            if not math.isfinite(numeric):
                raise ValueError(
                    f"{schema_name}[{index}].{field} is not finite"
                )

        for field, domain in domains.items():
            if record[field] not in domain:
                raise ValueError(
                    f"{schema_name}[{index}].{field}={record[field]!r} is outside "
                    f"the allowed domain {list(domain)!r}"
                )

        if unique_fields:
            key = tuple(record[field] for field in unique_fields)
            try:
                duplicate = key in seen_keys
            except TypeError as error:
                raise TypeError(
                    f"{schema_name}[{index}] has an unhashable uniqueness key"
                ) from error
            if duplicate:
                raise ValueError(
                    f"{schema_name} has duplicate {unique_fields}: {key!r}"
                )
            seen_keys.add(key)

    return {
        "status": "PASS",
        "schema_name": str(schema_name),
        "n_rows": int(len(records)),
        "required_fields": sorted(required),
        "finite_fields": sorted(finite),
        "categorical_fields": sorted(domains),
        "unique_by": list(unique_fields),
        "allow_extra_fields": bool(allow_extra_fields),
        "observed_fields": sorted(observed_fields),
    }


def _group_blocks(
    groups: Sequence[Any] | np.ndarray, *, expected_size: int
) -> tuple[list[Any], list[np.ndarray]]:
    """Return deterministically ordered group IDs and their row-index blocks."""

    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1 or group_array.size != expected_size:
        raise ValueError(
            f"groups must be one-dimensional with {expected_size} entries"
        )
    positions: dict[Any, list[int]] = {}
    for index, group in enumerate(group_array.tolist()):
        if group is None:
            raise ValueError(f"groups contains None at index {index}")
        try:
            positions.setdefault(group, []).append(index)
        except TypeError as error:
            raise TypeError(f"group at index {index} is not hashable") from error
    if not positions:
        raise ValueError("at least one dependency group is required")
    ordered_groups = sorted(
        positions,
        key=lambda value: (type(value).__name__, repr(value)),
    )
    blocks = [np.asarray(positions[group], dtype=np.int64) for group in ordered_groups]
    return ordered_groups, blocks


def _bootstrap_indices(
    blocks: Sequence[np.ndarray], *, rng: np.random.Generator
) -> np.ndarray:
    """Draw group blocks with replacement and concatenate their row indices."""

    selected = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[int(index)] for index in selected])


def _percentile_interval(
    samples: np.ndarray, *, confidence: float
) -> tuple[float | None, float | None, int]:
    """Return a finite-sample percentile interval and valid sample count."""

    valid = samples[np.isfinite(samples)]
    if not valid.size:
        return None, None, 0
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(valid, [tail, 1.0 - tail])
    return float(low), float(high), int(valid.size)


def group_bootstrap_median(
    records: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    group_key: str = "dependency_group",
    absolute: bool = False,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict[str, Any], np.ndarray]:
    """Estimate a median with a dependency-group percentile interval.

    Complete groups are sampled with replacement, preserving every repeated
    context within a concept dependency group.  ``absolute=True`` applies the
    absolute value before both the observed median and every resampled median.
    The summary is JSON serializable and the separate sample vector is retained
    only for diagnostics.
    """

    n_bootstrap, seed, confidence = _validate_bootstrap_options(
        n_draws=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    columns, groups = _columns_from_records(
        records,
        finite_fields=(value_key,),
        group_key=group_key,
        schema_name=f"grouped_median_{value_key}",
    )
    values = columns[value_key]
    if absolute:
        values = np.abs(values)
    group_names, blocks = _group_blocks(groups, expected_size=int(values.size))
    estimate = float(np.median(values))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        indices = _bootstrap_indices(blocks, rng=rng)
        samples[draw] = float(np.median(values[indices]))
    ci_low, ci_high, valid_draws = _percentile_interval(
        samples,
        confidence=confidence,
    )
    return (
        {
            "statistic": "median_absolute" if absolute else "median",
            "estimate": estimate,
            "confidence": float(confidence),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci95_low": ci_low if confidence == 0.95 else None,
            "ci95_high": ci_high if confidence == 0.95 else None,
            "value_key": str(value_key),
            "absolute": bool(absolute),
            "group_key": str(group_key),
            "n_rows": int(values.size),
            "n_dependency_groups": int(len(group_names)),
            "bootstrap_unit": "dependency_group",
            "bootstrap_draws": int(n_bootstrap),
            "valid_bootstrap_draws": int(valid_draws),
            "undefined_bootstrap_draws": int(n_bootstrap - valid_draws),
            "bootstrap_seed": int(seed),
        },
        samples,
    )


def group_bootstrap_spearman_arrays(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict[str, Any], np.ndarray]:
    """Estimate Spearman rho and its dependency-group bootstrap interval.

    Complete groups are sampled with replacement.  A draw whose resampled
    ``x`` or ``y`` is constant is retained as ``NaN`` in the returned sample
    array and counted as undefined in the serializable summary.  This makes
    small-group degeneracy visible instead of converting it into evidence.
    Repeated calls with the same inputs, seed, and options are deterministic.
    """

    n_draws, seed, confidence = _validate_bootstrap_options(
        n_draws=n_draws, seed=seed, confidence=confidence
    )
    x_array = validate_finite_vector(x, name="x", min_size=2)
    y_array = validate_finite_vector(y, name="y", min_size=2)
    if x_array.size != y_array.size:
        raise ValueError("x and y must have the same number of rows")
    group_names, blocks = _group_blocks(groups, expected_size=int(x_array.size))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed = spearmanr(x_array, y_array)
    observed_rho = _finite_float_or_none(observed.statistic)
    observed_p = _finite_float_or_none(observed.pvalue)

    rng = np.random.default_rng(seed)
    samples = np.full(n_draws, np.nan, dtype=np.float64)
    for draw in range(n_draws):
        indices = _bootstrap_indices(blocks, rng=rng)
        if np.ptp(x_array[indices]) == 0.0 or np.ptp(y_array[indices]) == 0.0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = spearmanr(x_array[indices], y_array[indices]).statistic
        if math.isfinite(float(rho)):
            samples[draw] = float(rho)

    ci_low, ci_high, valid_draws = _percentile_interval(
        samples, confidence=confidence
    )
    summary = {
        "statistic": "spearman_rho",
        "estimate": observed_rho,
        "p_value_descriptive": observed_p,
        "confidence": float(confidence),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci95_low": ci_low if confidence == 0.95 else None,
        "ci95_high": ci_high if confidence == 0.95 else None,
        "n_rows": int(x_array.size),
        "n_dependency_groups": int(len(group_names)),
        "bootstrap_unit": "dependency_group",
        "bootstrap_draws": int(n_draws),
        "valid_bootstrap_draws": int(valid_draws),
        "undefined_bootstrap_draws": int(n_draws - valid_draws),
        "bootstrap_seed": int(seed),
        "ci_lower_above_zero": bool(ci_low is not None and ci_low > 0.0),
    }
    return summary, samples


def group_bootstrap_auc_arrays(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict[str, Any], np.ndarray]:
    """Estimate binary ROC AUC and its dependency-group bootstrap interval.

    Labels must be exactly ``0`` and ``1``.  Complete groups are resampled with
    replacement.  Draws containing only one label are represented by ``NaN``
    and reported as undefined rather than discarded invisibly.  The summary is
    JSON serializable; the separate sample array is intended for diagnostics.
    """

    n_draws, seed, confidence = _validate_bootstrap_options(
        n_draws=n_draws, seed=seed, confidence=confidence
    )
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if raw_labels.dtype.kind not in "biuf":
        raise TypeError("labels must contain numeric binary values")
    if not np.isfinite(raw_labels.astype(np.float64)).all():
        raise ValueError("labels must be finite")
    if not np.all(np.equal(raw_labels, raw_labels.astype(np.int64))):
        raise ValueError("labels must contain integer-valued 0/1 entries")
    label_array = raw_labels.astype(np.int64, copy=True)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")

    score_array = validate_finite_vector(scores, name="scores", min_size=2)
    if score_array.size != label_array.size:
        raise ValueError("labels and scores must have the same number of rows")
    group_names, blocks = _group_blocks(groups, expected_size=int(label_array.size))

    estimate = float(roc_auc_score(label_array, score_array))
    rng = np.random.default_rng(seed)
    samples = np.full(n_draws, np.nan, dtype=np.float64)
    for draw in range(n_draws):
        indices = _bootstrap_indices(blocks, rng=rng)
        sampled_labels = label_array[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        samples[draw] = float(roc_auc_score(sampled_labels, score_array[indices]))

    ci_low, ci_high, valid_draws = _percentile_interval(
        samples, confidence=confidence
    )
    summary = {
        "statistic": "roc_auc",
        "estimate": float(estimate),
        "confidence": float(confidence),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci95_low": ci_low if confidence == 0.95 else None,
        "ci95_high": ci_high if confidence == 0.95 else None,
        "n_rows": int(score_array.size),
        "n_positive": int(np.count_nonzero(label_array == 1)),
        "n_negative": int(np.count_nonzero(label_array == 0)),
        "n_dependency_groups": int(len(group_names)),
        "bootstrap_unit": "dependency_group",
        "bootstrap_draws": int(n_draws),
        "valid_bootstrap_draws": int(valid_draws),
        "undefined_bootstrap_draws": int(n_draws - valid_draws),
        "bootstrap_seed": int(seed),
    }
    return summary, samples


def _columns_from_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    finite_fields: Sequence[str],
    group_key: str,
    schema_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Validate records and extract numeric columns plus one grouping column."""

    required_fields = [*finite_fields, group_key]
    validate_record_schema(
        rows,
        required_fields=required_fields,
        finite_fields=finite_fields,
        schema_name=schema_name,
    )
    columns = {
        field: np.asarray([row[field] for row in rows], dtype=np.float64)
        for field in finite_fields
    }
    groups = np.asarray([row[group_key] for row in rows], dtype=object)
    return columns, groups


def group_bootstrap_spearman(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
    *,
    target_key: str = "abs_C",
    group_key: str = "dependency_group",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 1729,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute row-oriented grouped-bootstrap Spearman rho for the v6 checks.

    This convenience API reads only ``score_key``, ``target_key``, and
    ``group_key`` from each flat record.  The returned summary records those
    column names and is JSON serializable; the separate sample array contains
    one rho per requested draw (with ``NaN`` for undefined draws).
    """

    score_key = str(score_key)
    target_key = str(target_key)
    group_key = str(group_key)
    columns, groups = _columns_from_records(
        rows,
        finite_fields=[score_key, target_key],
        group_key=group_key,
        schema_name="spearman_rows",
    )
    summary, samples = group_bootstrap_spearman_arrays(
        columns[score_key],
        columns[target_key],
        groups,
        n_draws=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    summary.update(
        {
            "score_key": score_key,
            "target_key": target_key,
            "group_key": group_key,
        }
    )
    return summary, samples


def group_bootstrap_auc(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
    *,
    label_key: str = "label",
    group_key: str = "dependency_group",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 1729,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute row-oriented grouped-bootstrap ROC AUC for the v6 checks.

    The score and binary label are extracted from flat held-out records; no
    fitting, threshold selection, or estimator tuning occurs in this helper.
    The summary is JSON serializable and the returned sample array retains
    ``NaN`` entries for bootstrap draws that contain only one class.
    """

    score_key = str(score_key)
    label_key = str(label_key)
    group_key = str(group_key)
    columns, groups = _columns_from_records(
        rows,
        finite_fields=[score_key, label_key],
        group_key=group_key,
        schema_name="auc_rows",
    )
    summary, samples = group_bootstrap_auc_arrays(
        columns[label_key],
        columns[score_key],
        groups,
        n_draws=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    summary.update(
        {
            "score_key": score_key,
            "label_key": label_key,
            "group_key": group_key,
        }
    )
    return summary, samples


def distribution_summary(
    values: Sequence[float] | np.ndarray,
    *,
    name: str = "distribution",
    near_zero_atol: float = 1e-8,
) -> dict[str, Any]:
    """Describe a finite scalar distribution without fitting a model.

    The exact observed range, quartiles, population standard deviation, median
    absolute deviation, unique-value count, and explicitly parameterized
    near-zero fraction make collapsed dashboard distributions easy to audit.
    """

    atol = float(near_zero_atol)
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("near_zero_atol must be finite and non-negative")
    array = validate_finite_vector(values, name=name, min_size=1)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    span = maximum - minimum
    return {
        "name": str(name),
        "n": int(array.size),
        "minimum": minimum,
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "maximum": maximum,
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
        "median_absolute_deviation": float(np.median(np.abs(array - median))),
        "iqr": float(q3 - q1),
        "range_width": float(span),
        "n_unique": int(np.unique(array).size),
        "all_identical": bool(span == 0.0),
        "near_zero_atol": atol,
        "n_near_zero": int(np.count_nonzero(np.abs(array) <= atol)),
        "fraction_near_zero": float(np.mean(np.abs(array) <= atol)),
    }


def range_overlap_diagnostics(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    *,
    name_a: str = "a",
    name_b: str = "b",
    near_zero_atol: float = 1e-8,
) -> dict[str, Any]:
    """Report observed-range overlap and gap diagnostics for two classes.

    ``range_gap`` is zero when ranges overlap or touch and otherwise gives the
    empty distance between them.  Fractions inside the other class's observed
    range expose graded overlap even when one class has a much narrower span.
    No distributional assumptions or fitted thresholds are used.
    """

    if str(name_a) == str(name_b):
        raise ValueError("distribution names must be distinct")
    array_a = validate_finite_vector(values_a, name=name_a, min_size=1)
    array_b = validate_finite_vector(values_b, name=name_b, min_size=1)
    summary_a = distribution_summary(
        array_a, name=name_a, near_zero_atol=near_zero_atol
    )
    summary_b = distribution_summary(
        array_b, name=name_b, near_zero_atol=near_zero_atol
    )

    lo_a, hi_a = summary_a["minimum"], summary_a["maximum"]
    lo_b, hi_b = summary_b["minimum"], summary_b["maximum"]
    intersection_low = max(lo_a, lo_b)
    intersection_high = min(hi_a, hi_b)
    signed_overlap = intersection_high - intersection_low
    ranges_overlap = signed_overlap >= 0.0
    overlap_width = max(0.0, signed_overlap)
    union_width = max(hi_a, hi_b) - min(lo_a, lo_b)
    gap = max(0.0, -signed_overlap)

    if hi_a < lo_b:
        ordering = f"{name_b}_above_{name_a}"
    elif hi_b < lo_a:
        ordering = f"{name_a}_above_{name_b}"
    elif signed_overlap == 0.0:
        ordering = "touching"
    else:
        ordering = "overlap"

    return {
        "class_a": summary_a,
        "class_b": summary_b,
        "ranges_overlap_or_touch": bool(ranges_overlap),
        "strictly_disjoint_ranges": bool(signed_overlap < 0.0),
        "ordering": ordering,
        "intersection": (
            [float(intersection_low), float(intersection_high)]
            if ranges_overlap
            else None
        ),
        "overlap_width": float(overlap_width),
        "range_gap": float(gap),
        "union_width": float(union_width),
        "overlap_fraction_of_union": (
            float(overlap_width / union_width)
            if union_width > 0.0
            else 1.0
        ),
        "fraction_a_inside_b_range": float(
            np.mean((array_a >= lo_b) & (array_a <= hi_b))
        ),
        "fraction_b_inside_a_range": float(
            np.mean((array_b >= lo_a) & (array_b <= hi_a))
        ),
        "absolute_median_gap": float(
            abs(summary_a["median"] - summary_b["median"])
        ),
    }


def distribution_overlap(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    *,
    name_a: str = "a",
    name_b: str = "b",
    near_zero_atol: float = 1e-8,
) -> dict[str, Any]:
    """Return the pairwise range/overlap diagnostic under a concise API name."""

    return range_overlap_diagnostics(
        values_a,
        values_b,
        name_a=name_a,
        name_b=name_b,
        near_zero_atol=near_zero_atol,
    )


def distribution_diagnostics(
    distributions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    near_zero_atol: float = 1e-8,
) -> dict[str, Any]:
    """Summarize multiple raw distributions and every pairwise range overlap."""

    if not isinstance(distributions, Mapping) or len(distributions) < 2:
        raise ValueError("distributions must contain at least two named classes")
    names = sorted(str(name) for name in distributions)
    if len(names) != len(set(names)):
        raise ValueError("distribution names must remain unique when stringified")
    by_name = {str(name): values for name, values in distributions.items()}
    classes = {
        name: distribution_summary(
            by_name[name], name=name, near_zero_atol=near_zero_atol
        )
        for name in names
    }
    pairwise = {}
    for name_a, name_b in itertools.combinations(names, 2):
        key = f"{name_a}__vs__{name_b}"
        pairwise[key] = range_overlap_diagnostics(
            by_name[name_a],
            by_name[name_b],
            name_a=name_a,
            name_b=name_b,
            near_zero_atol=near_zero_atol,
        )
    return {
        "n_classes": int(len(names)),
        "classes": classes,
        "pairwise": pairwise,
    }


def _rows_from_artifact(
    artifact: Mapping[str, Any], *, artifact_name: str, key: str = "rows"
) -> Sequence[Mapping[str, Any]]:
    """Return a record sequence from an in-memory artifact without file I/O."""

    if not isinstance(artifact, Mapping):
        raise TypeError(f"{artifact_name} must be a mapping")
    rows = artifact.get(key)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{artifact_name}.{key} must be a sequence of records")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{artifact_name}.{key} must contain only mappings")
    return rows


def _index_unique_rows(
    rows: Sequence[Mapping[str, Any]], *, key: str, source_name: str
) -> dict[Any, Mapping[str, Any]]:
    """Index records by a required hashable key and reject duplicates."""

    indexed: dict[Any, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if key not in row:
            raise ValueError(f"{source_name}[{index}] is missing {key!r}")
        value = row[key]
        try:
            duplicate = value in indexed
        except TypeError as error:
            raise TypeError(
                f"{source_name}[{index}].{key} must be hashable"
            ) from error
        if duplicate:
            raise ValueError(f"{source_name} has duplicate {key}={value!r}")
        indexed[value] = row
    return indexed


def _require_matching_metadata(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    fields: Sequence[str],
    context: str,
) -> None:
    """Require duplicated join metadata to agree exactly."""

    mismatches = {
        field: (left.get(field), right.get(field))
        for field in fields
        if left.get(field) != right.get(field)
    }
    if mismatches:
        raise ValueError(f"{context} metadata mismatch: {mismatches}")


def _fold_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    label_key: str = "label",
    fold_key: str = "fold",
    fold_order: Sequence[int] = DEFAULT_FOLD_ORDER,
) -> dict[str, float]:
    """Compute AUC in the frozen fold order, requiring both classes per fold."""

    expected = tuple(int(fold) for fold in fold_order)
    if len(expected) != len(set(expected)):
        raise ValueError("fold_order must not contain duplicates")
    observed = {int(row[fold_key]) for row in rows}
    if observed != set(expected):
        raise ValueError(
            f"observed folds {sorted(observed)} do not equal {list(expected)}"
        )
    result: dict[str, float] = {}
    for fold in expected:
        subset = [row for row in rows if int(row[fold_key]) == fold]
        labels = np.asarray([row[label_key] for row in subset], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"fold {fold} does not contain both binary classes")
        scores = np.asarray([row[score_key] for row in subset], dtype=np.float64)
        result[str(fold)] = float(roc_auc_score(labels, scores))
    return result


def join_base_causal_and_cheap(
    causal_artifact: Mapping[str, Any],
    cheap_artifact: Mapping[str, Any],
    *,
    require_firewall_pass: bool = True,
) -> list[dict[str, Any]]:
    """Join frozen causal truth and cheap READ estimates by ``pair_id``.

    The join is deliberately model-free.  It requires exact pair coverage,
    agrees duplicated group/fold/category metadata, and emits two flat rows per
    pair in the stable order ``engine`` then ``dashboard``.  The final pipeline
    intentionally retains only signed, unclipped full-residual interchange as
    causal truth; superseded J-Lens-subspace diagnostics live in the archive.
    """

    causal_protocol = causal_artifact.get("protocol_sha256")
    cheap_protocol = cheap_artifact.get("protocol_sha256")
    if causal_protocol is not None and cheap_protocol is not None:
        if causal_protocol != cheap_protocol:
            raise ValueError("causal and cheap artifacts use different protocols")
    if require_firewall_pass:
        audit = cheap_artifact.get("anti_circularity_audit")
        if not isinstance(audit, Mapping) or audit.get("status") != "PASS":
            raise ValueError("cheap READ anti-circularity audit did not pass")

    causal_rows = _rows_from_artifact(causal_artifact, artifact_name="causal")
    cheap_rows = _rows_from_artifact(cheap_artifact, artifact_name="cheap")
    causal_by_pair = _index_unique_rows(
        causal_rows, key="pair_id", source_name="causal.rows"
    )
    cheap_by_pair = _index_unique_rows(
        cheap_rows, key="pair_id", source_name="cheap.rows"
    )
    if set(causal_by_pair) != set(cheap_by_pair):
        missing_cheap = sorted(set(causal_by_pair) - set(cheap_by_pair))
        missing_causal = sorted(set(cheap_by_pair) - set(causal_by_pair))
        raise ValueError(
            "causal and cheap pair coverage differs: "
            f"missing_cheap={missing_cheap}, missing_causal={missing_causal}"
        )

    joined: list[dict[str, Any]] = []
    for pair_id in sorted(causal_by_pair, key=str):
        causal_pair = causal_by_pair[pair_id]
        cheap_pair = cheap_by_pair[pair_id]
        _require_matching_metadata(
            causal_pair,
            cheap_pair,
            fields=("dependency_group", "fold", "category"),
            context=f"pair {pair_id}",
        )
        baseline = cheap_pair.get("weight_norm_capacity_baseline")
        if not isinstance(baseline, Mapping):
            raise ValueError(f"pair {pair_id} has no capacity-baseline record")

        for task, label in (("engine", 1), ("dashboard", 0)):
            truth_bucket = causal_pair.get(task)
            estimate = cheap_pair.get(task)
            if not isinstance(truth_bucket, Mapping) or not isinstance(
                estimate, Mapping
            ):
                raise ValueError(f"pair {pair_id} has no {task} result")
            truth = truth_bucket.get("full_residual")
            if not isinstance(truth, Mapping):
                raise ValueError(f"pair {pair_id} {task} full-residual truth is absent")

            joined.append(
                {
                    "pair_id": pair_id,
                    "dependency_group": causal_pair["dependency_group"],
                    "fold": int(causal_pair["fold"]),
                    "category": causal_pair["category"],
                    "concept_a": causal_pair.get("concept_a"),
                    "concept_b": causal_pair.get("concept_b"),
                    "task": task,
                    "label": label,
                    "C": float(truth["C"]),
                    "abs_C": abs(float(truth["C"])),
                    "R_a_from_b": float(truth["R_a_from_b"]),
                    "R_b_from_a": float(truth["R_b_from_a"]),
                    "T": float(truth["T"]),
                    "directional_abs_difference": float(
                        truth["directional_abs_difference"]
                    ),
                    "sharp_directional_disagreement": bool(
                        truth["sharp_directional_disagreement"]
                    ),
                    "READ_IG": float(estimate["READ_IG"]),
                    "READ_local": float(estimate["READ_local"]),
                    "weight_norm_baseline": float(
                        baseline["weight_norm_baseline"]
                    ),
                    "baseline_label": baseline.get("baseline"),
                }
            )

    validate_record_schema(
        joined,
        required_fields={
            "pair_id",
            "dependency_group",
            "fold",
            "category",
            "task",
            "label",
            "C",
            "abs_C",
            "R_a_from_b",
            "R_b_from_a",
            "T",
            "directional_abs_difference",
            "sharp_directional_disagreement",
            "READ_IG",
            "READ_local",
            "weight_norm_baseline",
        },
        finite_fields={
            "label",
            "C",
            "abs_C",
            "R_a_from_b",
            "R_b_from_a",
            "T",
            "directional_abs_difference",
            "READ_IG",
            "READ_local",
            "weight_norm_baseline",
        },
        allowed_values={"task": {"engine", "dashboard"}, "label": {0, 1}},
        unique_by=("pair_id", "task"),
        schema_name="base_task_rows",
    )
    return joined


def evaluate_old_binary_detection(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    score_keys: Sequence[str] = DEFAULT_SCORE_KEYS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    fold_order: Sequence[int] = DEFAULT_FOLD_ORDER,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the original engine-versus-dashboard binary discrimination.

    AUC confidence intervals resample complete dependency groups.  Fold AUCs
    are emitted in the frozen order 0--4.  Overall Spearman correlations are
    descriptive and do not replace the later engine-only graded-use check.
    """

    rows = list(task_rows)
    if not rows:
        raise ValueError("task_rows must not be empty")
    estimators = tuple(str(key) for key in score_keys)
    if not estimators or len(set(estimators)) != len(estimators):
        raise ValueError("score_keys must contain unique estimator names")

    auc_table: list[dict[str, Any]] = []
    samples_by_estimator: dict[str, np.ndarray] = {}
    for score_key in estimators:
        summary, samples = group_bootstrap_auc(
            rows,
            score_key,
            label_key="label",
            group_key="dependency_group",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        scores = validate_finite_vector(
            [row[score_key] for row in rows], name=score_key, min_size=2
        )
        abs_c = validate_finite_vector(
            [row["abs_C"] for row in rows], name="abs_C", min_size=2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation = spearmanr(scores, abs_c)
        heldout_auc = float(summary["estimate"])
        ci_low = summary["ci95_low"]
        ci_high = summary["ci95_high"]
        record = {
            **summary,
            "estimator": score_key,
            "heldout_auc": heldout_auc,
            "fold_auc": _fold_auc(
                rows,
                score_key=score_key,
                fold_order=fold_order,
            ),
            "n_prompt_pairs": int(len(rows) // 2),
            "passes_numeric_bar": bool(
                heldout_auc >= 0.70 and ci_low is not None and ci_low > 0.50
            ),
            "eligible_to_trigger_go": score_key == "READ_IG",
            "spearman_rho_with_abs_C": _finite_float_or_none(
                correlation.statistic
            ),
            "spearman_p_value_descriptive": _finite_float_or_none(
                correlation.pvalue
            ),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_unit": "unordered concept dependency group",
        }
        auc_table.append(record)
        samples_by_estimator[score_key] = samples

    primary = next(
        (row for row in auc_table if row["estimator"] == "READ_IG"), None
    )
    if primary is None:
        raise ValueError("READ_IG must be present in score_keys")
    decision = "GO" if primary["passes_numeric_bar"] else "NO-GO"
    if decision == "GO":
        decision_one_line = (
            "GO: READ_IG predicts causal use on held-out Qwen2.5-7B concepts "
            f"(AUC={primary['heldout_auc']:.3f}, 95% CI "
            f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}])."
        )
    else:
        decision_one_line = (
            "NO-GO: on Qwen2.5-7B, gradient READ_IG does not clear the "
            "pre-registered held-out causal-use bar "
            f"(AUC={primary['heldout_auc']:.3f}, 95% CI "
            f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}])."
        )

    engine_rows = [row for row in rows if row["task"] == "engine"]
    dashboard_rows = [row for row in rows if row["task"] == "dashboard"]
    if not engine_rows or not dashboard_rows:
        raise ValueError("task_rows must contain both engine and dashboard rows")
    median_intervals = {
        "engine_C": group_bootstrap_median(
            engine_rows,
            "C",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
        "engine_abs_C": group_bootstrap_median(
            engine_rows,
            "C",
            absolute=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
        "dashboard_C": group_bootstrap_median(
            dashboard_rows,
            "C",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
        "dashboard_abs_C": group_bootstrap_median(
            dashboard_rows,
            "C",
            absolute=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
    }
    sanity = {
        "n_pairs": int(len(engine_rows)),
        "engine_C_median": float(np.median([row["C"] for row in engine_rows])),
        "engine_abs_C_median": float(
            np.median([row["abs_C"] for row in engine_rows])
        ),
        "dashboard_C_median": float(
            np.median([row["C"] for row in dashboard_rows])
        ),
        "dashboard_abs_C_median": float(
            np.median([row["abs_C"] for row in dashboard_rows])
        ),
        "engine_sharp_directional_disagreements": int(
            sum(bool(row["sharp_directional_disagreement"]) for row in engine_rows)
        ),
        "dashboard_sharp_directional_disagreements": int(
            sum(
                bool(row["sharp_directional_disagreement"])
                for row in dashboard_rows
            )
        ),
        "median_intervals": median_intervals,
    }
    return (
        {
            "status": "COMPLETE",
            "auc_table": auc_table,
            "causal_sanity": sanity,
            "decision": decision,
            "decision_one_line": decision_one_line,
        },
        samples_by_estimator,
    )


def evaluate_engine_only(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    score_keys: Sequence[str] = DEFAULT_SCORE_KEYS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the decisive graded-use check on engine rows only.

    No weak/strong AUC is invented: the frozen protocol supplied no cutoff and
    the observed engines occupy a narrow, already-causal range.  The primary
    criterion is whether READ_IG's group-bootstrap Spearman lower bound is
    strictly above zero.
    """

    engine_rows = [dict(row) for row in task_rows if row.get("task") == "engine"]
    if not engine_rows:
        raise ValueError("no engine rows are available")
    correlations: dict[str, dict[str, Any]] = {}
    samples_by_estimator: dict[str, np.ndarray] = {}
    for score_key in tuple(str(key) for key in score_keys):
        summary, samples = group_bootstrap_spearman(
            engine_rows,
            score_key,
            target_key="abs_C",
            group_key="dependency_group",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        correlations[score_key] = summary
        samples_by_estimator[score_key] = samples

    if "READ_IG" not in correlations:
        raise ValueError("READ_IG must be present in score_keys")
    abs_c = np.sort(
        validate_finite_vector(
            [row["abs_C"] for row in engine_rows], name="engine_abs_C", min_size=2
        )
    )
    gaps = np.diff(abs_c)
    largest_index = int(np.argmax(gaps))
    within_engine_auc = {
        "status": "NOT_RUN_NO_PREREGISTERED_OR_NATURAL_SPLIT",
        "reason": (
            "All frozen engines are strongly causal and the protocol defines no "
            "weak/strong cutoff; introducing one after seeing outcomes would be "
            "post-hoc."
        ),
        "abs_C_min": float(abs_c[0]),
        "abs_C_median": float(np.median(abs_c)),
        "abs_C_max": float(abs_c[-1]),
        "largest_adjacent_gap": float(gaps[largest_index]),
        "rows_below_largest_gap": int(largest_index + 1),
    }
    supports = bool(correlations["READ_IG"]["ci_lower_above_zero"])
    return (
        {
            "status": "COMPLETE",
            "n_engines": int(len(engine_rows)),
            "n_dependency_groups": int(
                len({row["dependency_group"] for row in engine_rows})
            ),
            "correlations": correlations,
            "within_engine_auc": within_engine_auc,
            "supports_positive_graded_use": supports,
            "interpretation": (
                "SUPPORTS_GRADED_USE"
                if supports
                else "ARTIFACT_SIDE_CI_SPANS_ZERO"
            ),
        },
        samples_by_estimator,
    )


def join_hard_control_artifacts(
    base_task_rows: Sequence[Mapping[str, Any]],
    hard_manifest: Mapping[str, Any],
    hard_cheap_artifact: Mapping[str, Any],
    hard_causal_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Join verified hard controls to their frozen matched engine rows.

    Failed hard controls remain counted but are never relabelled or included in
    evaluation.  Cheap and causal hard-control coverage must exactly equal the
    set marked ``VERIFIED_HARD`` by the independent manifest.
    """

    manifest_rows = _rows_from_artifact(
        hard_manifest, artifact_name="hard_manifest"
    )
    verified = [
        row
        for row in manifest_rows
        if row.get("verification_status") == "VERIFIED_HARD"
    ]
    verified_ids = {row["pair_id"] for row in verified}
    if not verified_ids:
        raise ValueError("no VERIFIED_HARD controls are available")

    cheap_key = (
        "compact_rows" if "compact_rows" in hard_cheap_artifact else "rows"
    )
    cheap_rows = _rows_from_artifact(
        hard_cheap_artifact, artifact_name="hard_cheap", key=cheap_key
    )
    causal_rows = _rows_from_artifact(
        hard_causal_artifact, artifact_name="hard_causal"
    )
    cheap_by_pair = _index_unique_rows(
        cheap_rows, key="pair_id", source_name=f"hard_cheap.{cheap_key}"
    )
    causal_by_pair = _index_unique_rows(
        causal_rows, key="pair_id", source_name="hard_causal.rows"
    )
    if set(cheap_by_pair) != verified_ids or set(causal_by_pair) != verified_ids:
        raise ValueError(
            "hard cheap/causal coverage must exactly equal VERIFIED_HARD coverage"
        )
    engine_by_pair = _index_unique_rows(
        [row for row in base_task_rows if row.get("task") == "engine"],
        key="pair_id",
        source_name="base_engine_rows",
    )
    missing_engines = sorted(verified_ids - set(engine_by_pair), key=str)
    if missing_engines:
        raise ValueError(f"hard controls lack matched engines: {missing_engines}")

    task_rows: list[dict[str, Any]] = []
    causal_joined: list[dict[str, Any]] = []
    for pair_id in sorted(verified_ids, key=str):
        engine = engine_by_pair[pair_id]
        hard = cheap_by_pair[pair_id]
        causal = causal_by_pair[pair_id]
        _require_matching_metadata(
            engine,
            hard,
            fields=("dependency_group", "fold", "category"),
            context=f"hard pair {pair_id}",
        )
        _require_matching_metadata(
            engine,
            causal,
            fields=("dependency_group", "fold", "category"),
            context=f"hard causal pair {pair_id}",
        )
        hard_score = hard.get("READ_IG")
        if hard_score is None and isinstance(hard.get("hard_dashboard"), Mapping):
            hard_score = hard["hard_dashboard"].get("READ_IG")
        if hard_score is None:
            raise ValueError(f"hard pair {pair_id} has no READ_IG score")
        task_rows.extend(
            (
                {
                    "pair_id": pair_id,
                    "dependency_group": engine["dependency_group"],
                    "fold": int(engine["fold"]),
                    "category": engine["category"],
                    "task": "engine",
                    "label": 1,
                    "READ_IG": float(engine["READ_IG"]),
                },
                {
                    "pair_id": pair_id,
                    "dependency_group": engine["dependency_group"],
                    "fold": int(engine["fold"]),
                    "category": engine["category"],
                    "task": "hard_dashboard",
                    "label": 0,
                    "READ_IG": float(hard_score),
                },
            )
        )
        hard_truth = causal.get("hard_dashboard")
        if not isinstance(hard_truth, Mapping):
            raise ValueError(f"hard pair {pair_id} has no causal truth")
        causal_joined.append(
            {
                "pair_id": pair_id,
                "dependency_group": engine["dependency_group"],
                "fold": int(engine["fold"]),
                "category": engine["category"],
                "engine_C": float(causal["frozen_engine_C"]),
                "hard_dashboard_C": float(hard_truth["C"]),
                "hard_dashboard_R_a_from_b": float(hard_truth["R_a_from_b"]),
                "hard_dashboard_R_b_from_a": float(hard_truth["R_b_from_a"]),
                "hard_dashboard_sharp_directional_disagreement": bool(
                    hard_truth["sharp_directional_disagreement"]
                ),
            }
        )

    status_counts = Counter(
        str(row.get("verification_status")) for row in manifest_rows
    )
    reason_counts = Counter(
        str(reason)
        for row in manifest_rows
        for reason in row.get("verification_reasons", ())
    )
    return {
        "task_rows": task_rows,
        "causal_rows": causal_joined,
        "verification": {
            "candidates": int(len(manifest_rows)),
            "verified_hard": int(status_counts["VERIFIED_HARD"]),
            "unverified_hard": int(status_counts["UNVERIFIED_HARD"]),
            "reason_counts": dict(sorted(reason_counts.items())),
            "n_dependency_groups": int(
                len({row["dependency_group"] for row in verified})
            ),
            "failed_rows_excluded_not_relabeled": True,
        },
    }


def evaluate_hard_control(
    joined: Mapping[str, Any],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    fold_order: Sequence[int] = DEFAULT_FOLD_ORDER,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate READ_IG against verified answer-type-matched idle controls."""

    task_rows = joined.get("task_rows")
    causal_rows = joined.get("causal_rows")
    if not isinstance(task_rows, Sequence) or not isinstance(causal_rows, Sequence):
        raise TypeError("joined hard-control data has no task/causal row sequences")
    auc, samples = group_bootstrap_auc(
        task_rows,
        "READ_IG",
        label_key="label",
        group_key="dependency_group",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    auc["fold_auc"] = _fold_auc(
        task_rows,
        score_key="READ_IG",
        fold_order=fold_order,
    )
    auc["bootstrap_unit"] = "unordered concept dependency group"

    category_auc: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in task_rows}):
        subset = [row for row in task_rows if str(row["category"]) == category]
        category_auc[category] = {
            "n_pairs": int(len(subset) // 2),
            "n_dependency_groups": int(
                len({row["dependency_group"] for row in subset})
            ),
            "auc": float(
                roc_auc_score(
                    [row["label"] for row in subset],
                    [row["READ_IG"] for row in subset],
                )
            ),
            "diagnostic_only": True,
        }

    engine_c = validate_finite_vector(
        [row["engine_C"] for row in causal_rows], name="hard_engine_C"
    )
    hard_c = validate_finite_vector(
        [row["hard_dashboard_C"] for row in causal_rows],
        name="hard_dashboard_C",
    )
    median_intervals = {
        "engine_C": group_bootstrap_median(
            causal_rows,
            "engine_C",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
        "engine_abs_C": group_bootstrap_median(
            causal_rows,
            "engine_C",
            absolute=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
        "hard_dashboard_C": group_bootstrap_median(
            causal_rows,
            "hard_dashboard_C",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
        "hard_dashboard_abs_C": group_bootstrap_median(
            causal_rows,
            "hard_dashboard_C",
            absolute=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )[0],
    }
    sanity = {
        "n_pairs": int(len(causal_rows)),
        "engine_C_median": float(np.median(engine_c)),
        "engine_abs_C_median": float(np.median(np.abs(engine_c))),
        "hard_dashboard_C_median": float(np.median(hard_c)),
        "hard_dashboard_abs_C_median": float(np.median(np.abs(hard_c))),
        "hard_dashboard_C_min": float(np.min(hard_c)),
        "hard_dashboard_C_max": float(np.max(hard_c)),
        "hard_dashboard_sharp_directional_disagreements": int(
            sum(
                bool(row["hard_dashboard_sharp_directional_disagreement"])
                for row in causal_rows
            )
        ),
        "median_intervals": median_intervals,
        "engine_large_gate": bool(np.median(np.abs(engine_c)) > 0.50),
        "hard_dashboard_near_zero_gate": bool(np.median(np.abs(hard_c)) < 0.10),
    }
    sanity["status"] = (
        "PASS"
        if sanity["engine_large_gate"] and sanity["hard_dashboard_near_zero_gate"]
        else "FAIL"
    )
    survives = bool(sanity["status"] == "PASS" and auc["estimate"] >= 0.80)
    return (
        {
            "status": "COMPLETE",
            "verification": dict(joined.get("verification", {})),
            "causal_sanity": sanity,
            "hard_dashboard_auc": auc,
            "category_auc_diagnostic": category_auc,
            "hard_auc_survives_at_0_80": survives,
            "interpretation": (
                "HARD_CONTROL_SEPARATION_SURVIVES"
                if survives
                else "HARD_CONTROL_SEPARATION_COLLAPSES_OR_CAUSAL_SANITY_FAILS"
            ),
            "task_rows": list(task_rows),
            "causal_rows": list(causal_rows),
        },
        samples,
    )


def make_final_decision(
    old_binary: Mapping[str, Any],
    engine_only: Mapping[str, Any],
    hard_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen two-check rule and emit the exact final verdict."""

    correlations = engine_only.get("correlations")
    if not isinstance(correlations, Mapping) or not isinstance(
        correlations.get("READ_IG"), Mapping
    ):
        raise ValueError("engine_only has no READ_IG correlation")
    primary = correlations["READ_IG"]
    graded = bool(engine_only.get("supports_positive_graded_use"))
    hard_survives = bool(hard_control.get("hard_auc_survives_at_0_80"))
    confirmed = graded and hard_survives

    if confirmed:
        label = "CONFIRMED"
        one_line = (
            "CONFIRMED: engine-only READ_IG is clearly positive and the "
            "separation survives the answer-type-matched hard control."
        )
    elif hard_survives and not graded:
        label = "ARTIFACT (partial)"
        one_line = (
            "ARTIFACT (partial): READ_IG survives the answer-type-matched "
            "control, but has no positive graded association within engines "
            f"(rho={primary['estimate']:.3f}, 95% CI "
            f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}]); the "
            "perfect binary separation is not evidence of a graded causal-use "
            "meter."
        )
    elif graded:
        label = "ARTIFACT (partial)"
        one_line = (
            "ARTIFACT (partial): engine-only READ_IG is positively associated "
            "with causal magnitude, but binary separation does not survive the "
            "answer-type-matched hard control."
        )
    else:
        label = "ARTIFACT (partial)"
        one_line = (
            "ARTIFACT (partial): neither a positive engine-only graded "
            "association nor robust separation from the answer-type-matched "
            "hard control was established."
        )

    old_primary = next(
        (
            row
            for row in old_binary.get("auc_table", ())
            if row.get("estimator") == "READ_IG"
        ),
        None,
    )
    old_supported = bool(old_primary and old_primary.get("passes_numeric_bar"))
    return {
        "decision": label,
        "decision_one_line": one_line,
        "binary_detector": (
            "SUPPORTED" if old_supported and hard_survives else "NOT_SUPPORTED"
        ),
        "graded_meter": "SUPPORTED" if graded else "NOT_SUPPORTED",
        "stress_test_label": label,
        "decision_rule": {
            "confirmed_requires_both": True,
            "engine_only_ci_lower_above_zero": graded,
            "hard_control_auc_survives": hard_survives,
        },
    }


def _read_distribution_result(
    base_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build matched raw READ_IG distributions and mechanism diagnostics."""

    engine = {
        row["pair_id"]: float(row["READ_IG"])
        for row in base_rows
        if row.get("task") == "engine"
    }
    old = {
        row["pair_id"]: float(row["READ_IG"])
        for row in base_rows
        if row.get("task") == "dashboard"
    }
    hard = {
        row["pair_id"]: float(row["READ_IG"])
        for row in hard_rows
        if row.get("task") == "hard_dashboard"
    }
    pair_ids = sorted(hard, key=str)
    if not pair_ids or not set(pair_ids) <= set(engine) or not set(pair_ids) <= set(old):
        raise ValueError("hard controls are not matched to base engine/dashboard rows")
    raw = {
        "engine": [engine[pair_id] for pair_id in pair_ids],
        "old_dashboard": [old[pair_id] for pair_id in pair_ids],
        "hard_dashboard": [hard[pair_id] for pair_id in pair_ids],
    }
    diagnostics = distribution_diagnostics(raw, near_zero_atol=1e-3)
    engine_summary = diagnostics["classes"]["engine"]
    old_summary = diagnostics["classes"]["old_dashboard"]
    hard_summary = diagnostics["classes"]["hard_dashboard"]
    engine_old = diagnostics["pairwise"]["engine__vs__old_dashboard"]
    engine_hard = diagnostics["pairwise"]["engine__vs__hard_dashboard"]
    old_hard = diagnostics["pairwise"]["hard_dashboard__vs__old_dashboard"]
    mechanism = {
        "old_dashboard_all_identical": old_summary["all_identical"],
        "old_dashboard_compressed_tiny_band": bool(
            old_summary["maximum"] < engine_summary["minimum"]
            and old_summary["median"] < 0.01
        ),
        "hard_dashboard_compressed_tiny_band": bool(
            hard_summary["maximum"] < engine_summary["minimum"]
            and hard_summary["median"] < 0.01
        ),
        "old_and_hard_dashboard_ranges_overlap": old_hard[
            "ranges_overlap_or_touch"
        ],
        "old_and_hard_overlap_fraction_of_union": old_hard[
            "overlap_fraction_of_union"
        ],
        "engine_old_ranges_disjoint": engine_old["strictly_disjoint_ranges"],
        "engine_hard_ranges_disjoint": engine_hard["strictly_disjoint_ranges"],
        "arithmetic_answer_type_is_sole_explanation": False,
        "interpretation": (
            "Both old and answer-type-matched hard dashboards occupy an "
            "overlapping low-READ band that is disjoint from engines. READ_IG "
            "behaves as a binary relevant-versus-idle detector on this roster, "
            "without demonstrated graded resolution within engines."
        ),
    }
    return {
        "status": "COMPLETE",
        "pair_ids": pair_ids,
        "raw_READ_IG": raw,
        "diagnostics": diagnostics,
        "mechanism_finding": mechanism,
    }


def assemble_final_metrics(
    verification_artifact: Mapping[str, Any],
    causal_artifact: Mapping[str, Any],
    cheap_artifact: Mapping[str, Any],
    hard_manifest: Mapping[str, Any],
    hard_cheap_artifact: Mapping[str, Any],
    hard_causal_artifact: Mapping[str, Any],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    fold_order: Sequence[int] = DEFAULT_FOLD_ORDER,
) -> dict[str, Any]:
    """Assemble the complete JSON-ready final result from frozen artifacts.

    This is the single model-free entry point used by the final trust-check and
    report notebooks.  Bootstrap samples are intentionally omitted from the
    compact metrics object; every published estimate, interval, fold result,
    causal sanity check, raw distribution, and decision remains included.
    """

    base_rows = join_base_causal_and_cheap(causal_artifact, cheap_artifact)
    old_binary, _ = evaluate_old_binary_detection(
        base_rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
        fold_order=fold_order,
    )
    engine_only, _ = evaluate_engine_only(
        base_rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    hard_join = join_hard_control_artifacts(
        base_rows,
        hard_manifest,
        hard_cheap_artifact,
        hard_causal_artifact,
    )
    hard_control, _ = evaluate_hard_control(
        hard_join,
        n_bootstrap=n_bootstrap,
        seed=seed,
        fold_order=fold_order,
    )
    old_primary = next(
        row
        for row in old_binary["auc_table"]
        if row["estimator"] == "READ_IG"
    )
    hard_control["old_dashboard_auc"] = {
        "estimate": old_primary["heldout_auc"],
        "ci95_low": old_primary["ci95_low"],
        "ci95_high": old_primary["ci95_high"],
        "n_pairs": old_primary["n_prompt_pairs"],
    }
    distributions = _read_distribution_result(
        base_rows, hard_control["task_rows"]
    )
    decision = make_final_decision(old_binary, engine_only, hard_control)

    counts = verification_artifact.get("counts")
    selection = verification_artifact.get("selection")
    logit_agreement = verification_artifact.get("logit_agreement")
    model = causal_artifact.get("model", verification_artifact.get("model"))
    if not all(
        isinstance(value, Mapping)
        for value in (counts, selection, logit_agreement, model)
    ):
        raise ValueError("verification/causal metadata required for final metrics")
    return {
        "schema_version": "read-final-cleanup-v1",
        "status": "COMPLETE",
        "protocol": {
            "model": dict(model),
            "seed": int(seed),
            "source_layer": int(selection["layer"]),
            "position_rule": selection["position_rule"],
            "written_threshold": float(selection["written_threshold"]),
            "ig_steps": int(cheap_artifact["ig_steps"]),
            "fold_order": [int(fold) for fold in fold_order],
            "bootstrap_draws": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "bootstrap_unit": "unordered concept dependency group",
        },
        "preflight": {
            "logit_agreement_prompts": int(logit_agreement["n"]),
            "logit_agreement_max_mean_kl": float(
                logit_agreement["max_mean_kl"]
            ),
            "logit_agreement_threshold": float(logit_agreement["threshold"]),
            "logit_agreement_status": logit_agreement["status"],
        },
        "dataset": {
            "candidate_pairs": int(counts["candidates"]),
            "calibration_pairs": int(counts["calibration_pairs"]),
            "evaluation_pairs": int(counts["evaluation_pairs"]),
            "verified_pairs": int(counts["verified_pairs"]),
            "unverified_pairs": int(counts["unverified_pairs"]),
            "evaluation_dependency_groups": int(
                len({row["dependency_group"] for row in base_rows})
            ),
            "hard_dashboard_candidates": int(
                hard_control["verification"]["candidates"]
            ),
            "hard_dashboard_verified": int(
                hard_control["verification"]["verified_hard"]
            ),
            "hard_dashboard_unverified": int(
                hard_control["verification"]["unverified_hard"]
            ),
        },
        "task_rows": base_rows,
        "old_binary": old_binary,
        "engine_only": engine_only,
        "hard_control": hard_control,
        "distributions": distributions,
        "decision": decision,
    }


def build_final_metrics(
    verification_artifact: Mapping[str, Any],
    causal_artifact: Mapping[str, Any],
    cheap_artifact: Mapping[str, Any],
    hard_manifest: Mapping[str, Any],
    hard_cheap_artifact: Mapping[str, Any],
    hard_causal_artifact: Mapping[str, Any],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    fold_order: Sequence[int] = DEFAULT_FOLD_ORDER,
) -> dict[str, Any]:
    """Compatibility spelling for :func:`assemble_final_metrics`."""

    return assemble_final_metrics(
        verification_artifact,
        causal_artifact,
        cheap_artifact,
        hard_manifest,
        hard_cheap_artifact,
        hard_causal_artifact,
        n_bootstrap=n_bootstrap,
        seed=seed,
        fold_order=fold_order,
    )


def _auc_by_estimator(metrics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index the canonical old-control AUC table by estimator name."""

    table = metrics.get("old_binary", {}).get("auc_table", ())
    return {str(row["estimator"]): row for row in table}


def extract_provenance(
    final_metrics: Mapping[str, Any],
    *,
    snapshot: str,
    source_commit: str,
    branch: str,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extract the mandatory post-refactor snapshot from canonical metrics.

    The returned shape matches ``PROVENANCE_pre_refactor.json`` so comparison is
    mechanical.  Paths and hashes are context, while every scientific scalar is
    copied from the evaluated final metrics rather than restated by hand.
    """

    if final_metrics.get("status") != "COMPLETE":
        raise ValueError("final metrics must be COMPLETE before provenance")
    protocol = final_metrics["protocol"]
    old = final_metrics["old_binary"]
    engine = final_metrics["engine_only"]
    hard = final_metrics["hard_control"]
    distributions = final_metrics["distributions"]
    auc = _auc_by_estimator(final_metrics)
    required_estimators = set(DEFAULT_SCORE_KEYS)
    if not required_estimators <= set(auc):
        raise ValueError(f"missing AUC estimators: {sorted(required_estimators-set(auc))}")
    correlations = engine["correlations"]
    classes = distributions["diagnostics"]["classes"]
    pairwise = distributions["diagnostics"]["pairwise"]
    base_sanity = old["causal_sanity"]
    hard_sanity = hard["causal_sanity"]
    decision = final_metrics["decision"]

    def auc_value(estimator: str, field: str) -> Any:
        """Read one named field from the canonical estimator AUC record."""

        return auc[estimator][field]

    return {
        "schema_version": "cleanup-provenance-v1",
        "snapshot": str(snapshot),
        "source_commit": str(source_commit),
        "branch": str(branch),
        "comparison_policy": {
            "absolute_tolerance": 0.001,
            "relative_tolerance": 0.001,
            "rule": (
                "Every headline scalar must match within max(absolute_tolerance, "
                "relative_tolerance * abs(pre)). Counts, IDs, decisions, and "
                "booleans must match exactly."
            ),
        },
        "model": {
            **dict(protocol["model"]),
            "seed": int(protocol["seed"]),
            "source_layer": int(protocol["source_layer"]),
            "position_rule": protocol["position_rule"],
            "written_threshold": float(protocol["written_threshold"]),
            "ig_steps": int(protocol["ig_steps"]),
            "folds": int(len(protocol["fold_order"])),
            "bootstrap_draws": int(protocol["bootstrap_draws"]),
        },
        "source_hashes": dict(source_hashes or {}),
        "preflight": dict(final_metrics["preflight"]),
        "dataset": dict(final_metrics["dataset"]),
        "causal_sanity": {
            "n_pairs": int(base_sanity["n_pairs"]),
            "engine_C_median": base_sanity["engine_C_median"],
            "engine_abs_C_median": base_sanity["engine_abs_C_median"],
            "old_dashboard_C_median": base_sanity["dashboard_C_median"],
            "old_dashboard_abs_C_median": base_sanity[
                "dashboard_abs_C_median"
            ],
            "hard_dashboard_C_median": hard_sanity["hard_dashboard_C_median"],
            "hard_dashboard_abs_C_median": hard_sanity[
                "hard_dashboard_abs_C_median"
            ],
            "hard_dashboard_C_min": hard_sanity["hard_dashboard_C_min"],
            "hard_dashboard_C_max": hard_sanity["hard_dashboard_C_max"],
            "engine_sharp_directional_disagreements": base_sanity[
                "engine_sharp_directional_disagreements"
            ],
            "old_dashboard_sharp_directional_disagreements": base_sanity[
                "dashboard_sharp_directional_disagreements"
            ],
            "hard_dashboard_sharp_directional_disagreements": hard_sanity[
                "hard_dashboard_sharp_directional_disagreements"
            ],
        },
        "binary_detection": {
            "READ_IG_old_dashboard_auc": auc_value("READ_IG", "heldout_auc"),
            "READ_IG_old_dashboard_ci95_low": auc_value("READ_IG", "ci95_low"),
            "READ_IG_old_dashboard_ci95_high": auc_value("READ_IG", "ci95_high"),
            "READ_IG_hard_dashboard_auc": hard["hard_dashboard_auc"]["estimate"],
            "READ_IG_hard_dashboard_ci95_low": hard["hard_dashboard_auc"][
                "ci95_low"
            ],
            "READ_IG_hard_dashboard_ci95_high": hard["hard_dashboard_auc"][
                "ci95_high"
            ],
            "READ_local_old_dashboard_auc": auc_value(
                "READ_local", "heldout_auc"
            ),
            "READ_local_old_dashboard_ci95_low": auc_value(
                "READ_local", "ci95_low"
            ),
            "READ_local_old_dashboard_ci95_high": auc_value(
                "READ_local", "ci95_high"
            ),
            "capacity_baseline_auc": auc_value(
                "weight_norm_baseline", "heldout_auc"
            ),
            "capacity_baseline_ci95_low": auc_value(
                "weight_norm_baseline", "ci95_low"
            ),
            "capacity_baseline_ci95_high": auc_value(
                "weight_norm_baseline", "ci95_high"
            ),
        },
        "graded_use": {
            "READ_IG_overall_spearman_rho": auc_value(
                "READ_IG", "spearman_rho_with_abs_C"
            ),
            "READ_local_overall_spearman_rho": auc_value(
                "READ_local", "spearman_rho_with_abs_C"
            ),
            "capacity_baseline_overall_spearman_rho": auc_value(
                "weight_norm_baseline", "spearman_rho_with_abs_C"
            ),
            "READ_IG_engine_only_rho": correlations["READ_IG"]["estimate"],
            "READ_IG_engine_only_ci95_low": correlations["READ_IG"]["ci95_low"],
            "READ_IG_engine_only_ci95_high": correlations["READ_IG"]["ci95_high"],
            "READ_local_engine_only_rho": correlations["READ_local"]["estimate"],
            "READ_local_engine_only_ci95_low": correlations["READ_local"][
                "ci95_low"
            ],
            "READ_local_engine_only_ci95_high": correlations["READ_local"][
                "ci95_high"
            ],
            "capacity_baseline_engine_only_rho": correlations[
                "weight_norm_baseline"
            ]["estimate"],
            "capacity_baseline_engine_only_ci95_low": correlations[
                "weight_norm_baseline"
            ]["ci95_low"],
            "capacity_baseline_engine_only_ci95_high": correlations[
                "weight_norm_baseline"
            ]["ci95_high"],
            "engine_abs_C_min": engine["within_engine_auc"]["abs_C_min"],
            "engine_abs_C_max": engine["within_engine_auc"]["abs_C_max"],
            "within_engine_auc_status": engine["within_engine_auc"]["status"],
        },
        "read_distributions": {
            "engine_min": classes["engine"]["minimum"],
            "engine_median": classes["engine"]["median"],
            "engine_max": classes["engine"]["maximum"],
            "engine_iqr": classes["engine"]["iqr"],
            "old_dashboard_min": classes["old_dashboard"]["minimum"],
            "old_dashboard_median": classes["old_dashboard"]["median"],
            "old_dashboard_max": classes["old_dashboard"]["maximum"],
            "old_dashboard_iqr": classes["old_dashboard"]["iqr"],
            "hard_dashboard_min": classes["hard_dashboard"]["minimum"],
            "hard_dashboard_median": classes["hard_dashboard"]["median"],
            "hard_dashboard_max": classes["hard_dashboard"]["maximum"],
            "hard_dashboard_iqr": classes["hard_dashboard"]["iqr"],
            "old_hard_range_overlap_fraction": pairwise[
                "hard_dashboard__vs__old_dashboard"
            ]["overlap_fraction_of_union"],
            "engine_old_range_gap": pairwise["engine__vs__old_dashboard"][
                "range_gap"
            ],
            "engine_hard_range_gap": pairwise["engine__vs__hard_dashboard"][
                "range_gap"
            ],
        },
        "decision": {
            "binary_detector": decision["binary_detector"],
            "graded_meter": decision["graded_meter"],
            "stress_test_label": decision["stress_test_label"],
        },
    }


def compare_provenance(
    pre: Mapping[str, Any],
    post: Mapping[str, Any],
    *,
    absolute_tolerance: float = 0.001,
    relative_tolerance: float = 0.001,
    raise_on_regression: bool = False,
) -> dict[str, Any]:
    """Compare scientific provenance fields and identify every regression.

    Floating-point fields use ``max(abs_tol, rel_tol * abs(pre))``.  Integers,
    strings, booleans, IDs, counts, and decisions must match exactly.  Commit
    IDs, snapshot labels, branches, and source hashes are context rather than
    scientific results and are therefore reported but excluded from the gate.
    """

    abs_tol = float(absolute_tolerance)
    rel_tol = float(relative_tolerance)
    if not math.isfinite(abs_tol) or abs_tol < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if not math.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    sections = (
        "model",
        "preflight",
        "dataset",
        "causal_sanity",
        "binary_detection",
        "graded_use",
        "read_distributions",
        "decision",
    )
    regressions: list[dict[str, Any]] = []
    compared = 0

    def walk(before: Any, after: Any, path: str) -> None:
        """Recursively compare one scientific provenance subtree."""

        nonlocal compared
        if isinstance(before, Mapping):
            if not isinstance(after, Mapping):
                regressions.append(
                    {"path": path, "before": before, "after": after, "reason": "type"}
                )
                return
            for key, value in before.items():
                child = f"{path}.{key}" if path else str(key)
                if key not in after:
                    regressions.append(
                        {"path": child, "before": value, "after": None, "reason": "missing"}
                    )
                else:
                    walk(value, after[key], child)
            return
        compared += 1
        if isinstance(before, bool) or isinstance(after, bool):
            equal = type(before) is type(after) and before == after
            tolerance = None
        elif isinstance(before, int) and isinstance(after, int):
            equal = before == after
            tolerance = None
        elif isinstance(before, (int, float)) and isinstance(after, (int, float)):
            before_float = float(before)
            after_float = float(after)
            tolerance = max(abs_tol, rel_tol * abs(before_float))
            equal = bool(
                math.isfinite(before_float)
                and math.isfinite(after_float)
                and abs(after_float - before_float) <= tolerance
            )
        else:
            equal = type(before) is type(after) and before == after
            tolerance = None
        if not equal:
            record = {
                "path": path,
                "before": before,
                "after": after,
                "reason": "outside_tolerance" if tolerance is not None else "not_equal",
            }
            if tolerance is not None:
                record["absolute_delta"] = abs(float(after) - float(before))
                record["tolerance"] = tolerance
            regressions.append(record)

    for section in sections:
        if section not in pre or section not in post:
            regressions.append(
                {
                    "path": section,
                    "before": pre.get(section),
                    "after": post.get(section),
                    "reason": "missing_section",
                }
            )
            continue
        walk(pre[section], post[section], section)

    result = {
        "status": "PASS" if not regressions else "REGRESSION",
        "absolute_tolerance": abs_tol,
        "relative_tolerance": rel_tol,
        "compared_leaf_fields": int(compared),
        "regression_count": int(len(regressions)),
        "regressions": regressions,
        "excluded_context_fields": [
            "snapshot",
            "source_commit",
            "branch",
            "source_hashes",
            "comparison_policy",
        ],
    }
    if regressions and raise_on_regression:
        paths = ", ".join(record["path"] for record in regressions[:8])
        raise RuntimeError(f"provenance regression in {paths}")
    return result

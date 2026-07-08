"""Model-free statistics and validation helpers for the isolated v6 stress test.

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
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_CONFIDENCE = 0.95


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

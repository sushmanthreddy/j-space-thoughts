"""Leakage-safe validation metrics for cached mean-difference activations.

This module deliberately contains no model-loading or activation-capture code.  Its
inputs are cached residual matrices with shape ``[prompt_slot, d_model]`` and unit
directions with shape ``[d_model]``.  All calculations run deterministically on CPU
in float64.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch


PromptMatrices = Mapping[str, Mapping[int, torch.Tensor]]
DirectionBank = Mapping[str, Mapping[int, torch.Tensor]]
ExclusionGraph = Mapping[str, Iterable[str]]
CalibrationBank = Mapping[str, Mapping[int, Mapping[str, Any]]]


def _as_cpu_double(tensor: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if tensor.dtype == torch.bool or not (
        tensor.dtype.is_floating_point or tensor.dtype.is_complex
    ):
        raise TypeError(f"{label} must have a floating-point dtype")
    if tensor.dtype.is_complex:
        raise TypeError(f"{label} must be real-valued")
    value = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values")
    return value


def _canonical_graph(
    concepts: tuple[str, ...], exclusion_graph: ExclusionGraph
) -> dict[str, frozenset[str]]:
    concept_set = set(concepts)
    if set(exclusion_graph) != concept_set:
        missing = sorted(concept_set - set(exclusion_graph))
        extra = sorted(set(exclusion_graph) - concept_set)
        raise ValueError(
            "Exclusion graph keys must exactly match concepts: "
            f"missing={missing}, extra={extra}"
        )
    graph: dict[str, frozenset[str]] = {}
    for concept in concepts:
        excluded = frozenset(str(value) for value in exclusion_graph[concept])
        unknown = excluded - concept_set
        if unknown:
            raise ValueError(
                f"Exclusions for {concept!r} reference unknown concepts: "
                f"{sorted(unknown)}"
            )
        if concept in excluded:
            raise ValueError(f"Concept {concept!r} cannot exclude itself")
        graph[concept] = excluded
    for concept, excluded in graph.items():
        for other in excluded:
            if concept not in graph[other]:
                raise ValueError(
                    "Exclusion graph must be symmetric: "
                    f"{concept!r} excludes {other!r}, but not conversely"
                )
    return graph


def _eligible_concepts(
    concepts: tuple[str, ...],
    graph: Mapping[str, frozenset[str]],
    concept: str,
) -> tuple[str, ...]:
    eligible = tuple(
        other
        for other in concepts
        if other != concept and other not in graph[concept]
    )
    if not eligible:
        raise ValueError(f"No eligible baseline concepts remain for {concept!r}")
    return eligible


def _validate_matrices(
    matrices: PromptMatrices,
    *,
    label: str,
    expected_concepts: tuple[str, ...] | None = None,
    expected_layers: tuple[int, ...] | None = None,
    expected_dimensions: Mapping[int, int] | None = None,
    minimum_slots: int = 1,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    dict[int, int],
    int,
    dict[str, dict[int, torch.Tensor]],
]:
    if not matrices:
        raise ValueError(f"{label} must contain at least one concept")
    if not all(isinstance(concept, str) and concept for concept in matrices):
        raise TypeError(f"{label} concept keys must be nonempty strings")
    concepts = tuple(sorted(matrices))
    if expected_concepts is not None and concepts != expected_concepts:
        raise ValueError(f"{label} concepts do not match the direction bank")

    first_layers = tuple(sorted(matrices[concepts[0]]))
    if not first_layers:
        raise ValueError(f"{label} must contain at least one layer")
    if not all(isinstance(layer, int) and not isinstance(layer, bool) for layer in first_layers):
        raise TypeError(f"{label} layer keys must be integers")
    if expected_layers is not None and first_layers != expected_layers:
        raise ValueError(f"{label} layers do not match the direction bank")
    for concept in concepts:
        if tuple(sorted(matrices[concept])) != first_layers:
            raise ValueError(f"Every {label} concept must cover identical layers")

    converted: dict[str, dict[int, torch.Tensor]] = {}
    dimensions: dict[int, int] = {}
    prompt_counts: set[int] = set()
    for concept in concepts:
        converted[concept] = {}
        for layer in first_layers:
            value = _as_cpu_double(
                matrices[concept][layer],
                label=f"{label}[{concept!r}][{layer}]",
            )
            if value.ndim != 2:
                raise ValueError(
                    f"{label}[{concept!r}][{layer}] must have shape "
                    "[prompt_slot, d_model]"
                )
            if value.shape[0] < minimum_slots or value.shape[1] < 1:
                raise ValueError(
                    f"{label}[{concept!r}][{layer}] has invalid shape "
                    f"{tuple(value.shape)}"
                )
            prompt_counts.add(int(value.shape[0]))
            dimension = int(value.shape[1])
            if layer in dimensions and dimensions[layer] != dimension:
                raise ValueError(
                    f"{label} d_model differs across concepts at layer {layer}"
                )
            dimensions[layer] = dimension
            if expected_dimensions is not None and (
                expected_dimensions.get(layer) != dimension
            ):
                raise ValueError(
                    f"{label}[{concept!r}][{layer}] has d_model={dimension}; "
                    f"expected {expected_dimensions.get(layer)}. Matrices must be "
                    "oriented [prompt_slot, d_model]."
                )
            converted[concept][layer] = value
    if len(prompt_counts) != 1:
        raise ValueError(
            f"{label} requires equal prompt-slot counts for matched baselines"
        )
    return concepts, first_layers, dimensions, prompt_counts.pop(), converted


def _validate_directions(
    directions: DirectionBank,
    *,
    norm_tolerance: float,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    dict[int, int],
    dict[str, dict[int, torch.Tensor]],
]:
    if not directions:
        raise ValueError("Direction bank must contain at least one concept")
    if norm_tolerance <= 0:
        raise ValueError("norm_tolerance must be positive")
    if not all(isinstance(concept, str) and concept for concept in directions):
        raise TypeError("Direction concept keys must be nonempty strings")
    concepts = tuple(sorted(directions))
    layers = tuple(sorted(directions[concepts[0]]))
    if not layers:
        raise ValueError("Direction bank must contain at least one layer")
    if not all(isinstance(layer, int) and not isinstance(layer, bool) for layer in layers):
        raise TypeError("Direction layer keys must be integers")
    converted: dict[str, dict[int, torch.Tensor]] = {}
    dimensions: dict[int, int] = {}
    for concept in concepts:
        if tuple(sorted(directions[concept])) != layers:
            raise ValueError("Every direction concept must cover identical layers")
        converted[concept] = {}
        for layer in layers:
            value = _as_cpu_double(
                directions[concept][layer],
                label=f"directions[{concept!r}][{layer}]",
            )
            if value.ndim != 1 or value.numel() < 1:
                raise ValueError(
                    f"directions[{concept!r}][{layer}] must have shape [d_model]"
                )
            dimension = int(value.numel())
            if layer in dimensions and dimensions[layer] != dimension:
                raise ValueError(
                    f"Direction d_model differs across concepts at layer {layer}"
                )
            dimensions[layer] = dimension
            norm = float(torch.linalg.vector_norm(value))
            if abs(norm - 1.0) > norm_tolerance:
                raise ValueError(
                    f"Direction for {concept!r}, layer {layer} is not unit length: "
                    f"norm={norm:.9g}"
                )
            converted[concept][layer] = value
    return concepts, layers, dimensions, converted


def _matched_difference(
    matrices: Mapping[str, Mapping[int, torch.Tensor]],
    concept: str,
    layer: int,
    eligible: tuple[str, ...],
    slot_indices: torch.Tensor,
) -> torch.Tensor:
    positive = matrices[concept][layer].index_select(0, slot_indices)
    baseline = torch.stack(
        [matrices[other][layer].index_select(0, slot_indices) for other in eligible]
    ).mean(dim=0)
    return (positive - baseline).mean(dim=0)


def _validated_training_inputs(
    train_matrices: PromptMatrices,
    directions: DirectionBank,
    exclusion_graph: ExclusionGraph,
    *,
    norm_tolerance: float,
    orientation_tolerance: float,
    minimum_slots: int = 1,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    dict[str, frozenset[str]],
    int,
    dict[str, dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
]:
    if orientation_tolerance <= 0:
        raise ValueError("orientation_tolerance must be positive")
    concepts, layers, dimensions, direction_values = _validate_directions(
        directions,
        norm_tolerance=norm_tolerance,
    )
    _, _, _, n_slots, matrix_values = _validate_matrices(
        train_matrices,
        label="train_matrices",
        expected_concepts=concepts,
        expected_layers=layers,
        expected_dimensions=dimensions,
        minimum_slots=minimum_slots,
    )
    graph = _canonical_graph(concepts, exclusion_graph)
    all_slots = torch.arange(n_slots, dtype=torch.long)
    for concept in concepts:
        eligible = _eligible_concepts(concepts, graph, concept)
        for layer in layers:
            expected = _matched_difference(
                matrix_values,
                concept,
                layer,
                eligible,
                all_slots,
            )
            expected_norm = float(torch.linalg.vector_norm(expected))
            if not np.isfinite(expected_norm) or expected_norm == 0.0:
                raise ValueError(
                    f"Degenerate matched mean difference for {concept!r}, "
                    f"layer {layer}"
                )
            cosine = float(
                torch.dot(expected / expected_norm, direction_values[concept][layer])
            )
            if cosine < 1.0 - orientation_tolerance:
                raise ValueError(
                    "Direction orientation/formula mismatch for "
                    f"{concept!r}, layer {layer}: cosine with the matched "
                    f"leave-exclusion-out mean difference is {cosine:.9g}"
                )
    return concepts, layers, graph, n_slots, matrix_values, direction_values


def fit_score_calibration(
    train_matrices: PromptMatrices,
    directions: DirectionBank,
    exclusion_graph: ExclusionGraph,
    *,
    norm_tolerance: float = 1e-5,
    orientation_tolerance: float = 1e-4,
    minimum_scale: float = 1e-12,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Fit per-direction affine score calibration using training negatives only.

    For a direction, negatives are all concepts other than itself and its
    predeclared exclusions.  Equal prompt-slot counts are required, and the center
    and population scale weight every eligible concept equally within every slot.
    ``slot_centers`` are retained to make the matched-slot calculation auditable;
    no held-out activation or label enters the fitted affine transformation.
    """

    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive")
    (
        concepts,
        layers,
        graph,
        n_slots,
        matrices,
        direction_values,
    ) = _validated_training_inputs(
        train_matrices,
        directions,
        exclusion_graph,
        norm_tolerance=norm_tolerance,
        orientation_tolerance=orientation_tolerance,
    )

    calibration: dict[str, dict[int, dict[str, Any]]] = {}
    for concept in concepts:
        eligible = _eligible_concepts(concepts, graph, concept)
        calibration[concept] = {}
        for layer in layers:
            direction = direction_values[concept][layer]
            negative_scores = torch.stack(
                [matrices[other][layer] @ direction for other in eligible]
            )
            slot_centers = negative_scores.mean(dim=0)
            center = float(slot_centers.mean())
            scale = float(torch.sqrt(torch.mean((negative_scores - center) ** 2)))
            if not np.isfinite(scale) or scale <= minimum_scale:
                raise ValueError(
                    f"Degenerate training calibration scale for {concept!r}, "
                    f"layer {layer}: {scale:.9g}"
                )
            calibration[concept][layer] = {
                "center": center,
                "scale": scale,
                "eligible_concepts": list(eligible),
                "n_eligible_concepts": len(eligible),
                "n_train_slots": n_slots,
                "n_negative_scores": int(negative_scores.numel()),
                "slot_centers": [float(value) for value in slot_centers],
            }
    return calibration


def _validate_calibration(
    calibration: CalibrationBank,
    concepts: tuple[str, ...],
    layers: tuple[int, ...],
) -> dict[str, dict[int, tuple[float, float]]]:
    if set(calibration) != set(concepts):
        raise ValueError("Calibration concepts do not match the direction bank")
    values: dict[str, dict[int, tuple[float, float]]] = {}
    for concept in concepts:
        if set(calibration[concept]) != set(layers):
            raise ValueError(
                f"Calibration layers for {concept!r} do not match directions"
            )
        values[concept] = {}
        for layer in layers:
            try:
                center = float(calibration[concept][layer]["center"])
                scale = float(calibration[concept][layer]["scale"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid calibration for {concept!r}, layer {layer}"
                ) from error
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"Calibration for {concept!r}, layer {layer} must have a "
                    "finite center and positive finite scale"
                )
            values[concept][layer] = (center, scale)
    return values


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC requires both positive and negative examples")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def heldout_calibrated_retrieval(
    heldout_matrices: PromptMatrices,
    directions: DirectionBank,
    calibration: CalibrationBank,
    *,
    norm_tolerance: float = 1e-5,
    n_permutations: int = 5000,
    permutation_seed: int = 1729,
) -> dict[str, Any]:
    """Score every held-out prompt against every calibrated direction.

    Ties in retrieval rank are broken by the fixed, sorted concept order.  AUROC
    uses the usual half-credit convention for tied positive/negative scores.
    The result contains one JSON-ready row per prompt and per-layer aggregate
    top-1 accuracy, mean reciprocal rank, and macro one-vs-rest AUROC.  The
    fixed-label test holds score-derived predictions fixed and shuffles the
    balanced true labels; its one-sided p-value has the standard +1 correction.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if not isinstance(permutation_seed, int) or isinstance(permutation_seed, bool):
        raise TypeError("permutation_seed must be an integer")
    concepts, layers, dimensions, direction_values = _validate_directions(
        directions,
        norm_tolerance=norm_tolerance,
    )
    _, _, _, n_slots, matrices = _validate_matrices(
        heldout_matrices,
        label="heldout_matrices",
        expected_concepts=concepts,
        expected_layers=layers,
        expected_dimensions=dimensions,
    )
    affine = _validate_calibration(calibration, concepts, layers)

    rows: list[dict[str, Any]] = []
    by_layer: dict[int, dict[str, Any]] = {}
    for layer in layers:
        layer_rows: list[dict[str, Any]] = []
        score_columns: dict[str, list[float]] = {concept: [] for concept in concepts}
        labels: list[str] = []
        for true_concept in concepts:
            for slot in range(n_slots):
                activation = matrices[true_concept][layer][slot]
                raw_scores: dict[str, float] = {}
                calibrated_scores: dict[str, float] = {}
                for candidate in concepts:
                    raw = float(activation @ direction_values[candidate][layer])
                    center, scale = affine[candidate][layer]
                    raw_scores[candidate] = raw
                    calibrated_scores[candidate] = (raw - center) / scale
                ordering = sorted(
                    concepts,
                    key=lambda candidate: (-calibrated_scores[candidate], candidate),
                )
                rank = ordering.index(true_concept) + 1
                row = {
                    "layer": layer,
                    "true_concept": true_concept,
                    "heldout_slot": slot,
                    "predicted_concept": ordering[0],
                    "rank": rank,
                    "top1": int(rank == 1),
                    "reciprocal_rank": 1.0 / rank,
                    "true_raw_score": raw_scores[true_concept],
                    "true_calibrated_score": calibrated_scores[true_concept],
                    "raw_scores": raw_scores,
                    "calibrated_scores": calibrated_scores,
                }
                layer_rows.append(row)
                rows.append(row)
                labels.append(true_concept)
                for candidate in concepts:
                    score_columns[candidate].append(calibrated_scores[candidate])

        label_array = np.asarray(labels, dtype=object)
        per_concept_auroc = {
            concept: _binary_auroc(
                np.asarray(score_columns[concept], dtype=np.float64),
                label_array == concept,
            )
            for concept in concepts
        }
        concept_indices = {concept: index for index, concept in enumerate(concepts)}
        label_indices = np.asarray(
            [concept_indices[label] for label in labels], dtype=np.int64
        )
        prediction_indices = np.asarray(
            [concept_indices[row["predicted_concept"]] for row in layer_rows],
            dtype=np.int64,
        )
        observed_top1 = float(np.mean(prediction_indices == label_indices))
        rng = np.random.default_rng(permutation_seed)
        exceedances = 0
        for _ in range(n_permutations):
            permuted_labels = rng.permutation(label_indices)
            permuted_top1 = float(np.mean(prediction_indices == permuted_labels))
            exceedances += int(permuted_top1 >= observed_top1)
        permutation_p = (exceedances + 1.0) / (n_permutations + 1.0)
        by_layer[layer] = {
            "n_way": len(concepts),
            "n_heldout_slots_per_concept": n_slots,
            "n_rows": len(layer_rows),
            "top1_accuracy": observed_top1,
            "mean_reciprocal_rank": float(
                np.mean([row["reciprocal_rank"] for row in layer_rows])
            ),
            "macro_ovr_auroc": float(np.mean(list(per_concept_auroc.values()))),
            "per_concept_ovr_auroc": per_concept_auroc,
            "top1_fixed_label_permutation": {
                "statistic": observed_top1,
                "p_value": permutation_p,
                "n_permutations": n_permutations,
                "seed": permutation_seed,
                "alternative": "greater_or_equal",
                "plus_one_correction": True,
            },
        }
    return {"concept_order": list(concepts), "rows": rows, "by_layer": by_layer}


def heldout_matched_baseline_deltas(
    heldout_matrices: PromptMatrices,
    directions: DirectionBank,
    exclusion_graph: ExclusionGraph,
    *,
    norm_tolerance: float = 1e-5,
) -> list[dict[str, Any]]:
    """Return own-direction scores minus eligible held-out matched baselines.

    This is a descriptive sign check, not a classifier calibration: the baseline
    for held-out slot ``s`` uses other held-out concepts at that same slot.
    """

    concepts, layers, dimensions, direction_values = _validate_directions(
        directions,
        norm_tolerance=norm_tolerance,
    )
    _, _, _, n_slots, matrices = _validate_matrices(
        heldout_matrices,
        label="heldout_matrices",
        expected_concepts=concepts,
        expected_layers=layers,
        expected_dimensions=dimensions,
    )
    graph = _canonical_graph(concepts, exclusion_graph)
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for concept in concepts:
            eligible = _eligible_concepts(concepts, graph, concept)
            direction = direction_values[concept][layer]
            for slot in range(n_slots):
                own_score = float(matrices[concept][layer][slot] @ direction)
                eligible_scores = [
                    float(matrices[other][layer][slot] @ direction)
                    for other in eligible
                ]
                baseline_score = float(np.mean(eligible_scores))
                rows.append(
                    {
                        "layer": layer,
                        "concept": concept,
                        "heldout_slot": slot,
                        "eligible_concepts": list(eligible),
                        "own_score": own_score,
                        "matched_baseline_score": baseline_score,
                        "delta": own_score - baseline_score,
                    }
                )
    return rows


def leave_one_train_slot_out_stability(
    train_matrices: PromptMatrices,
    directions: DirectionBank,
    exclusion_graph: ExclusionGraph,
    *,
    norm_tolerance: float = 1e-5,
    orientation_tolerance: float = 1e-4,
) -> list[dict[str, Any]]:
    """Recompute each MD direction after leaving out one matched train slot.

    For every omission, the same slot is removed from the positive concept and
    every eligible baseline concept before applying the original matched
    leave-exclusion-out formula.  Returned cosine is against the supplied full
    unit direction; the leave-one-out vector is not sign-flipped.
    """

    (
        concepts,
        layers,
        graph,
        n_slots,
        matrices,
        direction_values,
    ) = _validated_training_inputs(
        train_matrices,
        directions,
        exclusion_graph,
        norm_tolerance=norm_tolerance,
        orientation_tolerance=orientation_tolerance,
        minimum_slots=2,
    )
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for concept in concepts:
            eligible = _eligible_concepts(concepts, graph, concept)
            full_direction = direction_values[concept][layer]
            for left_out_slot in range(n_slots):
                kept_slots = torch.tensor(
                    [slot for slot in range(n_slots) if slot != left_out_slot],
                    dtype=torch.long,
                )
                difference = _matched_difference(
                    matrices,
                    concept,
                    layer,
                    eligible,
                    kept_slots,
                )
                norm = float(torch.linalg.vector_norm(difference))
                if not np.isfinite(norm) or norm == 0.0:
                    raise ValueError(
                        "Degenerate leave-one-slot-out direction for "
                        f"{concept!r}, layer {layer}, slot {left_out_slot}"
                    )
                loo_direction = difference / norm
                rows.append(
                    {
                        "layer": layer,
                        "concept": concept,
                        "left_out_slot": left_out_slot,
                        "n_slots_retained": n_slots - 1,
                        "eligible_concepts": list(eligible),
                        "cosine_to_full_direction": float(
                            torch.dot(loo_direction, full_direction)
                        ),
                    }
                )
    return rows

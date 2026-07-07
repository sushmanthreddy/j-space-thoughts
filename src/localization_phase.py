"""Downstream component localization for the two-hop raw J-Lens directions.

This phase consumes successful notebook-02 rows rather than redefining the
two-hop sample.  For a single source layer it removes the preregistered raw
``W_U J`` direction, records the resulting changes in every downstream Qwen
MLP output and attention stream immediately before ``o_proj``, and scores each
change with the clean behavior gradient::

    score_k = <d M_clean / d a_k, a_k(ablated) - a_k(clean)>.

The component scores overlap: later components depend on earlier components,
and the clean-gradient linearization ignores interactions.  They are therefore
localization/mediation diagnostics, not an additive causal decomposition.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.interventions import ablate_direction, residual_edit_hooks
from src.jlens_iface import (
    jlens_direction_bank,
    load_local_lens,
    load_published_lens,
)
from src.metrics import save_json
from src.model_utils import load_model, release_model, set_seed
from src.read_scores import qwen_head_ov_read, qwen_mlp_gain
from src.twohop_phase import PRIMARY_DIRECTION_METHOD


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "localization-phase-v1"
SEED = 1729
NON_ADDITIVE_WARNING = (
    "Clean-gradient dot (single-source-ablated minus clean) component scores "
    "are overlapping first-order mediation/localization diagnostics. Attention "
    "and MLP paths, and earlier and later components, overlap; scores must not "
    "be summed or interpreted as an additive causal decomposition."
)


def _finite_number(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _json_ready(value: Any) -> Any:
    """Convert nested scientific values into strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_ready(value.detach().cpu().item())
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _average_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average-tie ranks scaled to ``[0, 1]``."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Percentile ranks require finite nonempty values")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    if array.size == 1:
        return np.full(1, 0.5, dtype=np.float64)
    return ranks / (array.size - 1)


def _successful_raw_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    names: set[str] = set()
    for row in rows:
        if row.get("measurement_status") != "OK":
            continue
        if row.get("direction_method") != PRIMARY_DIRECTION_METHOD:
            continue
        name = str(row.get("name", ""))
        if not name:
            raise ValueError("A successful raw J-Lens row has no item name")
        if name in names:
            raise ValueError(f"Duplicate successful raw J-Lens item name: {name!r}")
        aggregate = row.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"Row {name!r} has no aggregate mapping")
        _finite_number(aggregate.get("write_abs_mean"), name=f"{name} WRITE")
        _finite_number(aggregate.get("read_abs_mean"), name=f"{name} READ")
        names.add(name)
        selected.append(row)
    return sorted(selected, key=lambda row: str(row["name"]))


def select_localization_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    n_per_cell: int = 1,
) -> dict[str, Any]:
    """Select a deterministic 2x2 WRITE/READ quantile subset.

    The four requested cells are high/high, high/low, low/high, and low/low.
    Within a cell, rows closest to its empirical percentile corner are chosen;
    ties break on item name.  If a strict cell is empty, the nearest unused row
    is retained and explicitly marked as a fallback rather than silently
    changing the quantile threshold.  Cell labels are descriptive and assign
    no a priori narration or driver class.
    """

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")
    if n_per_cell < 1:
        raise ValueError("n_per_cell must be positive")
    eligible = _successful_raw_rows(rows)
    required = 4 * n_per_cell
    if len(eligible) < required:
        raise ValueError(
            f"Need at least {required} distinct successful raw rows, got {len(eligible)}"
        )

    write = np.asarray(
        [float(row["aggregate"]["write_abs_mean"]) for row in eligible],
        dtype=np.float64,
    )
    read = np.asarray(
        [float(row["aggregate"]["read_abs_mean"]) for row in eligible],
        dtype=np.float64,
    )
    if float(np.ptp(write)) == 0.0 or float(np.ptp(read)) == 0.0:
        raise ValueError("WRITE and READ must both vary to form localization strata")
    write_low, write_high = np.quantile(
        write, [lower_quantile, upper_quantile], method="linear"
    )
    read_low, read_high = np.quantile(
        read, [lower_quantile, upper_quantile], method="linear"
    )
    write_percentile = _average_percentile_ranks(write)
    read_percentile = _average_percentile_ranks(read)

    definitions = (
        ("high_write_high_read", "high", "high", 1.0, 1.0),
        ("high_write_low_read", "high", "low", 1.0, 0.0),
        ("low_write_high_read", "low", "high", 0.0, 1.0),
        ("low_write_low_read", "low", "low", 0.0, 0.0),
    )

    def matches(index: int, write_class: str, read_class: str) -> bool:
        write_match = (
            write[index] >= write_high
            if write_class == "high"
            else write[index] <= write_low
        )
        read_match = (
            read[index] >= read_high
            if read_class == "high"
            else read[index] <= read_low
        )
        return bool(write_match and read_match)

    chosen_indices: set[int] = set()
    selections: list[dict[str, Any]] = []
    cell_counts: dict[str, dict[str, int]] = {}
    for cell, write_class, read_class, write_corner, read_corner in definitions:
        strict = [
            index
            for index in range(len(eligible))
            if matches(index, write_class, read_class)
        ]

        def sort_key(index: int) -> tuple[float, str]:
            distance = abs(write_percentile[index] - write_corner) + abs(
                read_percentile[index] - read_corner
            )
            return float(distance), str(eligible[index]["name"])

        strict.sort(key=sort_key)
        picked = [index for index in strict if index not in chosen_indices][
            :n_per_cell
        ]
        if len(picked) < n_per_cell:
            fallback = [
                index
                for index in range(len(eligible))
                if index not in chosen_indices and index not in picked
            ]
            fallback.sort(key=sort_key)
            picked.extend(fallback[: n_per_cell - len(picked)])
        if len(picked) != n_per_cell:
            raise RuntimeError(f"Could not fill localization cell {cell!r}")
        cell_counts[cell] = {
            "n_strict_candidates": len(strict),
            "n_selected": len(picked),
            "n_fallback": sum(
                not matches(index, write_class, read_class) for index in picked
            ),
        }
        for rank, index in enumerate(picked, start=1):
            chosen_indices.add(index)
            strict_match = matches(index, write_class, read_class)
            selections.append(
                {
                    "row": eligible[index],
                    "name": str(eligible[index]["name"]),
                    "cell": cell,
                    "cell_write_class": write_class,
                    "cell_read_class": read_class,
                    "strict_threshold_match": strict_match,
                    "selection_mode": (
                        "strict_quantile_cell" if strict_match else "nearest_fallback"
                    ),
                    "selection_rank_within_cell": rank,
                    "write_strength": float(write[index]),
                    "read_strength": float(read[index]),
                    "write_empirical_percentile": float(write_percentile[index]),
                    "read_empirical_percentile": float(read_percentile[index]),
                    "corner_l1_distance": float(sort_key(index)[0]),
                }
            )

    return {
        "selected": selections,
        "provenance": {
            "source_filter": (
                "measurement_status == OK and direction_method == jlens_raw_wu_j"
            ),
            "n_eligible_raw_rows": len(eligible),
            "n_selected": len(selections),
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "quantile_method": "numpy linear",
            "tie_break": "minimum percentile-corner L1 distance, then item name",
            "write_variable": "aggregate.write_abs_mean",
            "read_variable": "aggregate.read_abs_mean",
            "thresholds": {
                "write_low": float(write_low),
                "write_high": float(write_high),
                "read_low": float(read_low),
                "read_high": float(read_high),
            },
            "cells": cell_counts,
            "class_guardrail": (
                "Quantile cells are descriptive sampling strata only. No item is "
                "assigned an a priori narration or driver class."
            ),
        },
    }


def choose_source_layers(
    row: Mapping[str, Any],
    *,
    n_source_layers: int = 1,
    max_source_layer: int | None = None,
) -> dict[str, Any]:
    """Choose layers by absolute first-order source contribution.

    This is a transparent localization targeting rule, not a new READ estimate:
    ``abs(sum_position(WRITE * attribution_READ))`` is evaluated independently
    at every notebook-02 workspace layer.  A source must have at least one
    downstream block available for component capture.
    """

    if n_source_layers < 1:
        raise ValueError("n_source_layers must be positive")
    name = str(row.get("name", "<unnamed>"))
    raw = row.get("raw_arrays")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Row {name!r} has no raw_arrays mapping")
    write_by_layer = raw.get("write_by_layer_position")
    read_by_layer = raw.get("attribution_read_by_layer_position")
    if not isinstance(write_by_layer, Mapping) or not isinstance(read_by_layer, Mapping):
        raise ValueError(f"Row {name!r} lacks notebook-02 WRITE/READ arrays")
    if set(map(str, write_by_layer)) != set(map(str, read_by_layer)):
        raise ValueError(f"Row {name!r} WRITE/READ layers do not align")

    candidates: list[dict[str, Any]] = []
    for raw_layer in write_by_layer:
        layer = int(raw_layer)
        if max_source_layer is not None and layer > max_source_layer:
            continue
        write = np.asarray(write_by_layer[raw_layer], dtype=np.float64).reshape(-1)
        read = np.asarray(read_by_layer[raw_layer], dtype=np.float64).reshape(-1)
        if write.size == 0 or write.shape != read.shape:
            raise ValueError(f"Row {name!r} layer {layer} has misaligned arrays")
        if not np.isfinite(write).all() or not np.isfinite(read).all():
            raise ValueError(f"Row {name!r} layer {layer} has non-finite arrays")
        signed_sum = float(np.sum(write * read))
        candidates.append(
            {
                "layer": layer,
                "selection_score": abs(signed_sum),
                "signed_first_order_positive_damage": signed_sum,
                "n_positions": int(write.size),
                "write_abs_mean": float(np.mean(np.abs(write))),
                "read_abs_mean": float(np.mean(np.abs(read))),
            }
        )
    if len(candidates) < n_source_layers:
        raise ValueError(
            f"Row {name!r} has only {len(candidates)} eligible source layers"
        )
    candidates.sort(key=lambda item: (-item["selection_score"], item["layer"]))
    return {
        "selected_layers": [item["layer"] for item in candidates[:n_source_layers]],
        "selected": candidates[:n_source_layers],
        "all_candidates_ranked": candidates,
        "formula": "abs(sum_position(WRITE * attribution_READ))",
        "role": "source-layer targeting only; not a headline READ estimator",
        "tie_break": "lower layer index",
    }


def _validate_qwen_component_layer(block: torch.nn.Module, layer: int) -> tuple[int, int]:
    if not hasattr(block, "mlp") or not hasattr(block, "self_attn"):
        raise TypeError(f"Layer {layer} is not a Qwen-like decoder block")
    attention = block.self_attn
    required = ("o_proj", "config", "head_dim")
    if not all(hasattr(attention, field) for field in required):
        raise TypeError(f"Layer {layer} has no Qwen-like attention module")
    num_heads = int(attention.config.num_attention_heads)
    head_dim = int(attention.head_dim)
    if num_heads < 1 or head_dim < 1:
        raise ValueError(f"Layer {layer} has invalid head geometry")
    expected = num_heads * head_dim
    in_features = getattr(attention.o_proj, "in_features", None)
    if in_features is not None and int(in_features) != expected:
        raise ValueError(
            f"Layer {layer} o_proj input {in_features} != heads*head_dim {expected}"
        )
    return num_heads, head_dim


@contextmanager
def capture_qwen_components(
    blocks: Sequence[torch.nn.Module],
    layers: Sequence[int],
    *,
    start_graph: bool,
) -> Iterator[dict[str, dict[int, torch.Tensor]]]:
    """Capture Qwen MLP outputs and concatenated pre-``o_proj`` head streams.

    Handles are removed even when the model or a shape check raises.  When
    ``start_graph`` is true, the first otherwise non-differentiable captured
    tensor is promoted to an autograd leaf; this is required because the HF
    model parameters are frozen for J-Lens experiments.
    """

    layer_list = sorted(set(int(layer) for layer in layers))
    if not layer_list:
        raise ValueError("At least one component layer is required")
    if any(layer < 0 or layer >= len(blocks) for layer in layer_list):
        raise IndexError(f"Component layers outside [0, {len(blocks)}): {layer_list}")
    geometry = {
        layer: _validate_qwen_component_layer(blocks[layer], layer)
        for layer in layer_list
    }
    captures: dict[str, dict[int, torch.Tensor]] = {
        "mlp": {},
        "attention_pre_o_proj": {},
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def prepare(tensor: torch.Tensor, *, kind: str, layer: int) -> torch.Tensor:
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor)
            raise ValueError(f"{kind} layer {layer} expected [B,S,D], got {shape}")
        if layer in captures[kind]:
            raise RuntimeError(f"{kind} layer {layer} executed more than once")
        if start_graph and not tensor.requires_grad:
            tensor.requires_grad_(True)
        captures[kind][layer] = tensor
        return tensor

    try:
        for layer in layer_list:
            block = blocks[layer]
            num_heads, head_dim = geometry[layer]

            def mlp_hook(module, inputs, output, *, _layer=layer):
                del module, inputs
                if not torch.is_tensor(output):
                    raise TypeError(f"MLP layer {_layer} returned a non-tensor output")
                return prepare(output, kind="mlp", layer=_layer)

            def attention_pre_hook(module, inputs, *, _layer=layer, _heads=num_heads, _dim=head_dim):
                del module
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(f"Attention o_proj layer {_layer} has no tensor input")
                tensor = inputs[0]
                if tensor.ndim != 3 or tensor.shape[-1] != _heads * _dim:
                    raise ValueError(
                        f"Attention layer {_layer} pre-o_proj expected [B,S,{_heads * _dim}], "
                        f"got {tuple(tensor.shape)}"
                    )
                prepared = prepare(
                    tensor, kind="attention_pre_o_proj", layer=_layer
                )
                return (prepared, *inputs[1:])

            handles.append(block.mlp.register_forward_hook(mlp_hook))
            handles.append(
                block.self_attn.o_proj.register_forward_pre_hook(attention_pre_hook)
            )
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def component_grad_delta_scores(
    clean: Mapping[str, Mapping[int, torch.Tensor]],
    perturbed: Mapping[str, Mapping[int, torch.Tensor]],
    gradients: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    head_geometry: Mapping[int, tuple[int, int]],
) -> dict[str, list[dict[str, Any]]]:
    """Compute exact tensor-level ``clean-gradient dot perturbed-clean`` scores."""

    expected_kinds = {"mlp", "attention_pre_o_proj"}
    if set(clean) != expected_kinds or set(perturbed) != expected_kinds:
        raise ValueError(f"Component captures must contain exactly {expected_kinds}")
    if set(gradients) != expected_kinds:
        raise ValueError(f"Gradient captures must contain exactly {expected_kinds}")
    output: dict[str, list[dict[str, Any]]] = {
        "mlps": [],
        "attention_heads": [],
    }
    for kind in sorted(expected_kinds):
        if not (set(clean[kind]) == set(perturbed[kind]) == set(gradients[kind])):
            raise ValueError(f"{kind} layers do not align across clean/delta/gradient")
        for layer in sorted(clean[kind]):
            clean_tensor = clean[kind][layer].detach().float()
            perturbed_tensor = perturbed[kind][layer].detach().float()
            gradient = gradients[kind][layer].detach().float()
            if not (
                clean_tensor.shape == perturbed_tensor.shape == gradient.shape
                and clean_tensor.ndim == 3
            ):
                raise ValueError(
                    f"{kind} layer {layer} shape mismatch: clean={tuple(clean_tensor.shape)}, "
                    f"perturbed={tuple(perturbed_tensor.shape)}, "
                    f"gradient={tuple(gradient.shape)}"
                )
            if not (
                torch.isfinite(clean_tensor).all()
                and torch.isfinite(perturbed_tensor).all()
                and torch.isfinite(gradient).all()
            ):
                raise ValueError(f"{kind} layer {layer} contains non-finite values")
            delta = perturbed_tensor - clean_tensor
            if kind == "mlp":
                score_by_position = (gradient * delta).sum(dim=(0, 2))
                output["mlps"].append(
                    {
                        "component": f"L{layer}.MLP",
                        "layer": layer,
                        "score": float(score_by_position.sum().cpu()),
                        "abs_score": float(score_by_position.sum().abs().cpu()),
                        "score_by_position": [
                            float(value) for value in score_by_position.cpu()
                        ],
                        "delta_norm": float(delta.norm().cpu()),
                        "clean_gradient_norm": float(gradient.norm().cpu()),
                        "clean_activation_norm": float(clean_tensor.norm().cpu()),
                        "perturbed_activation_norm": float(
                            perturbed_tensor.norm().cpu()
                        ),
                        "capture": "MLP output",
                    }
                )
                continue

            try:
                num_heads, head_dim = head_geometry[layer]
            except KeyError as error:
                raise ValueError(f"Missing head geometry for layer {layer}") from error
            if clean_tensor.shape[-1] != num_heads * head_dim:
                raise ValueError(
                    f"Attention layer {layer} width {clean_tensor.shape[-1]} != "
                    f"{num_heads}*{head_dim}"
                )
            shape = (*clean_tensor.shape[:2], num_heads, head_dim)
            delta_heads = delta.reshape(shape)
            gradient_heads = gradient.reshape(shape)
            score_by_position_head = (gradient_heads * delta_heads).sum(dim=(0, 3))
            delta_norms = delta_heads.square().sum(dim=(0, 1, 3)).sqrt()
            gradient_norms = gradient_heads.square().sum(dim=(0, 1, 3)).sqrt()
            for head in range(num_heads):
                position_scores = score_by_position_head[:, head]
                score = position_scores.sum()
                output["attention_heads"].append(
                    {
                        "component": f"L{layer}.H{head}",
                        "layer": layer,
                        "head": head,
                        "score": float(score.cpu()),
                        "abs_score": float(score.abs().cpu()),
                        "score_by_position": [
                            float(value) for value in position_scores.cpu()
                        ],
                        "delta_norm": float(delta_norms[head].cpu()),
                        "clean_gradient_norm": float(gradient_norms[head].cpu()),
                        "capture": "attention stream before o_proj",
                    }
                )
    return output


def _metric_from_logits(
    logits: torch.Tensor,
    *,
    target_token_id: int,
    foil_token_id: int,
    behavior_position: int,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits [1,S,V], got {tuple(logits.shape)}")
    if target_token_id == foil_token_id:
        raise ValueError("Target and foil token IDs must differ")
    if not 0 <= target_token_id < logits.shape[-1]:
        raise IndexError("Target token ID outside model vocabulary")
    if not 0 <= foil_token_id < logits.shape[-1]:
        raise IndexError("Foil token ID outside model vocabulary")
    position = int(behavior_position)
    if position < 0:
        position += logits.shape[1]
    if not 0 <= position < logits.shape[1]:
        raise IndexError("Behavior position outside sequence")
    return (
        logits[0, position, int(target_token_id)].float()
        - logits[0, position, int(foil_token_id)].float()
    )


def localize_source_direction(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    direction: torch.Tensor,
    *,
    source_layer: int,
    target_token_id: int,
    foil_token_id: int,
    attention_mask: torch.Tensor | None = None,
    behavior_position: int = -1,
    intervention_positions: Sequence[int] | None = None,
    component_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Localize one single-layer concept-direction ablation downstream."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Localization requires one unpadded item with shape [1,S]")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if any(parameter.requires_grad for parameter in hf_model.parameters()):
        raise ValueError("Freeze model parameters before component localization")
    source_layer = int(source_layer)
    if not 0 <= source_layer < len(blocks) - 1:
        raise IndexError("Source layer must leave at least one downstream block")
    downstream = (
        list(range(source_layer + 1, len(blocks)))
        if component_layers is None
        else sorted(set(int(layer) for layer in component_layers))
    )
    if not downstream or any(
        layer <= source_layer or layer >= len(blocks) for layer in downstream
    ):
        raise ValueError("Every component layer must be strictly downstream of source")
    source_width = getattr(blocks[source_layer].self_attn.o_proj, "out_features", None)
    if source_width is not None and direction.numel() != int(source_width):
        raise ValueError(
            f"Direction width {direction.numel()} != source residual width {source_width}"
        )
    unit = direction.detach().float()
    if not torch.isfinite(unit).all() or not torch.isclose(
        unit.norm(), torch.ones((), device=unit.device), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Localization direction must be finite and unit norm")
    head_geometry = {
        layer: _validate_qwen_component_layer(blocks[layer], layer)
        for layer in downstream
    }

    ordered_keys = [
        *(('attention_pre_o_proj', layer) for layer in downstream),
        *(('mlp', layer) for layer in downstream),
    ]
    with torch.enable_grad(), capture_qwen_components(
        blocks, downstream, start_graph=True
    ) as clean_live:
        clean_logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        clean_metric_tensor = _metric_from_logits(
            clean_logits,
            target_token_id=target_token_id,
            foil_token_id=foil_token_id,
            behavior_position=behavior_position,
        )
        tensors = tuple(clean_live[kind][layer] for kind, layer in ordered_keys)
        gradient_tuple = torch.autograd.grad(
            clean_metric_tensor,
            tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        clean = {
            kind: {
                layer: tensor.detach()
                for layer, tensor in clean_live[kind].items()
            }
            for kind in clean_live
        }
        gradients: dict[str, dict[int, torch.Tensor]] = {
            "mlp": {},
            "attention_pre_o_proj": {},
        }
        for (kind, layer), gradient in zip(
            ordered_keys, gradient_tuple, strict=True
        ):
            gradients[kind][layer] = gradient.detach()

    edit = lambda hidden: ablate_direction(  # noqa: E731 - hook closure is local
        hidden, unit, positions=intervention_positions
    )
    with torch.no_grad(), residual_edit_hooks(
        blocks, {source_layer: edit}
    ), capture_qwen_components(blocks, downstream, start_graph=False) as perturbed_live:
        perturbed_logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        perturbed_metric_tensor = _metric_from_logits(
            perturbed_logits,
            target_token_id=target_token_id,
            foil_token_id=foil_token_id,
            behavior_position=behavior_position,
        )
        perturbed = {
            kind: {
                layer: tensor.detach()
                for layer, tensor in perturbed_live[kind].items()
            }
            for kind in perturbed_live
        }

    scores = component_grad_delta_scores(
        clean,
        perturbed,
        gradients,
        head_geometry=head_geometry,
    )
    clean_metric = float(clean_metric_tensor.detach().cpu())
    perturbed_metric = float(perturbed_metric_tensor.detach().cpu())
    actual_delta = perturbed_metric - clean_metric
    naive_sum = sum(
        float(row["score"])
        for kind in ("mlps", "attention_heads")
        for row in scores[kind]
    )
    return {
        "source_layer": source_layer,
        "component_layers": downstream,
        "intervention": {
            "type": "single-source-layer direction ablation",
            "positions": (
                "all prompt positions"
                if intervention_positions is None
                else [int(position) for position in intervention_positions]
            ),
            "direction_norm": float(unit.norm().cpu()),
        },
        "behavior_metric": "logit(target) - logit(foil)",
        "clean_metric": clean_metric,
        "perturbed_metric": perturbed_metric,
        "actual_delta": actual_delta,
        "positive_damage": -actual_delta,
        "localization_estimator": {
            "formula": "grad(M_clean, component) dot (component_ablated - component_clean)",
            "gradient_point": "clean forward pass",
            "delta_order": "single-source-ablated minus clean",
            "warning": NON_ADDITIVE_WARNING,
            "naive_overlapping_score_sum_do_not_interpret": naive_sum,
            "naive_sum_minus_actual_delta": naive_sum - actual_delta,
        },
        **scores,
    }


def flag_top_components(
    localization: Mapping[str, Any],
    *,
    top_k_mlps: int = 4,
    top_k_heads: int = 8,
) -> dict[str, Any]:
    """Flag the largest absolute localization scores with deterministic ties."""

    if top_k_mlps < 1 or top_k_heads < 1:
        raise ValueError("top-k counts must be positive")
    mlps = [dict(row) for row in localization.get("mlps", [])]
    heads = [dict(row) for row in localization.get("attention_heads", [])]
    if not mlps or not heads:
        raise ValueError("Localization has no downstream MLP/head scores")
    mlps.sort(key=lambda row: (-float(row["abs_score"]), int(row["layer"])))
    heads.sort(
        key=lambda row: (
            -float(row["abs_score"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    selected_mlps = mlps[: min(top_k_mlps, len(mlps))]
    selected_heads = heads[: min(top_k_heads, len(heads))]
    for rank, row in enumerate(selected_mlps, start=1):
        row["attribution_abs_rank"] = rank
    for rank, row in enumerate(selected_heads, start=1):
        row["attribution_abs_rank"] = rank
    return {
        "mlps": selected_mlps,
        "attention_heads": selected_heads,
        "selection": {
            "metric": "absolute non-additive localization score",
            "top_k_mlps_requested": top_k_mlps,
            "top_k_heads_requested": top_k_heads,
            "tie_break": "lower layer, then lower head",
            "warning": NON_ADDITIVE_WARNING,
        },
    }


@torch.no_grad()
def qwen_attention_weight_read_with_null(
    attention: torch.nn.Module,
    direction: torch.Tensor,
    *,
    label_direction: torch.Tensor | None = None,
    n_random: int = 128,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Add seeded random-direction normalization to Qwen per-head OV READ."""

    if n_random < 1:
        raise ValueError("n_random must be positive")
    observed = qwen_head_ov_read(
        attention,
        direction,
        label_direction=label_direction,
    )
    width = int(direction.numel())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_vectors = F.normalize(
        torch.randn(n_random, width, generator=generator, dtype=torch.float32),
        dim=-1,
    )
    num_heads = int(attention.config.num_attention_heads)
    num_kv_heads = int(attention.config.num_key_value_heads)
    head_dim = int(attention.head_dim)
    if num_heads % num_kv_heads:
        raise ValueError("Query-head count must be divisible by KV-head count")
    if len(observed) != num_heads:
        raise RuntimeError("Qwen OV helper returned an inconsistent head count")
    v_weight = attention.v_proj.weight.detach().float()
    o_weight = attention.o_proj.weight.detach().float()
    if v_weight.shape != (num_kv_heads * head_dim, width):
        raise ValueError(
            "Qwen value weight shape disagrees with its GQA head configuration"
        )
    if o_weight.shape[1] != num_heads * head_dim:
        raise ValueError(
            "Qwen output weight shape disagrees with its query-head configuration"
        )
    random_value = random_vectors.to(v_weight.device) @ v_weight.T
    random_value = random_value.reshape(n_random, num_kv_heads, head_dim)
    random_value = random_value.repeat_interleave(
        num_heads // num_kv_heads, dim=1
    )
    output_by_head = o_weight.reshape(o_weight.shape[0], num_heads, head_dim)
    output_by_head = output_by_head.permute(1, 0, 2)
    random_outputs = torch.einsum(
        "rhd,hod->rho",
        random_value.to(output_by_head.device),
        output_by_head,
    )
    random_ov_norms = random_outputs.norm(dim=-1)
    label = F.normalize(
        (label_direction if label_direction is not None else direction)
        .detach()
        .float(),
        dim=0,
    ).to(random_outputs.device)
    if label.numel() != random_outputs.shape[-1]:
        raise ValueError("Label direction width disagrees with attention output width")
    random_label_cosines = torch.einsum("rho,o->rh", random_outputs, label)
    random_label_cosines = random_label_cosines / random_ov_norms
    random_ov_norms = random_ov_norms.cpu().numpy()
    random_label_cosines = random_label_cosines.cpu().numpy()

    result: list[dict[str, Any]] = []
    for row in observed:
        head = int(row["head"])
        ov_null = np.asarray(random_ov_norms[:, head], dtype=np.float64)
        cosine_null = np.asarray(random_label_cosines[:, head], dtype=np.float64)
        if not np.isfinite(ov_null).all():
            raise ValueError(f"Head {head} OV random null is non-finite")
        finite_cosine = cosine_null[np.isfinite(cosine_null)]
        median = float(np.median(ov_null))
        observed_norm = float(row["ov_norm"])
        observed_cosine = float(row["label_cosine"])
        result.append(
            {
                **row,
                "normalized_ov_norm": (
                    observed_norm / median if median > 0.0 else None
                ),
                "random_median_ov_norm": median,
                "random_ov_norms": [float(value) for value in ov_null],
                "ov_norm_random_percentile": float(
                    np.mean(ov_null <= observed_norm)
                ),
                "random_label_cosines": [float(value) for value in cosine_null],
                "label_cosine_random_percentile": (
                    float(np.mean(finite_cosine <= observed_cosine))
                    if finite_cosine.size
                    else None
                ),
                "label_weighted_normalized_ov": (
                    observed_norm / median * abs(observed_cosine)
                    if median > 0.0 and math.isfinite(observed_cosine)
                    else None
                ),
                "n_random": n_random,
                "seed": int(seed),
            }
        )
    return result


def _derived_seed(seed: int, *, kind: str, layer: int) -> int:
    offsets = {"mlp": 10_000, "attention": 20_000}
    try:
        offset = offsets[kind]
    except KeyError as error:
        raise ValueError(f"Unknown component kind {kind!r}") from error
    return int(seed + offset + 97 * int(layer))


def weight_read_for_flagged_components(
    blocks: Sequence[torch.nn.Module],
    direction: torch.Tensor,
    flagged: Mapping[str, Any],
    *,
    label_direction: torch.Tensor | None = None,
    n_random: int = 128,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute activation-independent weight READ for flagged components."""

    if n_random < 1:
        raise ValueError("n_random must be positive")
    unit = F.normalize(direction.detach().float(), dim=0)
    if not torch.isfinite(unit).all():
        raise ValueError("Weight READ direction is non-finite")
    mlp_flags = [dict(row) for row in flagged.get("mlps", [])]
    head_flags = [dict(row) for row in flagged.get("attention_heads", [])]
    if not mlp_flags or not head_flags:
        raise ValueError("Flagged components must include MLPs and attention heads")

    mlp_rows: list[dict[str, Any]] = []
    for flag in mlp_flags:
        layer = int(flag["layer"])
        if not 0 <= layer < len(blocks):
            raise IndexError(f"Flagged MLP layer {layer} outside model")
        component_seed = _derived_seed(seed, kind="mlp", layer=layer)
        weight = qwen_mlp_gain(
            blocks[layer],
            unit,
            n_random=n_random,
            seed=component_seed,
        )
        null = np.asarray(weight["random_gains"], dtype=np.float64)
        observed = float(weight["gain"])
        mlp_rows.append(
            {
                **flag,
                **weight,
                "gain_random_percentile": float(np.mean(null <= observed)),
                "n_random": n_random,
                "seed": component_seed,
                "weight_metric": "normalized Qwen MLP response norm",
            }
        )

    flags_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for flag in head_flags:
        flags_by_layer[int(flag["layer"])].append(flag)
    head_rows: list[dict[str, Any]] = []
    for layer in sorted(flags_by_layer):
        if not 0 <= layer < len(blocks):
            raise IndexError(f"Flagged attention layer {layer} outside model")
        component_seed = _derived_seed(seed, kind="attention", layer=layer)
        all_weights = qwen_attention_weight_read_with_null(
            blocks[layer].self_attn,
            unit,
            label_direction=label_direction,
            n_random=n_random,
            seed=component_seed,
        )
        weights_by_head = {int(row["head"]): row for row in all_weights}
        for flag in flags_by_layer[layer]:
            head = int(flag["head"])
            if head not in weights_by_head:
                raise IndexError(f"Flagged head L{layer}.H{head} outside attention")
            head_rows.append(
                {
                    **flag,
                    **weights_by_head[head],
                    "weight_metric": (
                        "random-normalized OV norm with label preservation"
                    ),
                }
            )
    return {
        "mlps": mlp_rows,
        "attention_heads": head_rows,
        "metadata": {
            "activation_independent": True,
            "direction": "same raw W_UJ source-layer unit direction",
            "label_direction": (
                "same raw W_UJ source-layer direction"
                if label_direction is None
                else "explicit supplied label direction"
            ),
            "n_random": n_random,
            "base_seed": int(seed),
            "raw_random_nulls_retained": True,
            "mlp_formula": (
                "||MLP(post_attention_RMSNorm(v))|| / median random-unit norm"
            ),
            "attention_formula": (
                "||W_O^head W_V^kv v|| / median random-unit OV norm; "
                "label cosine also retained"
            ),
        },
    }


def spearman_rank_agreement(
    rows: Sequence[Mapping[str, Any]],
    *,
    attribution_key: str = "abs_score",
    weight_key: str,
) -> dict[str, Any]:
    """Spearman agreement with transparent component rankings."""

    records: list[dict[str, Any]] = []
    for row in rows:
        if attribution_key not in row or weight_key not in row:
            continue
        if row[attribution_key] is None or row[weight_key] is None:
            continue
        attribution = float(row[attribution_key])
        weight = float(row[weight_key])
        if math.isfinite(attribution) and math.isfinite(weight):
            records.append(
                {
                    "component_id": str(row.get("global_component_id", row["component"])),
                    "attribution": attribution,
                    "weight": weight,
                }
            )
    if len(records) < 3:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "fewer than three finite paired components",
            "n": len(records),
        }
    attribution = np.asarray([row["attribution"] for row in records])
    weight = np.asarray([row["weight"] for row in records])
    if float(np.ptp(attribution)) == 0.0 or float(np.ptp(weight)) == 0.0:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "constant attribution or weight metric",
            "n": len(records),
        }
    attribution_ranks = _average_percentile_ranks(attribution)
    weight_ranks = _average_percentile_ranks(weight)
    rho = float(np.corrcoef(attribution_ranks, weight_ranks)[0, 1])
    for index, record in enumerate(records):
        record["attribution_percentile_rank"] = float(attribution_ranks[index])
        record["weight_percentile_rank"] = float(weight_ranks[index])
    records.sort(key=lambda row: (-row["attribution"], row["component_id"]))
    return {
        "status": "ESTIMATED",
        "n": len(records),
        "spearman_rho": rho,
        "attribution_metric": attribution_key,
        "weight_metric": weight_key,
        "paired_ranks": records,
    }


def weight_attribution_agreement(
    localization_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool flagged components and compare attribution and weight rankings."""

    mlps: list[dict[str, Any]] = []
    heads: list[dict[str, Any]] = []
    for record in localization_records:
        prefix = f"{record['name']}|L{record['source_layer']}"
        weights = record["weight_read"]
        for row in weights["mlps"]:
            mlps.append(
                {
                    **row,
                    "global_component_id": f"{prefix}|{row['component']}",
                }
            )
        for row in weights["attention_heads"]:
            heads.append(
                {
                    **row,
                    "global_component_id": f"{prefix}|{row['component']}",
                }
            )
    return {
        "mlp_attribution_vs_normalized_gain": spearman_rank_agreement(
            mlps, weight_key="normalized_gain"
        ),
        "head_attribution_vs_normalized_ov": spearman_rank_agreement(
            heads, weight_key="normalized_ov_norm"
        ),
        "head_attribution_vs_label_weighted_ov": spearman_rank_agreement(
            heads, weight_key="label_weighted_normalized_ov"
        ),
        "scope": "flagged top components pooled across selected items and sources",
        "guardrail": (
            "Agreement is descriptive and selection-conditioned because components "
            "were flagged by attribution magnitude before weight evaluation."
        ),
    }


def choose_f4_candidates(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Choose a measured high-READ driver candidate and a low-READ contrast."""

    rows = list(selection.get("selected", []))
    if len(rows) < 2:
        raise ValueError("F4 requires at least two selected rows")
    high_read = [row for row in rows if row["cell_read_class"] == "high"]
    low_read = [row for row in rows if row["cell_read_class"] == "low"]
    if not high_read or not low_read:
        raise ValueError("F4 candidates require both high- and low-READ strata")

    def causal_damage(row: Mapping[str, Any]) -> float:
        return float(row["row"].get("ablation", {}).get("positive_damage", -math.inf))

    high_read.sort(
        key=lambda row: (
            -causal_damage(row),
            -float(row["read_strength"]),
            str(row["name"]),
        )
    )
    preferred_low = [
        row for row in low_read if row["cell_write_class"] == "high"
    ] or low_read
    preferred_low.sort(
        key=lambda row: (
            float(row["read_strength"]),
            -float(row["write_strength"]),
            str(row["name"]),
        )
    )
    driver = high_read[0]
    low = next((row for row in preferred_low if row["name"] != driver["name"]), None)
    if low is None:
        raise ValueError("Could not choose distinct F4 candidates")
    return {
        "driver_candidate": {
            "name": driver["name"],
            "selection_cell": driver["cell"],
            "write_strength": driver["write_strength"],
            "read_strength": driver["read_strength"],
            "all_band_ablation_positive_damage": causal_damage(driver),
            "selection_rule": (
                "largest measured all-band ablation damage within the sampled "
                "high-READ strata; attribution READ and name break ties"
            ),
        },
        "low_read_candidate": {
            "name": low["name"],
            "selection_cell": low["cell"],
            "write_strength": low["write_strength"],
            "read_strength": low["read_strength"],
            "all_band_ablation_positive_damage": causal_damage(low),
            "selection_rule": (
                "lowest attribution READ, preferring the high-WRITE/low-READ cell; "
                "WRITE and name break ties"
            ),
        },
        "guardrail": (
            "These are visualization candidates selected after notebook-02 "
            "measurements. The low-READ candidate is not declared narration, and "
            "the driver label is not an independently validated class assignment."
        ),
    }


def _localization_matrix(
    localization: Mapping[str, Any],
    layers: Sequence[int],
    num_heads: int,
) -> np.ndarray:
    matrix = np.full((num_heads + 1, len(layers)), np.nan, dtype=np.float64)
    layer_index = {int(layer): index for index, layer in enumerate(layers)}
    for row in localization["mlps"]:
        layer = int(row["layer"])
        if layer in layer_index:
            matrix[0, layer_index[layer]] = float(row["score"])
    for row in localization["attention_heads"]:
        layer = int(row["layer"])
        head = int(row["head"])
        if layer in layer_index and 0 <= head < num_heads:
            matrix[head + 1, layer_index[layer]] = float(row["score"])
    return matrix


def plot_f4_localization(
    localization_records: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Save F4: non-additive head/MLP localization for two candidates."""

    panels: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for role in ("driver_candidate", "low_read_candidate"):
        candidate = candidates[role]
        matches = [
            record
            for record in localization_records
            if record["name"] == candidate["name"]
        ]
        if not matches:
            raise ValueError(f"No localization record for F4 {role}")
        matches.sort(key=lambda record: int(record["source_selection_rank"]))
        panels.append((role, candidate, matches[0]))

    layers = sorted(
        {
            int(layer)
            for _, _, record in panels
            for layer in record["localization"]["component_layers"]
        }
    )
    num_heads = max(
        int(row["head"]) + 1
        for _, _, record in panels
        for row in record["localization"]["attention_heads"]
    )
    matrices = [
        _localization_matrix(record["localization"], layers, num_heads)
        for _, _, record in panels
    ]
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    limit = float(np.quantile(np.abs(finite), 0.99)) if finite.size else 1.0
    if not math.isfinite(limit) or limit == 0.0:
        limit = 1.0

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(13.0, 0.55 * len(layers) * 2), max(7.0, 0.25 * num_heads)),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, matrix, (role, candidate, record) in zip(
        axes, matrices, panels, strict=True
    ):
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xticks(range(len(layers)), [str(layer) for layer in layers], rotation=90)
        axis.set_xlabel("downstream block")
        axis.set_title(
            f"{role.replace('_', ' ')}\n{candidate['name']} "
            f"(source L{record['source_layer']})"
        )
    axes[0].set_yticks(
        range(num_heads + 1), ["MLP", *[f"H{head}" for head in range(num_heads)]]
    )
    axes[0].set_ylabel("captured component")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            shrink=0.82,
            label=(
                "grad(M_clean) · (single-source-ablated − clean)\n"
                "non-additive localization score"
            ),
        )
    figure.suptitle(
        "F4 — downstream READ localization (candidate contrast; no class assignment)"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target.resolve()


def _selection_without_rows(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected": [
            {key: value for key, value in item.items() if key != "row"}
            for item in selection["selected"]
        ],
        "provenance": selection["provenance"],
    }


def run_localization_phase(
    bundle: Any,
    lens: Any,
    twohop_payload: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    figure_path: str | Path | None = None,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    n_per_cell: int = 1,
    n_source_layers: int = 1,
    top_k_mlps: int = 4,
    top_k_heads: int = 8,
    n_random: int = 128,
    max_length: int = 128,
    seed: int = SEED,
) -> dict[str, Any]:
    """Notebook-04 orchestration over successful notebook-02 raw rows."""

    metadata = twohop_payload.get("metadata")
    rows = twohop_payload.get("rows")
    if not isinstance(metadata, Mapping) or not isinstance(rows, Sequence):
        raise ValueError("Expected a notebook-02 payload with metadata and rows")
    if metadata.get("primary_direction") != PRIMARY_DIRECTION_METHOD:
        raise ValueError("Notebook-02 payload does not declare raw W_UJ as primary")
    if metadata.get("rms_gain_folded_included") is not False:
        raise ValueError("Localization requires the unfurled raw W_UJ primary payload")
    if metadata.get("model_id") != bundle.model_id:
        raise ValueError("Notebook-02 model ID does not match loaded model")
    if metadata.get("model_revision") != bundle.revision:
        raise ValueError("Notebook-02 model revision does not match loaded model")
    if max_length < 1:
        raise ValueError("max_length must be positive")

    set_seed(seed)
    selection = select_localization_subset(
        rows,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_per_cell=n_per_cell,
    )
    blocks = bundle.lens_model.layers
    plans: list[dict[str, Any]] = []
    for item in selection["selected"]:
        source = choose_source_layers(
            item["row"],
            n_source_layers=n_source_layers,
            max_source_layer=len(blocks) - 2,
        )
        plans.append({"selection": item, "source": source})

    token_ids = {
        int(plan["selection"]["row"]["token_ids"]["concept"]) for plan in plans
    }
    source_layers = {
        int(layer) for plan in plans for layer in plan["source"]["selected_layers"]
    }
    device = next(bundle.hf_model.parameters()).device
    direction_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        source_layers,
        fold_rms_gain=False,
        compute_device=device,
        output_device=device,
    )

    localization_records: list[dict[str, Any]] = []
    for plan in plans:
        sampled = plan["selection"]
        row = sampled["row"]
        encoded = bundle.tokenizer(
            str(row["prompt"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        untruncated = len(
            bundle.tokenizer.encode(str(row["prompt"]), add_special_tokens=True)
        )
        if untruncated > max_length:
            raise ValueError(f"Refusing truncated localization for {row['name']!r}")
        if [int(value) for value in input_ids[0].detach().cpu()] != [
            int(value) for value in row["prompt_token_ids"]
        ]:
            raise ValueError(
                f"Tokenization drift relative to notebook 02 for {row['name']!r}"
            )
        concept_token_id = int(row["token_ids"]["concept"])
        for source_rank, source_layer in enumerate(
            plan["source"]["selected_layers"], start=1
        ):
            direction = direction_bank[concept_token_id][int(source_layer)]
            localization = localize_source_direction(
                bundle.hf_model,
                blocks,
                input_ids,
                direction,
                source_layer=int(source_layer),
                target_token_id=int(row["token_ids"]["target"]),
                foil_token_id=int(row["token_ids"]["foil"]),
                attention_mask=attention_mask,
            )
            flagged = flag_top_components(
                localization,
                top_k_mlps=top_k_mlps,
                top_k_heads=top_k_heads,
            )
            component_seed = seed + 1_000_000 * len(localization_records)
            weight_read = weight_read_for_flagged_components(
                blocks,
                direction,
                flagged,
                label_direction=direction,
                n_random=n_random,
                seed=component_seed,
            )
            localization_records.append(
                {
                    "name": str(row["name"]),
                    "source": row.get("source"),
                    "category": row.get("category"),
                    "prompt": row["prompt"],
                    "intermediate": row["intermediate"],
                    "concept_token_id": concept_token_id,
                    "concept_token_surface": row["token_surfaces"]["concept"],
                    "target_token_id": int(row["token_ids"]["target"]),
                    "foil_token_id": int(row["token_ids"]["foil"]),
                    "direction_method": PRIMARY_DIRECTION_METHOD,
                    "direction_formula": "normalize(W_U[token] @ J_source_layer)",
                    "source_layer": int(source_layer),
                    "source_selection_rank": source_rank,
                    "source_selection": next(
                        item
                        for item in plan["source"]["selected"]
                        if int(item["layer"]) == int(source_layer)
                    ),
                    "sampling_cell": sampled["cell"],
                    "notebook02_summary": {
                        "write_strength": sampled["write_strength"],
                        "read_strength": sampled["read_strength"],
                        "all_band_ablation_positive_damage": float(
                            row["ablation"]["positive_damage"]
                        ),
                    },
                    "localization": localization,
                    "flagged_components": flagged,
                    "weight_read": weight_read,
                }
            )

    agreement = weight_attribution_agreement(localization_records)
    candidates = choose_f4_candidates(selection)
    model_slug = bundle.model_id.split("/")[-1].lower().replace("-instruct", "")
    figure_target = (
        Path(figure_path)
        if figure_path is not None
        else ROOT / "results/figures" / f"f4_read_localization_{model_slug}.png"
    )
    f4 = plot_f4_localization(localization_records, candidates, figure_target)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPUTED",
        "metadata": {
            "model_id": bundle.model_id,
            "model_revision": bundle.revision,
            "seed": seed,
            "source_payload_schema": twohop_payload.get("schema_version"),
            "source_payload_status": twohop_payload.get("status"),
            "primary_direction": PRIMARY_DIRECTION_METHOD,
            "primary_direction_formula": "normalize(W_U[token] @ J_layer)",
            "fold_rms_gain": False,
            "subset_role": "deterministic descriptive localization subset",
            "component_score_warning": NON_ADDITIVE_WARNING,
            "weight_read_activation_independent": True,
            "n_random": n_random,
            "raw_weight_nulls_retained": True,
        },
        "selection": _selection_without_rows(selection),
        "source_layer_selection": {
            "n_source_layers_per_item": n_source_layers,
            "formula": "abs(sum_position(WRITE * attribution_READ))",
            "plans": [
                {
                    "name": plan["selection"]["name"],
                    **plan["source"],
                }
                for plan in plans
            ],
        },
        "sample_counts": {
            "n_notebook02_raw_success": selection["provenance"][
                "n_eligible_raw_rows"
            ],
            "n_selected_items": len(selection["selected"]),
            "n_localization_records": len(localization_records),
            "n_flagged_mlps": sum(
                len(record["weight_read"]["mlps"])
                for record in localization_records
            ),
            "n_flagged_attention_heads": sum(
                len(record["weight_read"]["attention_heads"])
                for record in localization_records
            ),
        },
        "f4_candidates": candidates,
        "localizations": localization_records,
        "attribution_weight_rank_agreement": agreement,
        "figures": {"f4": _relative_or_absolute(f4)},
        "interpretation_guardrail": (
            "No a priori narration class is assigned. Component localization is "
            "non-additive, weight agreement is descriptive and selection-conditioned, "
            "and candidates remain candidates until real intervention evidence is interpreted."
        ),
    }
    strict_payload = _json_ready(payload)
    destination = (
        Path(output_path)
        if output_path is not None
        else ROOT / "data/raw" / f"04_localization_{model_slug}.json"
    )
    save_json(destination, strict_payload)
    print(
        f"LOCALIZATION COMPUTED: items={len(selection['selected'])}, "
        f"sources={len(localization_records)}, direction={PRIMARY_DIRECTION_METHOD}"
    )
    print(NON_ADDITIVE_WARNING)
    return strict_payload


def run_qwen_localization_phase(
    *,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    twohop_path: str | Path | None = None,
    lens_path: str | Path | None = None,
    **phase_kwargs: Any,
) -> dict[str, Any]:
    """Model-loading notebook-04 entry point."""

    if not model_id.startswith("Qwen/Qwen2.5-"):
        raise ValueError("Localization entry point is restricted to Qwen2.5")
    model_slug = model_id.split("/")[-1].lower().replace("-instruct", "")
    source = (
        Path(twohop_path)
        if twohop_path is not None
        else ROOT / "data/raw" / f"02_twohop_{model_slug}.json"
    )
    with source.open(encoding="utf-8") as handle:
        twohop_payload = json.load(handle)
    bundle = load_model(model_id)
    try:
        if lens_path is None:
            if model_id != "Qwen/Qwen2.5-7B-Instruct":
                raise ValueError(f"lens_path is required for {model_id}")
            lens = load_published_lens(model_id)
        else:
            lens = load_local_lens(lens_path)
        return run_localization_phase(
            bundle,
            lens,
            twohop_payload,
            **phase_kwargs,
        )
    finally:
        release_model(bundle)

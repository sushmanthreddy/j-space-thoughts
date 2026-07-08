"""Expensive causal ground truth for the final symmetric READ experiment.

This module is the intervention side of the anti-circularity firewall.  It
captures clean post-block residual states, performs a complete residual-state
interchange at the explicit concept token, and reports the signed, unclipped
two-direction causal score ``C``.  The cheap gradient estimator lives in
``src.cheap_read`` and neither module imports the other.

Only the full-residual intervention used as final causal truth is implemented
here.  Superseded concept-subspace diagnostics and post-GO circuit localization
remain in the archived experiments rather than the final pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from statistics import median
from typing import Any, Literal, Protocol, TypedDict

import torch
from torch import nn


TaskKind = Literal["engine", "dashboard"]
TensorEdit = Callable[[torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor], torch.Tensor]


class TokenizerLike(Protocol):
    """Tokenizer surface required by the causal row builders."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        """Encode one prompt using the exact tokenizer surface required here."""

        ...


class LensModelLike(Protocol):
    """J-Lens adapter surface required by the causal row builders."""

    layers: Sequence[nn.Module]


class CausalModelBundle(Protocol):
    """Structural type shared with the repository's loaded model bundle."""

    hf_model: nn.Module
    tokenizer: TokenizerLike
    lens_model: LensModelLike


class CleanStateRecord(TypedDict):
    """One clean forward and its exact residual-donor provenance."""

    state: torch.Tensor
    logits: torch.Tensor
    layer: int
    requested_position: int
    resolved_position: int
    sequence_length: int
    input_token_ids: list[list[int]]


class CausalMeasurement(TypedDict):
    """JSON-ready scalar result of one symmetric full-residual interchange."""

    status: Literal["OK"]
    pair_id: str
    task_kind: TaskKind
    variant: Literal["full_residual"]
    eligible_as_primary_truth: bool
    layer: int
    position_rule: str
    position_a: int
    position_b: int
    metric_a: float
    metric_b: float
    metric_a_from_b: float
    metric_b_from_a: float
    T: float
    normalization_source: str
    R_a_from_b: float
    R_b_from_a: float
    C: float
    directional_abs_difference: float
    sharp_disagreement_threshold: float
    sharp_directional_disagreement: bool
    signed_unclipped: bool
    clean_top_token_id_a: int
    clean_top_token_id_b: int
    edited_top_token_id_a_from_b: int
    edited_top_token_id_b_from_a: int


class FullResidualResult(TypedDict):
    """Named container retained for compatibility with persisted result rows."""

    full_residual: CausalMeasurement


class BaseCausalRow(TypedDict):
    """Engine and original-dashboard causal truth for one verified pair."""

    pair_id: str
    dependency_group: str
    fold: int
    category: str
    concept_a: str
    concept_b: str
    answer_a: str
    answer_b: str
    engine: FullResidualResult
    dashboard: FullResidualResult


class HardDashboardCausalRow(TypedDict):
    """Answer-type-matched dashboard truth joined to its frozen engine."""

    pair_id: str
    dependency_group: str
    fold: int
    category: str
    hard_template_id: str
    frozen_engine_C: float
    frozen_engine_T: float
    hard_dashboard: CausalMeasurement


__all__ = [
    "BaseCausalRow",
    "CausalMeasurement",
    "CausalModelBundle",
    "CleanStateRecord",
    "HardDashboardCausalRow",
    "MetricFn",
    "clean_state_and_logits",
    "compute_base_causal_rows",
    "compute_hard_dashboard_causal_rows",
    "full_residual_interchange_edit",
    "summarize_base_causal_sanity",
    "summarize_hard_dashboard_causal_sanity",
    "symmetric_interchange",
    "token_difference_metric",
]


def _hidden_from_output(output: Any) -> torch.Tensor:
    """Extract the residual tensor from a decoder block's output."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder-block output type: {type(output).__name__}")


def _replace_hidden(output: Any, edit: TensorEdit) -> Any:
    """Edit tensor/tuple decoder outputs without dropping auxiliary values."""

    if torch.is_tensor(output):
        return edit(output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (edit(output[0]), *output[1:])
    raise TypeError(f"Unsupported decoder-block output type: {type(output).__name__}")


@contextmanager
def _residual_edit_hooks(
    blocks: Sequence[nn.Module],
    edits: Mapping[int, TensorEdit],
) -> Iterator[None]:
    """Install sorted post-block edits and always remove their hooks."""

    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for raw_layer, edit in sorted(edits.items()):
            layer = int(raw_layer)
            if not 0 <= layer < len(blocks):
                raise IndexError(f"Layer {layer} outside range [0, {len(blocks)})")

            def hook(
                module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                _edit: TensorEdit = edit,
            ) -> Any:
                """Apply the edit captured for this decoder block."""

                del module, inputs
                return _replace_hidden(output, _edit)

            handles.append(blocks[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _forward_logits(
    hf_model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    blocks: Sequence[nn.Module] | None = None,
    edits: Mapping[int, TensorEdit] | None = None,
) -> torch.Tensor:
    """Run the exact HF forward with optional residual edits in fp32 logits."""

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if edits:
        if blocks is None:
            raise ValueError("blocks must be supplied when edits are requested")
        with _residual_edit_hooks(blocks, edits):
            return hf_model(**kwargs).logits.float()
    return hf_model(**kwargs).logits.float()


def token_difference_metric(
    positive_token_id: int,
    negative_token_id: int,
) -> MetricFn:
    """Return the exact final-position positive-minus-negative logit metric."""

    positive = int(positive_token_id)
    negative = int(negative_token_id)
    if positive == negative:
        raise ValueError("Behavior metric tokens must differ")

    def metric(logits: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured positive-minus-negative logit difference."""

        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("Symmetric causal metrics require logits shaped [1,S,V]")
        return logits[0, -1, positive].float() - logits[0, -1, negative].float()

    return metric


def _position_index(sequence_length: int, position: int) -> int:
    """Resolve one possibly-negative position against a sequence length."""

    index = int(position)
    if index < 0:
        index += sequence_length
    if not 0 <= index < sequence_length:
        raise IndexError(f"Position {position} outside sequence length {sequence_length}")
    return index


@torch.no_grad()
def clean_state_and_logits(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    layer: int,
    *,
    position: int = -1,
) -> CleanStateRecord:
    """Capture a clean post-block donor state and logits in the same forward.

    The record binds the donor to its complete input-token sequence, source
    layer, and resolved token position.  :func:`symmetric_interchange` checks
    that provenance before allowing the state to be installed in another run.
    The residual donor and logits are converted to fp32 at the same points as
    in the frozen experiment; installation casts the donor back to model dtype.
    """

    layer = int(layer)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Clean donor capture requires input IDs shaped [1,S]")
    if not 0 <= layer < len(blocks):
        raise IndexError(f"Layer {layer} outside range [0, {len(blocks)})")
    if hf_model.training:
        raise ValueError("Clean donor capture requires eval mode")

    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        """Capture the selected decoder block's clean residual output."""

        del module, inputs
        captured["hidden"] = _hidden_from_output(output).detach()
        return output

    handle = blocks[layer].register_forward_hook(hook)
    try:
        logits = hf_model(input_ids=input_ids, use_cache=False).logits.float()
    finally:
        handle.remove()

    residual = captured.get("hidden")
    if residual is None:
        raise RuntimeError("Selected-layer clean state was not captured")
    index = _position_index(residual.shape[1], position)
    if residual.shape[0] != 1:
        raise ValueError("Clean donor capture requires batch size one")
    state = residual[0, index].detach().float()
    if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
        raise ValueError("Clean forward contains non-finite values")
    return {
        "state": state,
        "logits": logits,
        "layer": layer,
        "requested_position": int(position),
        "resolved_position": int(index),
        "sequence_length": int(input_ids.shape[1]),
        "input_token_ids": input_ids.detach().cpu().tolist(),
    }


def full_residual_interchange_edit(
    donor_state: torch.Tensor,
    *,
    position: int = -1,
) -> TensorEdit:
    """Create an edit that copies a clean donor's complete residual vector."""

    donor = donor_state.detach().float()
    if donor.ndim != 1 or not torch.isfinite(donor).all():
        raise ValueError("donor_state must be one finite vector")

    def edit(hidden: torch.Tensor) -> torch.Tensor:
        """Install the donor vector at the configured receiver position."""

        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != donor.numel():
            raise ValueError("Full interchange expects hidden [1,S,D] matching the donor")
        index = _position_index(hidden.shape[1], position)
        output = hidden.clone()
        output[0, index] = donor.to(hidden.device, hidden.dtype)
        return output

    return edit


def _validated_clean_tensors(
    record: Mapping[str, Any],
    input_ids: torch.Tensor,
    *,
    side: Literal["A", "B"],
    layer: int,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate donor provenance and return its state and fp32 clean logits."""

    expected_ids = input_ids.detach().cpu().tolist()
    if record.get("input_token_ids") != expected_ids:
        raise ValueError(f"Clean {side} record belongs to different input tokens")
    if int(record.get("layer", -1)) != layer:
        raise ValueError(f"Clean {side} record belongs to a different layer")
    if int(record.get("sequence_length", -1)) != int(input_ids.shape[1]):
        raise ValueError(f"Clean {side} record belongs to a different sequence length")
    expected_position = _position_index(input_ids.shape[1], position)
    if int(record.get("resolved_position", -1)) != expected_position:
        raise ValueError(f"Clean {side} record belongs to a different position")

    state = record.get("state")
    logits = record.get("logits")
    if not torch.is_tensor(state) or state.ndim != 1 or not torch.isfinite(state).all():
        raise ValueError(f"Clean {side} record has an invalid residual state")
    if (
        not torch.is_tensor(logits)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or not torch.isfinite(logits).all()
    ):
        raise ValueError(f"Clean {side} record has invalid logits")
    return state, logits.float()


@torch.no_grad()
def symmetric_interchange(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    clean_record_a: Mapping[str, Any],
    clean_record_b: Mapping[str, Any],
    metric_fn: MetricFn,
    *,
    pair_id: str,
    task_kind: TaskKind,
    layer: int,
    position_a: int,
    position_b: int,
    normalization_t: float | None = None,
    sharp_disagreement_threshold: float = 0.50,
) -> CausalMeasurement:
    """Measure signed symmetric full-residual interchange and return unclipped ``C``.

    For engines, ``T = M_A - M_B`` is derived from the two clean engine runs.
    Dashboards must receive that matched engine ``T`` explicitly.  Directional
    recoveries are

    ``R_A<-B = (M_A - M_A<-B) / T`` and
    ``R_B<-A = (M_B<-A - M_B) / T``;

    ``C`` is their arithmetic mean.  No absolute value or clipping is applied.
    """

    layer = int(layer)
    if hf_model.training:
        raise ValueError("Causal interchange requires eval mode")
    if not pair_id:
        raise ValueError("pair_id is required for causal provenance")
    if task_kind not in {"engine", "dashboard"}:
        raise ValueError("task_kind must be 'engine' or 'dashboard'")
    if not 0 <= layer < len(blocks):
        raise ValueError(f"Invalid interchange layer {layer!r}")
    if input_ids_a.ndim != 2 or input_ids_a.shape[0] != 1:
        raise ValueError("input_ids_a must have shape [1,S]")
    if input_ids_b.ndim != 2 or input_ids_b.shape[0] != 1:
        raise ValueError("input_ids_b must have shape [1,S]")
    if not math.isfinite(sharp_disagreement_threshold) or sharp_disagreement_threshold < 0:
        raise ValueError("Directional-disagreement threshold must be finite and nonnegative")

    state_a, clean_a = _validated_clean_tensors(
        clean_record_a,
        input_ids_a,
        side="A",
        layer=layer,
        position=position_a,
    )
    state_b, clean_b = _validated_clean_tensors(
        clean_record_b,
        input_ids_b,
        side="B",
        layer=layer,
        position=position_b,
    )
    metric_a = float(metric_fn(clean_a).cpu())
    metric_b = float(metric_fn(clean_b).cpu())
    if task_kind == "engine":
        if normalization_t is not None:
            raise ValueError("Engine T must be derived from its own two clean metrics")
        scale = metric_a - metric_b
        normalization_source = "engine_clean_metric_a_minus_clean_metric_b"
    else:
        if normalization_t is None:
            raise ValueError("Dashboard C requires the matched engine T")
        scale = float(normalization_t)
        normalization_source = "matched_engine_clean_T"
    if not math.isfinite(scale) or abs(scale) <= 1e-8:
        raise ValueError(f"Symmetric normalization T must be finite and nonzero: {scale}")

    logits_a_from_b = _forward_logits(
        hf_model,
        input_ids_a,
        blocks=blocks,
        edits={layer: full_residual_interchange_edit(state_b, position=position_a)},
    )
    logits_b_from_a = _forward_logits(
        hf_model,
        input_ids_b,
        blocks=blocks,
        edits={layer: full_residual_interchange_edit(state_a, position=position_b)},
    )
    metric_a_from_b = float(metric_fn(logits_a_from_b).cpu())
    metric_b_from_a = float(metric_fn(logits_b_from_a).cpu())
    r_a_from_b = (metric_a - metric_a_from_b) / scale
    r_b_from_a = (metric_b_from_a - metric_b) / scale
    causal_c = 0.5 * (r_a_from_b + r_b_from_a)
    disagreement = abs(r_a_from_b - r_b_from_a)
    finite_values = (
        metric_a,
        metric_b,
        metric_a_from_b,
        metric_b_from_a,
        scale,
        r_a_from_b,
        r_b_from_a,
        causal_c,
        disagreement,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("Symmetric interchange produced a non-finite scalar")

    return {
        "status": "OK",
        "pair_id": str(pair_id),
        "task_kind": task_kind,
        "variant": "full_residual",
        "eligible_as_primary_truth": True,
        "layer": layer,
        "position_rule": "explicit_concept_token_in_shared_context",
        "position_a": int(position_a),
        "position_b": int(position_b),
        "metric_a": metric_a,
        "metric_b": metric_b,
        "metric_a_from_b": metric_a_from_b,
        "metric_b_from_a": metric_b_from_a,
        "T": scale,
        "normalization_source": normalization_source,
        "R_a_from_b": r_a_from_b,
        "R_b_from_a": r_b_from_a,
        "C": causal_c,
        "directional_abs_difference": disagreement,
        "sharp_disagreement_threshold": float(sharp_disagreement_threshold),
        "sharp_directional_disagreement": disagreement > sharp_disagreement_threshold,
        "signed_unclipped": True,
        "clean_top_token_id_a": int(clean_a[0, -1].argmax().cpu()),
        "clean_top_token_id_b": int(clean_b[0, -1].argmax().cpu()),
        "edited_top_token_id_a_from_b": int(logits_a_from_b[0, -1].argmax().cpu()),
        "edited_top_token_id_b_from_a": int(logits_b_from_a[0, -1].argmax().cpu()),
    }


def _model_device(hf_model: nn.Module) -> torch.device:
    """Infer the live device from the model's first parameter."""

    try:
        return next(hf_model.parameters()).device
    except StopIteration as error:
        raise ValueError("The HF model has no parameters from which to infer a device") from error


def _encode_prompt(
    bundle: CausalModelBundle,
    prompt: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Encode one causal prompt as a single tensor row on ``device``."""

    encoded = bundle.tokenizer.encode(
        str(prompt),
        add_special_tokens=False,
        return_tensors="pt",
    )
    if not torch.is_tensor(encoded) or encoded.ndim != 2 or encoded.shape[0] != 1:
        raise TypeError("Tokenizer encode must return one tensor row for a prompt")
    return encoded.to(device)


def _require_fields(
    row: Mapping[str, Any],
    fields: Sequence[str],
    *,
    row_name: str,
) -> None:
    """Require a fixed field set in one externally supplied artifact row."""

    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{row_name} is missing required fields {missing}")


def _validate_expected_top_tokens(
    measurement: CausalMeasurement,
    *,
    expected_a: int,
    expected_b: int,
    label: str,
) -> None:
    """Fail if clean model predictions drift from the verification manifest."""

    if measurement["clean_top_token_id_a"] != int(expected_a):
        raise RuntimeError(f"{label} clean A top-token verification drifted")
    if measurement["clean_top_token_id_b"] != int(expected_b):
        raise RuntimeError(f"{label} clean B top-token verification drifted")


@torch.no_grad()
def compute_base_causal_rows(
    bundle: CausalModelBundle,
    verified_pairs: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    sharp_disagreement_threshold: float = 0.50,
) -> list[BaseCausalRow]:
    """Compute engine and original-dashboard truth in supplied pair order.

    The function does not filter or sort.  Every input must already carry the
    ``VERIFIED`` status, making exclusions an explicit dataset-stage decision.
    Each dashboard reuses only its matched engine's clean normalization ``T``.
    """

    if not verified_pairs:
        raise ValueError("At least one verified pair is required")
    if bundle.hf_model.training:
        raise ValueError("Base causal truth requires eval mode")
    blocks = bundle.lens_model.layers
    layer = int(layer)
    if not 0 <= layer < len(blocks):
        raise ValueError(f"Invalid causal layer {layer}")
    device = _model_device(bundle.hf_model)
    required = (
        "pair_id",
        "dependency_group",
        "fold",
        "category",
        "concept_a",
        "concept_b",
        "answer_a",
        "answer_b",
        "answer_a_token_id",
        "answer_b_token_id",
        "engine_prompt_a",
        "engine_prompt_b",
        "dashboard_prompt_a",
        "dashboard_prompt_b",
        "dashboard_token_id",
        "dashboard_distractor_token_id",
        "intervention_position_a",
        "intervention_position_b",
        "verification_status",
    )

    output: list[BaseCausalRow] = []
    seen_pair_ids: set[str] = set()
    for index, pair in enumerate(verified_pairs):
        row_name = f"verified_pairs[{index}]"
        _require_fields(pair, required, row_name=row_name)
        pair_id = str(pair["pair_id"])
        if not pair_id or pair_id in seen_pair_ids:
            raise ValueError(f"{row_name} has an empty or duplicate pair_id {pair_id!r}")
        seen_pair_ids.add(pair_id)
        if pair["verification_status"] != "VERIFIED":
            raise ValueError(f"{pair_id} is not VERIFIED; causal builders never relabel rows")

        position_a = int(pair["intervention_position_a"])
        position_b = int(pair["intervention_position_b"])
        engine_ids_a = _encode_prompt(bundle, str(pair["engine_prompt_a"]), device=device)
        engine_ids_b = _encode_prompt(bundle, str(pair["engine_prompt_b"]), device=device)
        engine_clean_a = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            engine_ids_a,
            layer,
            position=position_a,
        )
        engine_clean_b = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            engine_ids_b,
            layer,
            position=position_b,
        )
        engine = symmetric_interchange(
            bundle.hf_model,
            blocks,
            engine_ids_a,
            engine_ids_b,
            engine_clean_a,
            engine_clean_b,
            token_difference_metric(
                int(pair["answer_a_token_id"]),
                int(pair["answer_b_token_id"]),
            ),
            pair_id=pair_id,
            task_kind="engine",
            layer=layer,
            position_a=position_a,
            position_b=position_b,
            sharp_disagreement_threshold=sharp_disagreement_threshold,
        )
        _validate_expected_top_tokens(
            engine,
            expected_a=int(pair["answer_a_token_id"]),
            expected_b=int(pair["answer_b_token_id"]),
            label=f"{pair_id} engine",
        )

        dashboard_ids_a = _encode_prompt(
            bundle, str(pair["dashboard_prompt_a"]), device=device
        )
        dashboard_ids_b = _encode_prompt(
            bundle, str(pair["dashboard_prompt_b"]), device=device
        )
        dashboard_clean_a = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            dashboard_ids_a,
            layer,
            position=position_a,
        )
        dashboard_clean_b = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            dashboard_ids_b,
            layer,
            position=position_b,
        )
        dashboard = symmetric_interchange(
            bundle.hf_model,
            blocks,
            dashboard_ids_a,
            dashboard_ids_b,
            dashboard_clean_a,
            dashboard_clean_b,
            token_difference_metric(
                int(pair["dashboard_token_id"]),
                int(pair["dashboard_distractor_token_id"]),
            ),
            pair_id=pair_id,
            task_kind="dashboard",
            layer=layer,
            position_a=position_a,
            position_b=position_b,
            normalization_t=engine["T"],
            sharp_disagreement_threshold=sharp_disagreement_threshold,
        )
        _validate_expected_top_tokens(
            dashboard,
            expected_a=int(pair["dashboard_token_id"]),
            expected_b=int(pair["dashboard_token_id"]),
            label=f"{pair_id} dashboard",
        )

        output.append(
            {
                "pair_id": pair_id,
                "dependency_group": str(pair["dependency_group"]),
                "fold": int(pair["fold"]),
                "category": str(pair["category"]),
                "concept_a": str(pair["concept_a"]),
                "concept_b": str(pair["concept_b"]),
                "answer_a": str(pair["answer_a"]),
                "answer_b": str(pair["answer_b"]),
                "engine": {"full_residual": engine},
                "dashboard": {"full_residual": dashboard},
            }
        )
    return output


def _base_engine_measurement(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract and validate one base engine's signed full-residual truth."""

    engine = row.get("engine")
    if not isinstance(engine, Mapping):
        raise ValueError(f"Base row {row.get('pair_id')!r} has no engine result")
    measurement = engine.get("full_residual")
    if not isinstance(measurement, Mapping):
        raise ValueError(f"Base row {row.get('pair_id')!r} has no full-residual engine")
    if measurement.get("variant") != "full_residual" or not measurement.get(
        "signed_unclipped"
    ):
        raise ValueError(f"Base row {row.get('pair_id')!r} is not signed full-residual truth")
    return measurement


@torch.no_grad()
def compute_hard_dashboard_causal_rows(
    bundle: CausalModelBundle,
    verified_hard_rows: Sequence[Mapping[str, Any]],
    base_causal_rows: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    sharp_disagreement_threshold: float = 0.50,
) -> list[HardDashboardCausalRow]:
    """Compute answer-type-matched dashboard truth in supplied row order.

    The matched engine is not recomputed.  Its signed ``C`` is copied for the
    joined output and its clean ``T`` is the sole hard-dashboard normalization.
    Every hard row must be ``VERIFIED_HARD`` and must match exactly one base
    engine pair.
    """

    if not verified_hard_rows:
        raise ValueError("At least one VERIFIED_HARD row is required")
    if bundle.hf_model.training:
        raise ValueError("Hard-dashboard causal truth requires eval mode")
    blocks = bundle.lens_model.layers
    layer = int(layer)
    if not 0 <= layer < len(blocks):
        raise ValueError(f"Invalid causal layer {layer}")
    device = _model_device(bundle.hf_model)

    base_by_pair: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(base_causal_rows):
        pair_id = str(row.get("pair_id", ""))
        if not pair_id or pair_id in base_by_pair:
            raise ValueError(f"base_causal_rows[{index}] has an empty or duplicate pair_id")
        _base_engine_measurement(row)
        base_by_pair[pair_id] = row

    required = (
        "pair_id",
        "dependency_group",
        "fold",
        "category",
        "hard_template_id",
        "hard_prompt_a",
        "hard_prompt_b",
        "hard_target_token_id",
        "hard_distractor_token_id",
        "intervention_position_a",
        "intervention_position_b",
        "verification_status",
    )
    output: list[HardDashboardCausalRow] = []
    seen_pair_ids: set[str] = set()
    for index, hard_row in enumerate(verified_hard_rows):
        row_name = f"verified_hard_rows[{index}]"
        _require_fields(hard_row, required, row_name=row_name)
        pair_id = str(hard_row["pair_id"])
        if not pair_id or pair_id in seen_pair_ids:
            raise ValueError(f"{row_name} has an empty or duplicate pair_id {pair_id!r}")
        seen_pair_ids.add(pair_id)
        if hard_row["verification_status"] != "VERIFIED_HARD":
            raise ValueError(f"{pair_id} is not VERIFIED_HARD; causal builders never relabel rows")
        try:
            base_row = base_by_pair[pair_id]
        except KeyError as error:
            raise ValueError(f"Hard dashboard {pair_id!r} has no matched base engine") from error
        engine = _base_engine_measurement(base_row)
        engine_t = float(engine["T"])
        engine_c = float(engine["C"])
        if not math.isfinite(engine_t) or not math.isfinite(engine_c):
            raise ValueError(f"Matched engine {pair_id!r} has non-finite truth")

        position_a = int(hard_row["intervention_position_a"])
        position_b = int(hard_row["intervention_position_b"])
        input_ids_a = _encode_prompt(bundle, str(hard_row["hard_prompt_a"]), device=device)
        input_ids_b = _encode_prompt(bundle, str(hard_row["hard_prompt_b"]), device=device)
        clean_a = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            input_ids_a,
            layer,
            position=position_a,
        )
        clean_b = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            input_ids_b,
            layer,
            position=position_b,
        )
        hard_dashboard = symmetric_interchange(
            bundle.hf_model,
            blocks,
            input_ids_a,
            input_ids_b,
            clean_a,
            clean_b,
            token_difference_metric(
                int(hard_row["hard_target_token_id"]),
                int(hard_row["hard_distractor_token_id"]),
            ),
            pair_id=pair_id,
            task_kind="dashboard",
            layer=layer,
            position_a=position_a,
            position_b=position_b,
            normalization_t=engine_t,
            sharp_disagreement_threshold=sharp_disagreement_threshold,
        )
        _validate_expected_top_tokens(
            hard_dashboard,
            expected_a=int(hard_row["hard_target_token_id"]),
            expected_b=int(hard_row["hard_target_token_id"]),
            label=f"{pair_id} hard dashboard",
        )
        output.append(
            {
                "pair_id": pair_id,
                "dependency_group": str(hard_row["dependency_group"]),
                "fold": int(hard_row["fold"]),
                "category": str(hard_row["category"]),
                "hard_template_id": str(hard_row["hard_template_id"]),
                "frozen_engine_C": engine_c,
                "frozen_engine_T": engine_t,
                "hard_dashboard": hard_dashboard,
            }
        )
    return output


def _finite_median(values: Sequence[float], *, name: str) -> float:
    """Return a median after enforcing nonempty finite input."""

    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a nonempty sequence of finite values")
    return float(median(values))


def _task_measurement(
    row: Mapping[str, Any],
    task: Literal["engine", "dashboard"],
) -> Mapping[str, Any]:
    """Extract and validate one engine or dashboard causal measurement."""

    container = row.get(task)
    if not isinstance(container, Mapping):
        raise ValueError(f"Base row {row.get('pair_id')!r} has no {task} result")
    measurement = container.get("full_residual")
    if not isinstance(measurement, Mapping):
        raise ValueError(f"Base row {row.get('pair_id')!r} has no full-residual {task}")
    if measurement.get("variant") != "full_residual" or not measurement.get(
        "signed_unclipped"
    ):
        raise ValueError(f"Base row {row.get('pair_id')!r} has invalid {task} truth")
    return measurement


def summarize_base_causal_sanity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Summarize engine/old-dashboard causal separation without clipping signs."""

    if not rows:
        raise ValueError("At least one base causal row is required")
    engine = [_task_measurement(row, "engine") for row in rows]
    dashboard = [_task_measurement(row, "dashboard") for row in rows]
    engine_c = [float(measurement["C"]) for measurement in engine]
    dashboard_c = [float(measurement["C"]) for measurement in dashboard]
    return {
        "n_pairs": len(rows),
        "engine_C_median": _finite_median(engine_c, name="engine C"),
        "engine_abs_C_median": _finite_median(
            [abs(value) for value in engine_c], name="absolute engine C"
        ),
        "dashboard_C_median": _finite_median(dashboard_c, name="dashboard C"),
        "dashboard_abs_C_median": _finite_median(
            [abs(value) for value in dashboard_c], name="absolute dashboard C"
        ),
        "engine_sharp_directional_disagreement": sum(
            bool(measurement["sharp_directional_disagreement"])
            for measurement in engine
        ),
        "dashboard_sharp_directional_disagreement": sum(
            bool(measurement["sharp_directional_disagreement"])
            for measurement in dashboard
        ),
    }


def summarize_hard_dashboard_causal_sanity(
    rows: Sequence[Mapping[str, Any]],
    *,
    engine_large_threshold: float = 0.50,
    dashboard_near_zero_threshold: float = 0.10,
) -> dict[str, int | float | bool | str]:
    """Summarize frozen-engine and hard-dashboard truth with fixed sanity gates."""

    if not rows:
        raise ValueError("At least one hard-dashboard causal row is required")
    if not math.isfinite(engine_large_threshold) or engine_large_threshold < 0:
        raise ValueError("engine_large_threshold must be finite and nonnegative")
    if not math.isfinite(dashboard_near_zero_threshold) or dashboard_near_zero_threshold < 0:
        raise ValueError("dashboard_near_zero_threshold must be finite and nonnegative")

    engine_c: list[float] = []
    hard_c: list[float] = []
    disagreements = 0
    for row in rows:
        engine_c.append(float(row["frozen_engine_C"]))
        measurement = row.get("hard_dashboard")
        if not isinstance(measurement, Mapping):
            raise ValueError(f"Hard row {row.get('pair_id')!r} has no causal result")
        if measurement.get("variant") != "full_residual" or not measurement.get(
            "signed_unclipped"
        ):
            raise ValueError(f"Hard row {row.get('pair_id')!r} has invalid causal truth")
        hard_c.append(float(measurement["C"]))
        disagreements += bool(measurement["sharp_directional_disagreement"])

    engine_abs_median = _finite_median(
        [abs(value) for value in engine_c], name="absolute frozen engine C"
    )
    hard_abs_median = _finite_median(
        [abs(value) for value in hard_c], name="absolute hard-dashboard C"
    )
    engine_large = engine_abs_median > float(engine_large_threshold)
    hard_near_zero = hard_abs_median < float(dashboard_near_zero_threshold)
    return {
        "n_pairs": len(rows),
        "engine_C_median": _finite_median(engine_c, name="frozen engine C"),
        "engine_abs_C_median": engine_abs_median,
        "hard_dashboard_C_median": _finite_median(hard_c, name="hard-dashboard C"),
        "hard_dashboard_abs_C_median": hard_abs_median,
        "hard_dashboard_C_min": min(hard_c),
        "hard_dashboard_C_max": max(hard_c),
        "hard_dashboard_sharp_directional_disagreements": int(disagreements),
        "engine_large_gate": engine_large,
        "hard_dashboard_near_zero_gate": hard_near_zero,
        "status": "PASS" if engine_large and hard_near_zero else "FAIL",
    }

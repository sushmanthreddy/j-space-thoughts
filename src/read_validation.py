"""One-shot READ validation utilities for the Qwen2.5-7B Go/No-Go run.

This module intentionally implements only the frozen validation protocol:

* coordinate-resampling and masked source-to-foil causal measurements;
* the legacy global weight READ baseline (R1);
* train-fold top-k path-restricted weight READ profiles (R2);
* exact, real-activation local derivative profiles (R3); and
* model-free cross-fit calibration, AUC, bootstrap, and correlation summaries.

It does not choose interventions from outcomes, create new concept directions,
run hypothesis science, or provide alternate estimator definitions.  All
functions return JSON-ready metadata or tensors that are reduced before being
persisted by the notebooks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.interventions import clamped_swap_edits, forward_logits, residual_edit_hooks
from src.localization_phase import (
    capture_qwen_components,
    component_grad_delta_scores,
)
from src.read_scores import qwen_mlp_gain


SEED = 1729
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
WORKSPACE_LAYERS = tuple(range(13, 25))
COMPONENT_FRONTIER = (25, 26, 27)
N_FOLDS = 4
PATH_MLPS = 2
PATH_HEADS = 6
QWEN25_7B_HEADS_PER_LAYER = 28
RELATIVE_POSITION_GRID_SIZE = 101
SURGICAL_ALPHA = 1.5
N_BOOTSTRAP = 10_000

CANDIDATE_NAMES = (
    "R1",
    "R2_sum",
    "R2_peak",
    "R2_carrying",
    "R3_sum",
    "R3_peak",
    "R3_carrying",
)

READ_VALIDATION_PROTOCOL: dict[str, Any] = {
    "schema_version": "read-go-no-go-v1",
    "scope": "READ validation only; Written-vs-Read P1/P2/P3 not tested",
    "model": {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "torch.bfloat16",
        "scale_arm": "SKIPPED_NO_COMPARABLE_QWEN3_JLENS",
    },
    "seed": SEED,
    "workspace_layers": list(WORKSPACE_LAYERS),
    "component_frontier": list(COMPONENT_FRONTIER),
    "component_frontier_scope": (
        "compute-bounded common L25-27 frontier; not a claim to cover all "
        "downstream components"
    ),
    "labels": {
        "engine": 1,
        "dashboard": 0,
        "source": "fixed provenance labels; no outcome filtering",
        "engines": "all 155 strict raw02 jlens_raw_wu_j primary rows",
        "dashboards": (
            "all eight pinned known-narration rows; seven clean-capable and "
            "es2 retained as an explicit baseline-incapable failure"
        ),
    },
    "engine_donor_rule": (
        "different source concept and different clean target required; then "
        "prefer same category, exact token length, intended swap_to, sufficient "
        "for the frozen rank<=10 carrying positions or nearest token length, "
        "and lexical order"
    ),
    "folds": {
        "n": N_FOLDS,
        "engine_group": (
            "connected component of the undirected source/foil graph; the "
            "unordered pair ID is retained separately"
        ),
        "dashboard_group": "language",
        "assignment": "deterministic label-stratified greedy balance",
    },
    "donors": {
        "role": "predeclared exogenous corruption bank",
        "selection_timing": (
            "after freezing label-blind carrying-mask geometry, before causal "
            "outcomes, READ scores, and fold assignment"
        ),
        "fold_use": "donor fold is never used for estimator/path fitting",
        "audit": "same-fold, cross-fold, and unassigned donor links are reported",
    },
    "ground_truth": {
        "A_primary": {
            "name": "coordinate resampling",
            "operation": (
                "at every frozen rank<=10 position and L13-24, replace the "
                "recipient coefficient along v_c with the same-layer, "
                "same-position clean donor coefficient"
            ),
            "independence": (
                "independent of READ weights and READ gradients, but explicitly "
                "dependent on the frozen J-Lens concept direction"
            ),
        },
        "B_secondary": {
            "name": "masked source-to-foil clamped swap",
            "operation": "v3 carrying mask, L13-24, alpha=1.5",
            "alpha": SURGICAL_ALPHA,
        },
    },
    "path_localization": (
        "grad(M_clean, L25-27 component) dot "
        "(component_A_edited - component_clean), with position records"
    ),
    "path_selection": {
        "fit": "training folds only; labels are never read",
        "aggregation": (
            "normalize absolute component scores within prompt, average within "
            "concept first, then equally across training concepts"
        ),
        "R1": (
            "global legacy repaired-weight formula over all common L25-27 "
            "components (three MLPs and every attention head); no path selection"
        ),
        "R2_R3": {"mlps": PATH_MLPS, "heads": PATH_HEADS},
        "tie_break": "component ID lexical order",
    },
    "read_candidates": list(CANDIDATE_NAMES),
    "profiles": {
        "R2": (
            "static repaired weight READ; peak/carrying allocate each selected "
            "component with a training-fold relative-position profile interpolated "
            "to held-out length; R2_sum is the static equal-family composite"
        ),
        "R3": (
            "exact reverse-autograd local component-output derivative at the "
            "real activation, mean absolute derivative over causally reachable q"
        ),
        "summaries": ["sum", "peak", "carrying"],
        "summary_definitions": {
            "R2_sum": "static repaired-weight equal-family composite",
            "R3_sum": (
                "mean over positions after summing the selected component profiles"
            ),
            "peak": "maximum over all layer-position cells",
            "carrying": "mean over the layer-by-carrying-position cells",
        },
    },
    "label_verification": {
        "retention": "all rows, including failures, remain in every report subset",
        "engine_B": "edited top token must equal the declared counterfactual target",
        "dashboard_B": "absolute metric delta must be <=0.5",
        "clean_capability": "reported separately as a diagnostic; not a GO gate",
        "gate": "incomplete declared-label coverage cannot trigger GO",
    },
    "cross_fit": (
        "support scores under all four train-fold paths; assigned-fold score "
        "calibrated by the training-concept empirical CDF"
    ),
    "decision": {
        "auc_min": 0.70,
        "ci_low_strictly_above": 0.5,
        "bootstrap_draws": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "undefined_policy": (
            "preserve undefined; a candidate cannot pass without complete "
            "concept coverage"
        ),
        "auc_ground_truth_note": (
            "fixed-label AUC is identical under A and B; duplicated rows are "
            "reported only to identify the requested ground-truth comparison"
        ),
        "inferential_look_count": 1,
    },
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def protocol_sha256(
    protocol: Mapping[str, Any] = READ_VALIDATION_PROTOCOL,
) -> str:
    """Return the canonical SHA-256 for a frozen protocol mapping."""

    return hashlib.sha256(_canonical_json_bytes(protocol)).hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert nested scientific values to strict JSON-compatible objects."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


READ_VALIDATION_PROTOCOL_SHA256 = protocol_sha256()


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object and reject other top-level shapes."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {source}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write deterministic JSON and return its path, size, and digest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    raw = target.read_bytes()
    return {
        "path": str(target),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stable_int(text: str, *, seed: int = SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _canonical_label(value: Any) -> str:
    text = " ".join(str(value).split()).casefold()
    if not text:
        raise ValueError("Concept/language labels must be nonempty")
    return text


def _engine_group(source: str, foil: str) -> str:
    pair = sorted((_canonical_label(source), _canonical_label(foil)))
    return "engine-pair:" + "::".join(pair)


def _engine_connected_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Map every engine concept node to its undirected graph component ID."""

    graph: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = _canonical_label(row["source_concept"])
        foil = _canonical_label(row["foil_concept"])
        graph[source].add(foil)
        graph[foil].add(source)
    output: dict[str, str] = {}
    remaining = set(graph)
    while remaining:
        root = min(remaining)
        stack = [root]
        members: set[str] = set()
        while stack:
            node = stack.pop()
            if node in members:
                continue
            members.add(node)
            stack.extend(sorted(graph[node] - members, reverse=True))
        remaining -= members
        component_id = "engine-connected:" + "::".join(sorted(members))
        for member in members:
            output[member] = component_id
    return output


def _engine_concept(source: str) -> str:
    return "engine-concept:" + _canonical_label(source)


def _dashboard_group(language: str) -> str:
    return "dashboard-language:" + _canonical_label(language)


def strict_engine_roster(
    raw02: Mapping[str, Any],
    *,
    carrying_positions_by_row: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Build the full strict engine roster and hierarchical exogenous donor map.

    Rows are never filtered by intervention outcomes or by donor geometry.  A
    too-short frozen donor is recorded as an undefined A corruption for that row
    rather than causing a pre-analysis exclusion.
    """

    raw_rows = raw02.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("raw02 payload is missing rows")
    primary = [
        dict(row)
        for row in raw_rows
        if row.get("direction_method") == "jlens_raw_wu_j"
        and row.get("measurement_status") == "OK"
        and row.get("clean_eligibility", {}).get("eligible") is True
    ]
    if not primary:
        raise ValueError("No strict raw02 primary engine rows")

    retained: list[dict[str, Any]] = []
    donor_failures: list[dict[str, Any]] = []
    for row in primary:
        source = str(row["intermediate"])
        foil = str(row["swap_to"])
        clean_target = int(row["token_ids"]["target"])
        if carrying_positions_by_row is not None:
            supplied_positions = carrying_positions_by_row.get(str(row["name"]))
            if supplied_positions is None:
                raise ValueError(
                    f"Missing frozen carrying positions for engine row {row['name']!r}"
                )
            recipient_positions = [int(value) for value in supplied_positions]
            required_position_source = "supplied frozen rank<=10 carrying mask"
        else:
            recipient_positions = [
                int(value)
                for value in row.get(
                    "intervention_positions", range(int(row["n_prompt_tokens"]))
                )
            ]
            required_position_source = "raw intervention-position upper bound"
        if not recipient_positions:
            raise ValueError(f"Engine row {row['name']!r} has no recipient positions")
        max_required_position = max(recipient_positions)
        candidates = [
            donor
            for donor in primary
            if donor["name"] != row["name"]
            and _canonical_label(donor["intermediate"])
            != _canonical_label(source)
            and int(donor["token_ids"]["target"]) != clean_target
        ]
        candidates.sort(
            key=lambda donor: (
                str(donor["category"]) != str(row["category"]),
                int(donor["n_prompt_tokens"]) != int(row["n_prompt_tokens"]),
                _canonical_label(donor["intermediate"])
                != _canonical_label(foil),
                int(donor["n_prompt_tokens"]) <= max_required_position,
                abs(
                    int(donor["n_prompt_tokens"])
                    - int(row["n_prompt_tokens"])
                ),
                _canonical_label(donor["intermediate"]),
                str(donor["name"]),
            )
        )
        if not candidates:
            donor_failures.append(
                {
                    "row_id": str(row["name"]),
                    "reason": "NO_DIFFERENT_SOURCE_AND_TARGET_DONOR",
                    "category": row["category"],
                    "source_concept": source,
                    "foil_concept": foil,
                    "n_prompt_tokens": int(row["n_prompt_tokens"]),
                }
            )
        donor = candidates[0] if candidates else None
        donor_length = int(donor["n_prompt_tokens"]) if donor is not None else None
        donor_covers = bool(
            donor_length is not None and donor_length > max_required_position
        )
        retained.append(
            {
                "row_id": str(row["name"]),
                "label": 1,
                "class_name": "engine",
                "clean_capable": True,
                "concept_id": _engine_concept(source),
                "fold_group": None,
                "pair_id": _engine_group(source, foil),
                "source_concept": source,
                "foil_concept": foil,
                "category": str(row["category"]),
                "prompt": str(row["prompt"]),
                "n_prompt_tokens": int(row["n_prompt_tokens"]),
                "source_token_id": int(row["token_ids"]["concept"]),
                "foil_concept_token_id": int(row["token_ids"]["foil_concept"]),
                "clean_target_token_id": clean_target,
                "counterfactual_target_token_id": int(row["token_ids"]["foil"]),
                "donor_row_id": str(donor["name"]) if donor is not None else None,
                "donor_source_concept": (
                    str(donor["intermediate"]) if donor is not None else None
                ),
                "donor_clean_target_token_id": (
                    int(donor["token_ids"]["target"])
                    if donor is not None
                    else None
                ),
                "donor_n_prompt_tokens": donor_length,
                "donor_same_category": bool(
                    donor is not None and donor["category"] == row["category"]
                ),
                "donor_exact_token_length": bool(
                    donor_length == int(row["n_prompt_tokens"])
                ),
                "donor_preferred_intended_foil": bool(
                    donor is not None
                    and _canonical_label(donor["intermediate"])
                    == _canonical_label(foil)
                ),
                "donor_max_required_position": max_required_position,
                "donor_required_position_source": required_position_source,
                "donor_covers_recipient_positions": donor_covers,
                "coordinate_resampling_availability": (
                    "AVAILABLE" if donor_covers else "UNDEFINED_DONOR_TOO_SHORT_OR_MISSING"
                ),
                "provenance": {
                    "source": "data/raw/02_twohop_qwen2.5-7b.json",
                    "direction_method": "jlens_raw_wu_j",
                    "label_fixed_before_current_outcomes": True,
                    "donor_rule": READ_VALIDATION_PROTOCOL["engine_donor_rule"],
                    "donor_bank_role": "predeclared exogenous corruption",
                },
            }
        )
    connected = _engine_connected_groups(retained)
    for row in retained:
        row["fold_group"] = connected[_canonical_label(row["source_concept"])]
        row["bootstrap_pair_id"] = row["pair_id"]
    return {
        "status": "OK" if retained else "EMPTY",
        "rows": retained,
        "unavailable": [],
        "donor_failures": donor_failures,
        "n_primary": len(primary),
        "n_retained": len(retained),
        "n_unavailable": 0,
        "n_coordinate_resampling_undefined_donor_geometry": sum(
            not row["donor_covers_recipient_positions"] for row in retained
        ),
        "expected_frozen_primary_count": 155,
        "all_strict_primary_rows_retained": len(retained) == len(primary),
        "canonical_examples_retained": {
            name: any(
                _canonical_label(row["source_concept"]) == name for row in retained
            )
            for name in ("spider", "buffalo")
        },
        "strict_primary_roster_coverage_complete": len(retained) == len(primary),
        "canonical_domain_coverage_claimed": False,
        "coverage_note": (
            "All strict primary rows are retained; donor insufficiency is an "
            "explicit undefined A measurement, never a roster exclusion."
        ),
    }


def clean_dashboard_roster(
    narration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return all eight fixed dashboard rows, including the es2 failure.

    The caller supplies prepared narration rows containing prompts, token IDs,
    masks, and explicit clean capability.  Donors may be predeclared with
    ``donor_row_id``.  Otherwise a deterministic different-language donor with
    clean capability and sufficient sequence length is selected without looking
    at any outcomes.  Baseline-incapable es2 is retained rather than filtered.
    """

    prepared = [dict(row) for row in narration_rows]
    if len(prepared) != 8:
        raise ValueError(f"Expected all eight pinned dashboards, got {len(prepared)}")
    by_id = {str(row.get("row_id", row.get("key"))): row for row in prepared}
    if len(by_id) != len(prepared):
        raise ValueError("Dashboard row IDs must be unique")
    clean_ids = sorted(
        row_id for row_id, row in by_id.items() if row.get("clean_capable") is True
    )
    incapable_ids = sorted(set(by_id) - set(clean_ids))
    if len(clean_ids) != 7 or len(incapable_ids) != 1:
        raise ValueError(
            "Pinned dashboard roster must contain seven clean-capable rows and "
            "one explicit baseline-incapable row"
        )
    if _canonical_label(incapable_ids[0]) != "es2":
        raise ValueError(
            f"The explicit baseline-incapable dashboard must be es2, got {incapable_ids[0]!r}"
        )

    output: list[dict[str, Any]] = []
    for row_id, row in sorted(by_id.items()):
        language = str(row.get("language", row.get("category")))
        sequence_length = int(row["sequence_length"])
        carrying = sorted(set(int(value) for value in row["carrying_positions"]))
        if not carrying or carrying[-1] >= sequence_length:
            raise ValueError(f"Invalid dashboard carrying positions for {row_id}")
        declared = row.get("donor_row_id")
        if declared is not None:
            donor_id = str(declared)
            if donor_id not in by_id:
                raise ValueError(f"Unknown dashboard donor {donor_id!r}")
            if by_id[donor_id].get("clean_capable") is not True:
                raise ValueError(f"Dashboard donor {donor_id!r} is not clean-capable")
            if donor_id == row_id or _canonical_label(
                by_id[donor_id].get("language", by_id[donor_id].get("category"))
            ) == _canonical_label(language):
                raise ValueError(
                    f"Dashboard donor {donor_id!r} must use a different language"
                )
            if int(by_id[donor_id]["sequence_length"]) <= carrying[-1]:
                raise ValueError(f"Dashboard donor {donor_id!r} is too short")
            donors = [by_id[donor_id]]
        else:
            donors = [
                candidate
                for candidate_id, candidate in by_id.items()
                if candidate_id != row_id
                and candidate.get("clean_capable") is True
                and _canonical_label(candidate.get("language", candidate.get("category")))
                != _canonical_label(language)
                and int(candidate["sequence_length"]) > carrying[-1]
            ]
            donors.sort(
                key=lambda candidate: (
                    int(candidate["sequence_length"]) != sequence_length,
                    abs(int(candidate["sequence_length"]) - sequence_length),
                    _canonical_label(
                        candidate.get("language", candidate.get("category"))
                    ),
                    str(candidate.get("row_id", candidate.get("key"))),
                )
            )
        if not donors:
            raise ValueError(f"No valid different-language dashboard donor for {row_id}")
        donor = donors[0]
        donor_id = str(donor.get("row_id", donor.get("key")))
        output.append(
            {
                **row,
                "row_id": row_id,
                "label": 0,
                "class_name": "dashboard",
                "concept_id": _dashboard_group(language),
                "fold_group": _dashboard_group(language),
                "language": language,
                "clean_capable": bool(row.get("clean_capable") is True),
                "carrying_positions": carrying,
                "donor_row_id": donor_id,
                "declared_label_baseline_status": (
                    "CAPABLE" if row.get("clean_capable") is True else "FAIL_ES2"
                ),
                "provenance": {
                    "source": "pinned known-narration passages",
                    "label_fixed_before_current_outcomes": True,
                    "clean_capability_reported_not_filtered": True,
                    "donor_bank_role": "predeclared exogenous corruption",
                    "donor_rule": (
                        "different language; sufficient same positions; exact "
                        "sequence length preferred, then nearest length and lexical"
                    ),
                },
            }
        )
    return {
        "status": "OK_WITH_RETAINED_BASELINE_FAILURE",
        "rows": output,
        "n": len(output),
        "n_clean_capable": len(clean_ids),
        "baseline_incapable_row_ids": incapable_ids,
        "all_rows_retained": True,
    }


def assign_balanced_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Assign deterministic balanced folds at the frozen class-specific groups."""

    if n_folds != 4:
        raise ValueError("The frozen protocol requires exactly four folds")
    row_list = [dict(row) for row in rows]
    if not row_list:
        raise ValueError("Fold assignment requires rows")
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for row in row_list:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("Validation labels must be zero or one")
        grouped[label][str(row["fold_group"])].append(row)
    if len(grouped[0]) != n_folds:
        raise ValueError(
            "Four dashboard-language groups are required so each fold has a negative"
        )

    assignment: dict[str, int] = {}
    counts = {label: [0] * n_folds for label in (0, 1)}
    group_counts = {label: [0] * n_folds for label in (0, 1)}
    for label in (0, 1):
        groups = list(grouped[label].items())
        groups.sort(
            key=lambda item: (
                -len(item[1]),
                _stable_int(item[0], seed=seed),
                item[0],
            )
        )
        for group, members in groups:
            fold = min(
                range(n_folds),
                key=lambda index: (
                    counts[label][index],
                    group_counts[label][index],
                    index,
                ),
            )
            assignment[group] = fold
            counts[label][fold] += len(members)
            group_counts[label][fold] += 1

    manifested: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in row_list:
        row_id = str(row["row_id"])
        if row_id in seen_ids:
            raise ValueError(f"Duplicate validation row ID {row_id!r}")
        seen_ids.add(row_id)
        manifested.append({**row, "fold": assignment[str(row["fold_group"])]})
    for fold in range(n_folds):
        labels = {int(row["label"]) for row in manifested if row["fold"] == fold}
        if labels != {0, 1}:
            raise RuntimeError(f"Fold {fold} does not contain both fixed labels")
    fold_by_row_id = {str(row["row_id"]): int(row["fold"]) for row in manifested}
    donor_links: list[dict[str, Any]] = []
    for row in manifested:
        donor_id = row.get("donor_row_id")
        if donor_id is None:
            continue
        recipient_fold = int(row["fold"])
        donor_fold = fold_by_row_id.get(str(donor_id))
        if donor_fold is None:
            relation = "UNASSIGNED_EXOGENOUS_BANK_ROW"
        elif donor_fold == recipient_fold:
            relation = "SAME_FOLD"
        else:
            relation = "CROSS_FOLD"
        donor_links.append(
            {
                "row_id": str(row["row_id"]),
                "donor_row_id": str(donor_id),
                "recipient_fold": recipient_fold,
                "donor_fold": donor_fold,
                "relation": relation,
            }
        )
    donor_relation_counts = {
        relation: sum(link["relation"] == relation for link in donor_links)
        for relation in (
            "SAME_FOLD",
            "CROSS_FOLD",
            "UNASSIGNED_EXOGENOUS_BANK_ROW",
        )
    }
    return {
        "schema_version": "read-validation-folds-v1",
        "seed": seed,
        "n_folds": n_folds,
        "group_assignment": assignment,
        "row_counts_by_label_fold": {
            str(label): counts[label] for label in (0, 1)
        },
        "group_counts_by_label_fold": {
            str(label): group_counts[label] for label in (0, 1)
        },
        "donor_fold_audit": {
            "donor_bank_role": "predeclared exogenous corruption",
            "donor_fold_used_for_estimator_fitting": False,
            "links": donor_links,
            "relation_counts": donor_relation_counts,
        },
        "rows": manifested,
    }


def _positions_by_layer(
    positions: Sequence[int] | Mapping[int | str, Sequence[int]],
    layers: Sequence[int],
) -> dict[int, list[int]]:
    if isinstance(positions, Mapping):
        output = {
            int(layer): sorted(
                set(int(value) for value in positions.get(layer, positions.get(str(layer), [])))
            )
            for layer in layers
        }
    else:
        shared = sorted(set(int(value) for value in positions))
        output = {int(layer): list(shared) for layer in layers}
    if any(not values for values in output.values()):
        raise ValueError("Every resampled layer requires at least one carrying position")
    return output


def coordinate_resampling_edits(
    recipient_clean_residuals: Mapping[int, torch.Tensor],
    donor_clean_residuals: Mapping[int, torch.Tensor],
    directions: Mapping[int, torch.Tensor],
    positions: Sequence[int] | Mapping[int | str, Sequence[int]],
) -> tuple[dict[int, Callable[[torch.Tensor], torch.Tensor]], dict[str, Any]]:
    """Clamp recipient concept coefficients to clean donor coefficients.

    Only the scalar coefficient along the frozen recipient direction is copied.
    Orthogonal residual content is retained.  This makes the operation
    independent of READ weights/gradients but not independent of the direction.
    """

    layers = sorted(int(layer) for layer in directions)
    if layers != list(WORKSPACE_LAYERS):
        raise ValueError(f"Coordinate resampling requires layers {WORKSPACE_LAYERS}")
    if set(recipient_clean_residuals) != set(layers) or set(
        donor_clean_residuals
    ) != set(layers):
        raise ValueError("Recipient/donor residuals must cover all workspace layers")
    selected = _positions_by_layer(positions, layers)
    edits: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    coefficient_manifest: dict[str, Any] = {}
    for layer in layers:
        recipient = recipient_clean_residuals[layer]
        donor = donor_clean_residuals[layer]
        if recipient.ndim != 3 or donor.ndim != 3:
            raise ValueError("Clean residuals must have shape [B,S,D]")
        if recipient.shape[0] != 1 or donor.shape[0] != 1:
            raise ValueError("Coordinate resampling requires unpadded batch size one")
        indices = selected[layer]
        if min(indices) < 0 or max(indices) >= recipient.shape[1]:
            raise IndexError(f"Recipient position outside L{layer} sequence")
        if max(indices) >= donor.shape[1]:
            raise IndexError(f"Donor lacks a same-position coefficient at L{layer}")
        vector = directions[layer].detach().float()
        if not torch.isfinite(vector).all() or not torch.isclose(
            vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4
        ):
            raise ValueError(f"Coordinate-resampling direction L{layer} is not unit")
        donor_values = torch.einsum(
            "bsd,d->bs",
            donor[:, indices].float(),
            vector.to(donor.device),
        ).detach()
        recipient_values = torch.einsum(
            "bsd,d->bs",
            recipient[:, indices].float(),
            vector.to(recipient.device),
        ).detach()

        def edit(
            hidden: torch.Tensor,
            *,
            _indices=tuple(indices),
            _vector=vector,
            _donor_values=donor_values,
        ) -> torch.Tensor:
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("Coordinate-resampling hook requires [1,S,D]")
            if max(_indices) >= hidden.shape[1]:
                raise IndexError("Coordinate-resampling position outside live sequence")
            live_vector = _vector.to(hidden.device, torch.float32)
            current = hidden[:, list(_indices)].float()
            current_values = torch.einsum("bsd,d->bs", current, live_vector)
            desired = _donor_values.to(hidden.device, torch.float32)
            replacement = current + (desired - current_values).unsqueeze(-1) * live_vector
            output = hidden.clone()
            output[:, list(_indices)] = replacement.to(hidden.dtype)
            return output

        edits[layer] = edit
        coefficient_manifest[str(layer)] = {
            "positions": indices,
            "recipient_clean_coefficients": recipient_values.cpu().tolist()[0],
            "donor_clean_coefficients": donor_values.cpu().tolist()[0],
        }
    return edits, {
        "operation": "same-layer/same-position concept-coordinate resampling",
        "layers": layers,
        "direction_dependency": "frozen raw J-Lens v_c",
        "independent_of_read_weights": True,
        "independent_of_read_gradients": True,
        "orthogonal_recipient_content_preserved": True,
        "coefficients": coefficient_manifest,
    }


def coordinate_resampling_edits_from_manifest(
    directions: Mapping[int, torch.Tensor],
    coefficient_manifest: Mapping[str | int, Any],
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Rebuild A edits from nb20's persisted scalar donor coefficients.

    This deliberately needs neither donor prompts nor full donor/recipient
    residual tensors.  It lets nb21 reproduce the exact A edit while retaining
    only small, auditable scalar manifests between notebooks.
    """

    layers = sorted(int(layer) for layer in directions)
    if layers != list(WORKSPACE_LAYERS):
        raise ValueError(f"Manifest resampling requires layers {WORKSPACE_LAYERS}")
    edits: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for layer in layers:
        record = coefficient_manifest.get(str(layer), coefficient_manifest.get(layer))
        if not isinstance(record, Mapping):
            raise ValueError(f"Coefficient manifest is missing L{layer}")
        indices = tuple(int(value) for value in record.get("positions", []))
        donor_values = tuple(
            float(value) for value in record.get("donor_clean_coefficients", [])
        )
        if not indices or len(indices) != len(donor_values):
            raise ValueError(f"Malformed coefficient manifest at L{layer}")
        if len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError(f"Invalid manifest positions at L{layer}")
        if not all(math.isfinite(value) for value in donor_values):
            raise ValueError(f"Non-finite donor coefficient at L{layer}")
        vector = directions[layer].detach().float()
        if not torch.isfinite(vector).all() or not torch.isclose(
            vector.norm(), torch.ones((), device=vector.device), atol=1e-4, rtol=1e-4
        ):
            raise ValueError(f"Manifest direction L{layer} is not unit")

        def edit(
            hidden: torch.Tensor,
            *,
            _indices=indices,
            _donor_values=donor_values,
            _vector=vector,
        ) -> torch.Tensor:
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("Manifest resampling hook requires [1,S,D]")
            if max(_indices) >= hidden.shape[1]:
                raise IndexError("Manifest position outside live sequence")
            live_vector = _vector.to(hidden.device, torch.float32)
            current = hidden[:, list(_indices)].float()
            current_values = torch.einsum("bsd,d->bs", current, live_vector)
            desired = torch.tensor(
                _donor_values,
                device=hidden.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            replacement = current + (desired - current_values).unsqueeze(-1) * live_vector
            output = hidden.clone()
            output[:, list(_indices)] = replacement.to(hidden.dtype)
            return output

        edits[layer] = edit
    return edits


def _metric_scalar(
    metric_fn: Callable[[torch.Tensor], torch.Tensor], logits: torch.Tensor
) -> torch.Tensor:
    value = metric_fn(logits)
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("Behavior metric must return one finite scalar")
    return value.float()


@torch.no_grad()
def coordinate_resampling_effect(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    edits: Mapping[int, Callable[[torch.Tensor], torch.Tensor]],
    *,
    clean_logits: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure primary A ground truth for already-frozen resampling edits."""

    clean = (
        clean_logits.float()
        if clean_logits is not None
        else forward_logits(hf_model, input_ids, attention_mask=attention_mask)
    )
    edited = forward_logits(
        hf_model,
        input_ids,
        attention_mask=attention_mask,
        blocks=blocks,
        edits=edits,
    )
    clean_metric = float(_metric_scalar(metric_fn, clean).cpu())
    edited_metric = float(_metric_scalar(metric_fn, edited).cpu())
    return {
        "status": "OK",
        "ground_truth": "A_coordinate_resampling_primary",
        "clean_metric": clean_metric,
        "edited_metric": edited_metric,
        "signed_delta_edited_minus_clean": edited_metric - clean_metric,
        "causal_abs": abs(edited_metric - clean_metric),
        "clean_top_token_id": int(clean[0, -1].argmax().cpu()),
        "edited_top_token_id": int(edited[0, -1].argmax().cpu()),
        "method_independence": (
            "independent of READ weights/gradients; dependent on frozen v_c"
        ),
    }


@torch.no_grad()
def masked_source_to_foil_effect(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    clean_residuals: Mapping[int, torch.Tensor],
    source_directions: Mapping[int, torch.Tensor],
    foil_directions: Mapping[int, torch.Tensor],
    carrying_positions: Sequence[int],
    *,
    clean_logits: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    alpha: float = SURGICAL_ALPHA,
) -> dict[str, Any]:
    """Measure secondary B using the accurately named frozen v3/v4 swap."""

    if float(alpha) != SURGICAL_ALPHA:
        raise ValueError(f"Frozen surgical alpha is {SURGICAL_ALPHA}")
    layers = set(WORKSPACE_LAYERS)
    if set(clean_residuals) != layers or set(source_directions) != layers or set(
        foil_directions
    ) != layers:
        raise ValueError("Surgical swap inputs must cover exactly L13-24")
    edits = clamped_swap_edits(
        clean_residuals,
        source_directions,
        foil_directions,
        positions=sorted(set(int(position) for position in carrying_positions)),
        strength=alpha,
    )
    clean = (
        clean_logits.float()
        if clean_logits is not None
        else forward_logits(hf_model, input_ids, attention_mask=attention_mask)
    )
    edited = forward_logits(
        hf_model,
        input_ids,
        attention_mask=attention_mask,
        blocks=blocks,
        edits=edits,
    )
    clean_metric = float(_metric_scalar(metric_fn, clean).cpu())
    edited_metric = float(_metric_scalar(metric_fn, edited).cpu())
    return {
        "status": "OK",
        "ground_truth": "B_masked_source_to_foil_clamped_swap_secondary",
        "alpha": float(alpha),
        "layers": list(WORKSPACE_LAYERS),
        "positions": sorted(set(int(value) for value in carrying_positions)),
        "clean_metric": clean_metric,
        "edited_metric": edited_metric,
        "signed_delta_edited_minus_clean": edited_metric - clean_metric,
        "causal_abs": abs(edited_metric - clean_metric),
        "clean_top_token_id": int(clean[0, -1].argmax().cpu()),
        "edited_top_token_id": int(edited[0, -1].argmax().cpu()),
        "intervention_label": (
            "masked source-to-foil clamped swap, not source-only deletion"
        ),
    }


def path_localization_from_edits(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    a_edits: Mapping[int, Callable[[torch.Tensor], torch.Tensor]],
    *,
    attention_mask: torch.Tensor | None = None,
    component_layers: Sequence[int] = COMPONENT_FRONTIER,
) -> dict[str, Any]:
    """Localize arbitrary multi-layer A edits at the frozen component frontier."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Path localization requires one unpadded item")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if any(parameter.requires_grad for parameter in hf_model.parameters()):
        raise ValueError("Freeze model parameters before path localization")
    frontier = tuple(sorted(set(int(layer) for layer in component_layers)))
    if frontier != COMPONENT_FRONTIER:
        raise ValueError(f"Frozen component frontier is {COMPONENT_FRONTIER}")
    head_geometry = {
        layer: (
            int(blocks[layer].self_attn.config.num_attention_heads),
            int(blocks[layer].self_attn.head_dim),
        )
        for layer in frontier
    }
    ordered = [
        *(("attention_pre_o_proj", layer) for layer in frontier),
        *(("mlp", layer) for layer in frontier),
    ]
    with (
        torch.enable_grad(),
        capture_qwen_components(blocks, frontier, start_graph=True) as clean_live,
    ):
        clean_logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        metric = _metric_scalar(metric_fn, clean_logits)
        tensors = tuple(clean_live[kind][layer] for kind, layer in ordered)
        gradient_values = torch.autograd.grad(
            metric,
            tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        clean = {
            kind: {layer: tensor.detach() for layer, tensor in values.items()}
            for kind, values in clean_live.items()
        }
        gradients: dict[str, dict[int, torch.Tensor]] = {
            "mlp": {},
            "attention_pre_o_proj": {},
        }
        for (kind, layer), gradient in zip(ordered, gradient_values, strict=True):
            gradients[kind][layer] = gradient.detach()

    with (
        torch.no_grad(),
        residual_edit_hooks(blocks, a_edits),
        capture_qwen_components(blocks, frontier, start_graph=False) as edited_live,
    ):
        edited_logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        edited_metric_tensor = _metric_scalar(metric_fn, edited_logits)
        edited = {
            kind: {layer: tensor.detach() for layer, tensor in values.items()}
            for kind, values in edited_live.items()
        }
    scores = component_grad_delta_scores(
        clean,
        edited,
        gradients,
        head_geometry=head_geometry,
    )
    clean_metric = float(metric.detach().cpu())
    edited_metric = float(edited_metric_tensor.detach().cpu())
    return {
        "status": "OK",
        "clean_metric": clean_metric,
        "a_edited_metric": edited_metric,
        "a_actual_delta": edited_metric - clean_metric,
        "component_layers": list(frontier),
        "mlps": scores["mlps"],
        "attention_heads": scores["attention_heads"],
        "definition": {
            "formula": (
                "grad(M_clean, component) dot "
                "(component_A_edited - component_clean)"
            ),
            "A_edits_may_cover_multiple_layers": True,
            "component_frontier": list(frontier),
            "score_by_position_retained": True,
            "warning": "component scores overlap and are not additive",
        },
    }


def _component_rows(
    localization: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mlps = [dict(row) for row in localization.get("mlps", [])]
    heads = [dict(row) for row in localization.get("attention_heads", [])]
    expected_mlps = {f"L{layer}.MLP" for layer in COMPONENT_FRONTIER}
    expected_heads = {
        f"L{layer}.H{head}"
        for layer in COMPONENT_FRONTIER
        for head in range(QWEN25_7B_HEADS_PER_LAYER)
    }
    if {str(row.get("component")) for row in mlps} != expected_mlps:
        raise ValueError("Localization does not cover every frontier MLP exactly once")
    if {str(row.get("component")) for row in heads} != expected_heads:
        raise ValueError("Localization does not cover all 84 frontier heads exactly once")
    if any(int(row["layer"]) not in COMPONENT_FRONTIER for row in heads):
        raise ValueError("Localization contains a head outside L25-27")
    return mlps, heads


def normalized_prompt_path_shares(
    localization: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one prompt's absolute path scores over the full frontier."""

    mlps, heads = _component_rows(localization)
    all_rows = [*mlps, *heads]
    values = np.asarray([float(row["abs_score"]) for row in all_rows], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Path magnitudes must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0.0:
        return {
            "status": "UNDEFINED_ZERO_TOTAL_PATH_ATTRIBUTION",
            "shares": None,
            "total_abs_score": total,
        }
    return {
        "status": "OK",
        "shares": {
            str(row["component"]): float(value / total)
            for row, value in zip(all_rows, values, strict=True)
        },
        "total_abs_score": total,
    }


def _selected_component_row(
    component: str,
    score: float,
    *,
    rank: int,
    family: str,
) -> dict[str, Any]:
    prefix, suffix = component.split(".", 1)
    if not prefix.startswith("L"):
        raise ValueError(f"Malformed component ID {component!r}")
    layer = int(prefix[1:])
    row: dict[str, Any] = {
        "component": component,
        "layer": layer,
        "train_mean_normalized_path_share": float(score),
        "train_path_rank_within_family": int(rank),
        "family": family,
    }
    if family == "attention_heads":
        if not suffix.startswith("H"):
            raise ValueError(f"Malformed attention component ID {component!r}")
        row["head"] = int(suffix[1:])
    elif suffix != "MLP":
        raise ValueError(f"Malformed MLP component ID {component!r}")
    return row


def _relative_position_allocation_fit(
    training_by_concept: Mapping[str, Sequence[Mapping[str, Any]]],
    component_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit train-only component allocation shapes on a fixed relative grid."""

    grid = np.linspace(0.0, 1.0, RELATIVE_POSITION_GRID_SIZE, dtype=float)
    records: dict[str, Any] = {}
    for component in component_ids:
        concept_profiles: list[np.ndarray] = []
        missing_by_concept: dict[str, list[str]] = {}
        for concept in sorted(training_by_concept):
            prompt_profiles: list[np.ndarray] = []
            missing_rows: list[str] = []
            for row in training_by_concept[concept]:
                lookup = _localization_lookup(row["path_localization"])
                record = lookup.get(str(component))
                values = (
                    np.abs(np.asarray(record["score_by_position"], dtype=float))
                    if record is not None
                    else np.asarray([], dtype=float)
                )
                total = float(values.sum()) if len(values) else 0.0
                if (
                    not len(values)
                    or not np.isfinite(values).all()
                    or total <= 0.0
                ):
                    missing_rows.append(str(row["row_id"]))
                    continue
                normalized = values / total
                if len(normalized) == 1:
                    interpolated = np.ones_like(grid)
                else:
                    interpolated = np.interp(
                        grid,
                        np.linspace(0.0, 1.0, len(normalized), dtype=float),
                        normalized,
                    )
                interpolated /= float(interpolated.sum())
                prompt_profiles.append(interpolated)
            if missing_rows:
                missing_by_concept[concept] = sorted(missing_rows)
            elif prompt_profiles:
                concept_profile = np.mean(np.stack(prompt_profiles), axis=0)
                concept_profile /= float(concept_profile.sum())
                concept_profiles.append(concept_profile)
            else:
                missing_by_concept[concept] = ["NO_PROMPT_PROFILE"]
        if missing_by_concept or len(concept_profiles) != len(training_by_concept):
            records[str(component)] = {
                "status": "UNDEFINED_INCOMPLETE_TRAIN_POSITION_PROFILE",
                "relative_allocation": None,
                "missing_rows_by_concept": missing_by_concept,
            }
        else:
            profile = np.mean(np.stack(concept_profiles), axis=0)
            profile /= float(profile.sum())
            records[str(component)] = {
                "status": "OK",
                "relative_allocation": profile.tolist(),
                "missing_rows_by_concept": {},
            }
    return {
        "status": (
            "OK"
            if all(record["status"] == "OK" for record in records.values())
            else "INCOMPLETE_TRAIN_POSITION_PROFILES"
        ),
        "relative_grid": grid.tolist(),
        "grid_size": RELATIVE_POSITION_GRID_SIZE,
        "components": records,
        "fit_rows": "training folds only",
        "aggregation": (
            "absolute position attribution normalized within prompt, interpolated "
            "to relative grid, averaged within concept, then equally across concepts"
        ),
        "labels_read": False,
    }


def fit_fold_component_paths(
    rows: Sequence[Mapping[str, Any]],
    *,
    heldout_fold: int,
    n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    """Manifest global R1 and fit label-blind R2/R3 paths on training concepts."""

    if not 0 <= int(heldout_fold) < n_folds or n_folds != N_FOLDS:
        raise ValueError("Invalid held-out fold for the frozen four-fold protocol")
    row_list = [dict(row) for row in rows]
    training = [row for row in row_list if int(row["fold"]) != heldout_fold]
    if not training:
        raise ValueError("A fold path requires training rows")

    component_sets: list[set[str]] = []
    for row in row_list:
        localization = row.get("path_localization")
        if not isinstance(localization, Mapping) or localization.get("status") != "OK":
            continue
        mlps, heads = _component_rows(localization)
        component_sets.append(
            {str(record["component"]) for record in (*mlps, *heads)}
        )
    if not component_sets:
        raise ValueError("No usable localization defines the common R1 frontier")
    common_components = set.intersection(*component_sets)
    if any(component_set != common_components for component_set in component_sets):
        raise ValueError("Global R1 requires one common frontier component universe")
    global_mlps = sorted(
        component for component in common_components if component.endswith(".MLP")
    )
    global_heads = sorted(
        component for component in common_components if ".H" in component
    )
    if len(global_mlps) != len(COMPONENT_FRONTIER) or not global_heads:
        raise ValueError("Global R1 frontier is missing MLPs or attention heads")

    by_concept: dict[str, list[dict[str, float]]] = defaultdict(list)
    training_members_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unavailable: list[str] = []
    for row in training:
        localization = row.get("path_localization")
        if not isinstance(localization, Mapping) or localization.get("status") != "OK":
            unavailable.append(str(row["row_id"]))
            continue
        training_members_by_concept[str(row["concept_id"])].append(row)
        normalized = normalized_prompt_path_shares(localization)
        if normalized["status"] != "OK":
            unavailable.append(str(row["row_id"]))
            continue
        by_concept[str(row["concept_id"])].append(normalized["shares"])
    if not by_concept:
        raise ValueError("No training concepts have path attribution")

    component_ids = sorted(
        {
            component
            for concept_rows in by_concept.values()
            for shares in concept_rows
            for component in shares
        }
    )
    concept_means: dict[str, dict[str, float]] = {}
    for concept, concept_rows in sorted(by_concept.items()):
        concept_means[concept] = {
            component: float(
                np.mean([shares.get(component, 0.0) for shares in concept_rows])
            )
            for component in component_ids
        }
    train_means = {
        component: float(
            np.mean(
                [values.get(component, 0.0) for values in concept_means.values()]
            )
        )
        for component in component_ids
    }
    mlp_ranking = sorted(
        (
            (component, score)
            for component, score in train_means.items()
            if component.endswith(".MLP")
        ),
        key=lambda item: (-item[1], item[0]),
    )
    head_ranking = sorted(
        (
            (component, score)
            for component, score in train_means.items()
            if ".H" in component
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if len(mlp_ranking) < PATH_MLPS or len(head_ranking) < PATH_HEADS:
        raise ValueError("Component frontier cannot supply the frozen stratified top-k")

    def selection(n_mlps: int, n_heads: int, name: str) -> dict[str, Any]:
        mlp_rows = [
            _selected_component_row(
                component,
                score,
                rank=rank,
                family="mlps",
            )
            for rank, (component, score) in enumerate(
                mlp_ranking[:n_mlps], start=1
            )
        ]
        head_rows = [
            _selected_component_row(
                component,
                score,
                rank=rank,
                family="attention_heads",
            )
            for rank, (component, score) in enumerate(
                head_ranking[:n_heads], start=1
            )
        ]
        return {
            "name": name,
            "mlps": mlp_rows,
            "attention_heads": head_rows,
            "component_ids": [
                row["component"] for row in (*mlp_rows, *head_rows)
            ],
            "n_components": len(mlp_rows) + len(head_rows),
            "stratification": {"mlps": n_mlps, "attention_heads": n_heads},
            "selection_fit_on_train_fold": True,
        }

    def global_r1_selection() -> dict[str, Any]:
        mlp_rows = [
            {
                "component": component,
                "layer": int(component.split(".", 1)[0][1:]),
                "family": "mlps",
            }
            for component in global_mlps
        ]
        head_rows = [
            {
                "component": component,
                "layer": int(component.split(".", 1)[0][1:]),
                "head": int(component.split(".H", 1)[1]),
                "family": "attention_heads",
            }
            for component in global_heads
        ]
        return {
            "name": "R1_global_frontier_weight_read",
            "mlps": mlp_rows,
            "attention_heads": head_rows,
            "component_ids": [
                record["component"] for record in (*mlp_rows, *head_rows)
            ],
            "n_components": len(mlp_rows) + len(head_rows),
            "stratification": {
                "mlps": len(mlp_rows),
                "attention_heads": len(head_rows),
            },
            "selection_fit_on_train_fold": False,
            "scope_limitation": READ_VALIDATION_PROTOCOL["component_frontier_scope"],
        }

    r23_selection = selection(
        PATH_MLPS,
        PATH_HEADS,
        "R2_R3_train_fold_stratified_top8",
    )
    r23_selection["R2_train_relative_position_fit"] = (
        _relative_position_allocation_fit(
            training_members_by_concept,
            r23_selection["component_ids"],
        )
    )

    return {
        "status": "OK",
        "heldout_fold": int(heldout_fold),
        "training_row_ids": sorted(str(row["row_id"]) for row in training),
        "training_concept_ids": sorted(concept_means),
        "n_training_rows": len(training),
        "n_training_concepts": len(concept_means),
        "training_path_coverage_complete": not unavailable,
        "unavailable_training_row_ids": sorted(unavailable),
        "undefined_training_rows_retained_in_validation_roster": True,
        "labels_read_during_selection": False,
        "aggregation": (
            "prompt-normalized shares; concept mean first; equal concept mean second"
        ),
        "R1": global_r1_selection(),
        "R2_R3": r23_selection,
        "train_mean_normalized_path_share": train_means,
    }


def fit_all_fold_component_paths(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fit all four support paths without consulting labels."""

    return {
        str(fold): fit_fold_component_paths(rows, heldout_fold=fold)
        for fold in range(N_FOLDS)
    }


def _mlp_weight_component(
    bundle: Any,
    directions: Mapping[int, torch.Tensor],
    *,
    layer: int,
    seed: int,
) -> dict[str, Any]:
    """Compute one component using the unchanged v2/v3 MLP formula."""

    input_direction = directions[layer - 1]
    label_direction = directions[layer]
    block = bundle.lens_model.layers[layer]
    weight = qwen_mlp_gain(
        block,
        input_direction,
        n_random=32,
        seed=int(seed) + 10_007 * layer,
    )
    with torch.no_grad():
        vector = input_direction.to(
            next(block.parameters()).device,
            next(block.parameters()).dtype,
        )
        output = block.mlp(block.post_attention_layernorm(vector)).float()
        label = label_direction.to(output.device, torch.float32)
        cosine = float(F.cosine_similarity(output, label, dim=0).cpu())
    null = np.asarray(weight["random_gains"], dtype=float)
    return {
        **weight,
        "input_direction_layer": layer - 1,
        "label_direction_layer": layer,
        "label_cosine": cosine,
        "oriented_normalized_gain": float(weight["normalized_gain"]) * cosine,
        "gain_random_percentile": float(np.mean(null <= float(weight["gain"]))),
    }


@torch.no_grad()
def _attention_weight_component(
    bundle: Any,
    directions: Mapping[int, torch.Tensor],
    *,
    layer: int,
    head: int,
    seed: int,
) -> dict[str, Any]:
    """Exact per-head equivalent of the v2/v3 all-head null helper.

    The upstream helper recomputes every head for each requested head.  This
    implementation evaluates only the requested slice while preserving its
    observed formula, 32 random directions, and component-specific seed.
    """

    attention = bundle.lens_model.layers[layer].self_attn
    num_heads = int(attention.config.num_attention_heads)
    num_kv_heads = int(attention.config.num_key_value_heads)
    head_dim = int(attention.head_dim)
    width = int(directions[layer - 1].numel())
    if num_heads % num_kv_heads:
        raise ValueError("Query-head count must be divisible by KV-head count")
    if not 0 <= int(head) < num_heads:
        raise IndexError(f"Attention head {head} outside L{layer}")
    v_weight = attention.v_proj.weight.detach().float()
    o_weight = attention.o_proj.weight.detach().float()
    if v_weight.shape != (num_kv_heads * head_dim, width):
        raise ValueError("Qwen value weight shape disagrees with head geometry")
    if o_weight.shape[1] != num_heads * head_dim:
        raise ValueError("Qwen output weight shape disagrees with head geometry")

    group_size = num_heads // num_kv_heads
    kv_head = int(head) // group_size
    value_slice = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
    output_slice = slice(int(head) * head_dim, (int(head) + 1) * head_dim)
    input_direction = F.normalize(directions[layer - 1].detach().float(), dim=0)
    label = F.normalize(directions[layer].detach().float(), dim=0)
    vector = input_direction.to(v_weight.device)
    live_label = label.to(o_weight.device)
    value = v_weight[value_slice] @ vector
    observed_output = o_weight[:, output_slice] @ value.to(o_weight.device)
    observed_norm = float(observed_output.norm().cpu())
    observed_cosine = float(
        F.cosine_similarity(observed_output, live_label, dim=0).cpu()
        if observed_norm > 0.0
        else torch.tensor(float("nan"))
    )

    component_seed = int(seed) + 1_009 + 10_007 * layer + 101 * int(head)
    generator = torch.Generator(device="cpu").manual_seed(component_seed)
    random_vectors = F.normalize(
        torch.randn(32, width, generator=generator, dtype=torch.float32),
        dim=-1,
    )
    random_values = random_vectors.to(v_weight.device) @ v_weight[value_slice].T
    random_outputs = random_values.to(o_weight.device) @ o_weight[:, output_slice].T
    random_norms_tensor = random_outputs.norm(dim=-1)
    random_cosines_tensor = (random_outputs @ live_label) / random_norms_tensor
    random_norms = random_norms_tensor.cpu().numpy().astype(np.float64)
    random_cosines = random_cosines_tensor.cpu().numpy().astype(np.float64)
    if not np.isfinite(random_norms).all():
        raise ValueError(f"Head L{layer}.H{head} OV random null is non-finite")
    finite_cosines = random_cosines[np.isfinite(random_cosines)]
    median = float(np.median(random_norms))
    normalized = observed_norm / median if median > 0.0 else None
    label_weighted = (
        observed_norm / median * abs(observed_cosine)
        if median > 0.0 and math.isfinite(observed_cosine)
        else None
    )
    return {
        "head": int(head),
        "kv_head": kv_head,
        "ov_norm": observed_norm,
        "label_cosine": observed_cosine,
        "normalized_ov_norm": normalized,
        "random_median_ov_norm": median,
        "random_ov_norms": [float(value) for value in random_norms],
        "ov_norm_random_percentile": float(np.mean(random_norms <= observed_norm)),
        "random_label_cosines": [float(value) for value in random_cosines],
        "label_cosine_random_percentile": (
            float(np.mean(finite_cosines <= observed_cosine))
            if finite_cosines.size
            else None
        ),
        "label_weighted_normalized_ov": label_weighted,
        "n_random": 32,
        "seed": component_seed,
        "input_direction_layer": layer - 1,
        "label_direction_layer": layer,
        "oriented_normalized_ov": float(normalized) * observed_cosine,
    }


def repaired_weight_read(
    bundle: Any,
    directions: Mapping[int, torch.Tensor],
    selection: Mapping[str, Any],
    *,
    seed: int,
    component_cache: MutableMapping[tuple[str, int, str], dict[str, Any]] | None = None,
    cache_namespace: str = "",
) -> dict[str, Any]:
    """Apply the unchanged repaired v2/v3 formula with component caching."""

    mlp_flags = [dict(row) for row in selection.get("mlps", [])]
    head_flags = [dict(row) for row in selection.get("attention_heads", [])]
    if not mlp_flags or not head_flags:
        raise ValueError("Repaired weight READ requires both frozen component families")

    def cached_core(component: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        key = (str(cache_namespace), int(seed), str(component))
        if component_cache is None:
            return compute()
        if key not in component_cache:
            component_cache[key] = compute()
        return dict(component_cache[key])

    mlps: list[dict[str, Any]] = []
    for flag in mlp_flags:
        layer = int(flag["layer"])
        component = str(flag["component"])
        core = cached_core(
            component,
            lambda _layer=layer: _mlp_weight_component(
                bundle,
                directions,
                layer=_layer,
                seed=int(seed),
            ),
        )
        mlps.append({**flag, **core})
    heads: list[dict[str, Any]] = []
    for flag in head_flags:
        layer = int(flag["layer"])
        head = int(flag["head"])
        component = str(flag["component"])
        core = cached_core(
            component,
            lambda _layer=layer, _head=head: _attention_weight_component(
                bundle,
                directions,
                layer=_layer,
                head=_head,
                seed=int(seed),
            ),
        )
        heads.append({**flag, **core})

    mlp_primary = float(np.mean([row["normalized_gain"] for row in mlps]))
    attention_primary = float(
        np.mean([row["label_weighted_normalized_ov"] for row in heads])
    )
    result = {
        "mlps": mlps,
        "attention_heads": heads,
        "mlp_primary": mlp_primary,
        "attention_primary": attention_primary,
        "equal_family_composite": 0.5 * mlp_primary + 0.5 * attention_primary,
        "mlp_mean_random_percentile": float(
            np.mean([row["gain_random_percentile"] for row in mlps])
        ),
        "attention_mean_random_percentile": float(
            np.mean([row["ov_norm_random_percentile"] for row in heads])
        ),
        "mlp_mean_oriented": float(
            np.mean([row["oriented_normalized_gain"] for row in mlps])
        ),
        "attention_mean_oriented": float(
            np.mean([row["oriented_normalized_ov"] for row in heads])
        ),
        "metadata": {
            "activation_independent_primary_magnitude": True,
            "selection_conditioned": True,
            "input_direction": "v[layer-1]",
            "label_direction": "v[layer]",
            "n_random": 32,
            "per_head_exact_null_optimization": True,
        },
    }
    fit_on_train = bool(selection.get("selection_fit_on_train_fold", True))
    return {
        **result,
        "selection_name": selection.get("name"),
        "selection_fit_on_train_fold": fit_on_train,
        "formula_source": (
            "exact-equivalent optimized port of "
            "src.v2_read._layer_aligned_weight_read"
        ),
        "metadata": {
            **dict(result["metadata"]),
            "selection_conditioned": fit_on_train,
            "global_common_frontier": not fit_on_train,
        },
    }


def _localization_lookup(
    localization: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    mlps, heads = _component_rows(localization)
    lookup = {str(row["component"]): row for row in (*mlps, *heads)}
    if len(lookup) != len(mlps) + len(heads):
        raise ValueError("Localization component IDs are duplicated")
    return lookup


def _weight_lookup(weight_read: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        *[dict(row) for row in weight_read.get("mlps", [])],
        *[dict(row) for row in weight_read.get("attention_heads", [])],
    ]
    lookup = {str(row["component"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Weight READ component IDs are duplicated")
    return lookup


def profile_summaries(
    profile: Mapping[int | str, Sequence[float]],
    carrying_positions: Sequence[int],
    *,
    sum_reduction: str = "total",
) -> dict[str, Any]:
    """Return the frozen peak/carrying and requested sum-style summary.

    ``carrying`` is a mean over every selected layer-by-position cell, so a
    prompt is not rewarded merely for having more carrying positions.  R2 uses
    ``total`` only as an allocation audit; R3 uses ``position_mean`` so sequence
    length cannot itself act as a classifier.
    """

    if sum_reduction not in {"total", "position_mean"}:
        raise ValueError("sum_reduction must be total or position_mean")

    if not profile:
        return {
            "status": "UNDEFINED_EMPTY_PROFILE",
            "sum": None,
            "peak": None,
            "carrying": None,
        }
    arrays = {
        int(layer): np.asarray(values, dtype=float).reshape(-1)
        for layer, values in profile.items()
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not lengths or not all(
        np.isfinite(values).all() and np.all(values >= 0.0)
        for values in arrays.values()
    ):
        return {
            "status": "UNDEFINED_NONFINITE_OR_MISALIGNED_PROFILE",
            "sum": None,
            "peak": None,
            "carrying": None,
        }
    sequence_length = lengths.pop()
    carrying = sorted(set(int(position) for position in carrying_positions))
    if not carrying or carrying[0] < 0 or carrying[-1] >= sequence_length:
        return {
            "status": "UNDEFINED_CARRYING_POSITION",
            "sum": None,
            "peak": None,
            "carrying": None,
        }
    matrix = np.stack([arrays[layer] for layer in sorted(arrays)], axis=0)
    sum_value = (
        float(matrix.sum())
        if sum_reduction == "total"
        else float(matrix.sum(axis=0).mean())
    )
    return {
        "status": "OK",
        "sum": sum_value,
        "peak": float(matrix.max()),
        "carrying": float(matrix[:, carrying].mean()),
        "sum_reduction": sum_reduction,
        "layers": sorted(arrays),
        "sequence_length": sequence_length,
        "carrying_positions": carrying,
    }


def r2_weight_profile(
    selection: Mapping[str, Any],
    weight_read: Mapping[str, Any],
    carrying_positions: Sequence[int],
    *,
    sequence_length: int,
) -> dict[str, Any]:
    """Allocate static R2 weights with train-only relative-position profiles."""

    if int(sequence_length) < 1:
        raise ValueError("R2 held-out sequence length must be positive")
    sequence_length = int(sequence_length)
    weights = _weight_lookup(weight_read)
    selected_mlps = [dict(row) for row in selection.get("mlps", [])]
    selected_heads = [dict(row) for row in selection.get("attention_heads", [])]
    if len(selected_mlps) != PATH_MLPS or len(selected_heads) != PATH_HEADS:
        raise ValueError("R2 requires the frozen 2-MLP/6-head path")
    fit = selection.get("R2_train_relative_position_fit")
    if not isinstance(fit, Mapping):
        raise ValueError("R2 selection lacks its training-fold position-profile fit")
    grid = np.asarray(fit.get("relative_grid"), dtype=float)
    fitted_components = fit.get("components")
    if (
        grid.shape != (RELATIVE_POSITION_GRID_SIZE,)
        or not np.isfinite(grid).all()
        or not isinstance(fitted_components, Mapping)
    ):
        raise ValueError("Malformed R2 training relative-position fit")
    target_grid = np.linspace(0.0, 1.0, sequence_length, dtype=float)
    profile = {
        layer: np.zeros(sequence_length, dtype=float) for layer in COMPONENT_FRONTIER
    }
    undefined: list[dict[str, Any]] = []

    for family, rows, divisor in (
        ("mlp", selected_mlps, 2.0 * len(selected_mlps)),
        ("attention", selected_heads, 2.0 * len(selected_heads)),
    ):
        for selected in rows:
            component = str(selected["component"])
            if component not in weights:
                raise KeyError(f"R2 component {component} missing from static weights")
            fitted = fitted_components.get(component)
            if not isinstance(fitted, Mapping) or fitted.get("status") != "OK":
                undefined.append(
                    {
                        "component": component,
                        "reason": "UNDEFINED_TRAIN_RELATIVE_POSITION_PROFILE",
                    }
                )
                continue
            relative = np.asarray(fitted.get("relative_allocation"), dtype=float)
            if relative.shape != grid.shape or not np.isfinite(relative).all():
                undefined.append(
                    {"component": component, "reason": "MALFORMED_TRAIN_PROFILE"}
                )
                continue
            allocation = np.interp(target_grid, grid, relative)
            allocation_total = float(allocation.sum())
            if allocation_total <= 0.0:
                undefined.append(
                    {"component": component, "reason": "ZERO_INTERPOLATED_PROFILE"}
                )
                continue
            allocation /= allocation_total
            if family == "mlp":
                static_weight = float(weights[component]["normalized_gain"])
            else:
                static_weight = float(
                    weights[component]["label_weighted_normalized_ov"]
                )
            if not math.isfinite(static_weight) or static_weight < 0.0:
                undefined.append(
                    {"component": component, "reason": "NONFINITE_STATIC_WEIGHT"}
                )
                continue
            layer = int(selected["layer"])
            profile[layer] += (static_weight / divisor) * allocation

    static_sum = _finite_float(weight_read.get("equal_family_composite"))
    if static_sum is None:
        raise ValueError("R2 static equal-family composite is undefined")
    if undefined:
        return {
            "status": "PARTIAL_STATIC_SUM_ONLY",
            "undefined_components": undefined,
            "profile": None,
            "summaries": {
                "status": "PARTIAL_STATIC_SUM_ONLY",
                "sum": static_sum,
                "peak": None,
                "carrying": None,
                "sum_definition": "static repaired-weight equal-family composite",
            },
            "heldout_A_localization_used_for_position_allocation": False,
        }
    json_profile = {str(layer): values.tolist() for layer, values in profile.items()}
    summaries = profile_summaries(json_profile, carrying_positions)
    expected_sum = static_sum
    if summaries["status"] == "OK" and not math.isclose(
        float(summaries["sum"]), expected_sum, rel_tol=1e-6, abs_tol=1e-7
    ):
        raise RuntimeError("R2 profile allocation does not preserve repaired weight READ")
    summaries["sum"] = static_sum
    summaries["sum_definition"] = "static repaired-weight equal-family composite"
    return {
        "status": summaries["status"],
        "profile": json_profile,
        "summaries": summaries,
        "selected_components": list(selection["component_ids"]),
        "allocation": (
            "equal-family static weight times training-fold relative-position "
            "allocation interpolated to held-out sequence length"
        ),
        "path_selection_fit_on_train_fold": True,
        "position_profile_fit_on_train_fold": True,
        "heldout_A_localization_used_for_position_allocation": False,
    }


def _component_parts(component: str) -> tuple[int, str, int | None]:
    prefix, suffix = str(component).split(".", 1)
    if not prefix.startswith("L"):
        raise ValueError(f"Malformed component ID {component!r}")
    layer = int(prefix[1:])
    if layer not in COMPONENT_FRONTIER:
        raise ValueError(f"R3 component {component!r} lies outside L25-27")
    if suffix == "MLP":
        return layer, "mlp", None
    if suffix.startswith("H"):
        return layer, "attention_head", int(suffix[1:])
    raise ValueError(f"Malformed component ID {component!r}")


@contextmanager
def _capture_r3_graph(
    blocks: Sequence[torch.nn.Module],
    layers: Sequence[int],
) -> Iterator[dict[str, dict[int, torch.Tensor]]]:
    """Capture real local roots and outputs without detaching their graph."""

    layer_list = tuple(sorted(set(int(layer) for layer in layers)))
    if layer_list != COMPONENT_FRONTIER:
        raise ValueError(f"R3 capture requires {COMPONENT_FRONTIER}")
    captures: dict[str, dict[int, torch.Tensor]] = {
        "pre_attention_residual": {},
        "post_attention_residual": {},
        "mlp_output": {},
        "attention_pre_o_proj": {},
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def root_tensor(tensor: torch.Tensor, *, kind: str, layer: int) -> torch.Tensor:
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            raise ValueError(f"R3 {kind} L{layer} must have shape [B,S,D]")
        if layer in captures[kind]:
            raise RuntimeError(f"R3 {kind} L{layer} executed more than once")
        if not tensor.requires_grad:
            tensor.requires_grad_(True)
        captures[kind][layer] = tensor
        return tensor

    try:
        for layer in layer_list:
            block = blocks[layer]

            def pre_attention_hook(module, inputs, *, _layer=layer):
                del module
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError("Qwen input layernorm has no tensor input")
                prepared = root_tensor(
                    inputs[0], kind="pre_attention_residual", layer=_layer
                )
                return (prepared, *inputs[1:])

            def post_attention_hook(module, inputs, *, _layer=layer):
                del module
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError("Qwen post-attention layernorm has no tensor input")
                prepared = root_tensor(
                    inputs[0], kind="post_attention_residual", layer=_layer
                )
                return (prepared, *inputs[1:])

            def mlp_hook(module, inputs, output, *, _layer=layer):
                del module, inputs
                if not torch.is_tensor(output) or output.ndim != 3:
                    raise TypeError("Qwen MLP output must be a [B,S,D] tensor")
                captures["mlp_output"][_layer] = output

            def attention_pre_o_hook(module, inputs, *, _layer=layer):
                del module
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError("Qwen o_proj has no tensor input")
                if inputs[0].ndim != 3:
                    raise ValueError("Qwen pre-o attention stream must be [B,S,D]")
                captures["attention_pre_o_proj"][_layer] = inputs[0]

            handles.append(
                block.input_layernorm.register_forward_pre_hook(pre_attention_hook)
            )
            handles.append(
                block.post_attention_layernorm.register_forward_pre_hook(
                    post_attention_hook
                )
            )
            handles.append(block.mlp.register_forward_hook(mlp_hook))
            handles.append(
                block.self_attn.o_proj.register_forward_pre_hook(
                    attention_pre_o_hook
                )
            )
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def _reverse_directional_profile(
    projected_output: torch.Tensor,
    local_root: torch.Tensor,
    input_direction: torch.Tensor,
    *,
    retain_graph: bool,
) -> list[float]:
    """Mean |d projected output q / d concept-coordinate p| over q>=p."""

    if projected_output.ndim != 1:
        raise ValueError("Projected R3 output must have one value per q position")
    if local_root.ndim != 3 or local_root.shape[0] != 1:
        raise ValueError("R3 local root must have shape [1,S,D]")
    sequence_length = int(projected_output.shape[0])
    if local_root.shape[1] != sequence_length:
        raise ValueError("R3 output/root sequence lengths disagree")
    identity = torch.eye(
        sequence_length,
        device=projected_output.device,
        dtype=projected_output.dtype,
    )
    jacobian = torch.autograd.grad(
        projected_output,
        local_root,
        grad_outputs=identity,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
        is_grads_batched=True,
    )[0]
    if jacobian.shape[:3] != (sequence_length, 1, sequence_length):
        raise RuntimeError(f"Unexpected R3 batched-VJP shape {tuple(jacobian.shape)}")
    vector = input_direction.detach().to(jacobian.device, torch.float32)
    directional = torch.einsum("qpd,d->qp", jacobian[:, 0].float(), vector)
    values: list[float] = []
    for source_position in range(sequence_length):
        reachable = directional[source_position:, source_position]
        if reachable.numel() != sequence_length - source_position:
            raise RuntimeError("R3 reachable-q accounting drifted")
        values.append(float(reachable.abs().sum().cpu() / reachable.numel()))
    return values


def r3_component_profiles(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    directions: Mapping[int, torch.Tensor],
    component_ids: Sequence[str],
    *,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute exact real-activation local derivative profiles once per component.

    Reverse autograd is used directly at the real activation.  MLP outputs are
    label-projected and differentiated with respect to the post-attention
    residual.  A head's pre-o slice is sent through its own W_O slice, projected
    to v[layer], and differentiated with respect to the pre-attention residual.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("R3 requires one unpadded item")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if any(parameter.requires_grad for parameter in hf_model.parameters()):
        raise ValueError("Freeze model parameters before R3")
    required_directions = set(range(min(COMPONENT_FRONTIER) - 1, max(COMPONENT_FRONTIER) + 1))
    if not required_directions.issubset(set(int(layer) for layer in directions)):
        raise ValueError("R3 directions must cover L24-27")
    components = sorted(set(str(value) for value in component_ids))
    if not components:
        raise ValueError("R3 requires selected components")
    parsed = {component: _component_parts(component) for component in components}

    with torch.enable_grad(), _capture_r3_graph(
        blocks, COMPONENT_FRONTIER
    ) as captured:
        hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        profiles: dict[str, dict[str, Any]] = {}
        for index, component in enumerate(components):
            layer, kind, head = parsed[component]
            label = directions[layer].detach().to(
                next(blocks[layer].parameters()).device,
                torch.float32,
            )
            if kind == "mlp":
                output = captured["mlp_output"][layer][0].float()
                projected = output @ label
                root = captured["post_attention_residual"][layer]
                root_name = "post_attention_residual"
                output_name = "label-projected MLP output"
            else:
                assert head is not None
                attention = blocks[layer].self_attn
                num_heads = int(attention.config.num_attention_heads)
                head_dim = int(attention.head_dim)
                if not 0 <= head < num_heads:
                    raise IndexError(f"R3 head {component} outside layer")
                start = head * head_dim
                stop = start + head_dim
                stream = captured["attention_pre_o_proj"][layer][0, :, start:stop]
                o_slice = attention.o_proj.weight[:, start:stop].float()
                head_read_vector = o_slice.T @ label.to(o_slice.device)
                projected = stream.float() @ head_read_vector
                root = captured["pre_attention_residual"][layer]
                root_name = "pre_attention_residual"
                output_name = "W_O-sliced, label-projected head output"
            values = _reverse_directional_profile(
                projected,
                root,
                directions[layer - 1],
                retain_graph=index < len(components) - 1,
            )
            profiles[component] = {
                "component": component,
                "layer": layer,
                "kind": kind,
                "head": head,
                "profile_by_source_position": values,
                "input_direction_layer": layer - 1,
                "label_direction_layer": layer,
                "local_root": root_name,
                "component_output": output_name,
                "reachable_q_normalization": (
                    "sum absolute q>=p derivatives divided by count(q>=p)"
                ),
            }
    return {
        "status": "OK",
        "components": profiles,
        "component_ids": components,
        "definition": READ_VALIDATION_PROTOCOL["profiles"]["R3"],
        "autograd": "exact reverse-mode batched VJP at the real activation",
    }


def r3_profile_for_selection(
    component_profiles: Mapping[str, Any],
    selection: Mapping[str, Any],
    carrying_positions: Sequence[int],
) -> dict[str, Any]:
    """Sum the selected R3 component profiles and return frozen summaries."""

    selected = [str(value) for value in selection.get("component_ids", [])]
    if len(selected) != PATH_MLPS + PATH_HEADS:
        raise ValueError("R3 requires the frozen stratified top-eight path")
    available = component_profiles.get("components")
    if not isinstance(available, Mapping):
        raise ValueError("R3 component profile payload is malformed")
    missing = [component for component in selected if component not in available]
    if missing:
        return {
            "status": "UNDEFINED_MISSING_COMPONENT_PROFILE",
            "missing_components": missing,
            "profile": None,
            "summaries": {
                "status": "UNDEFINED_MISSING_COMPONENT_PROFILE",
                "sum": None,
                "peak": None,
                "carrying": None,
            },
        }
    lengths = {
        len(available[component]["profile_by_source_position"])
        for component in selected
    }
    if len(lengths) != 1:
        raise ValueError("Selected R3 component profile lengths disagree")
    sequence_length = lengths.pop()
    profile = {
        layer: np.zeros(sequence_length, dtype=float) for layer in COMPONENT_FRONTIER
    }
    for component in selected:
        record = available[component]
        values = np.asarray(record["profile_by_source_position"], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            return {
                "status": "UNDEFINED_NONFINITE_COMPONENT_PROFILE",
                "component": component,
                "profile": None,
                "summaries": {
                    "status": "UNDEFINED_NONFINITE_COMPONENT_PROFILE",
                    "sum": None,
                    "peak": None,
                    "carrying": None,
                },
            }
        profile[int(record["layer"])] += values
    json_profile = {str(layer): values.tolist() for layer, values in profile.items()}
    summaries = profile_summaries(
        json_profile,
        carrying_positions,
        sum_reduction="position_mean",
    )
    summaries["sum_definition"] = (
        "mean over positions after summing all selected component profiles"
    )
    return {
        "status": summaries["status"],
        "profile": json_profile,
        "summaries": summaries,
        "selected_components": selected,
        "combination": "sum of eight selected component derivative profiles",
        "length_normalized_sum": True,
        "path_selection_fit_on_train_fold": True,
    }


def _compact_weight_read(weight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mlp_primary": float(weight["mlp_primary"]),
        "attention_primary": float(weight["attention_primary"]),
        "equal_family_composite": float(weight["equal_family_composite"]),
        "selection_name": weight.get("selection_name"),
        "component_values": {
            str(row["component"]): float(row["normalized_gain"])
            for row in weight["mlps"]
        }
        | {
            str(row["component"]): float(row["label_weighted_normalized_ov"])
            for row in weight["attention_heads"]
        },
        "metadata": dict(weight["metadata"]),
    }


def score_under_all_fold_paths(
    bundle: Any,
    directions: Mapping[int, torch.Tensor],
    localization: Mapping[str, Any],
    fold_paths: Mapping[str | int, Mapping[str, Any]],
    component_profiles: Mapping[str, Any],
    carrying_positions: Sequence[int],
    *,
    row_id: str,
    concept_id: str,
    sequence_length: int | None = None,
    seed: int = SEED,
    weight_component_cache: MutableMapping[
        tuple[str, int, str], dict[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Compute exactly seven support scores under each of four train-fold paths."""

    if {int(key) for key in fold_paths} != set(range(N_FOLDS)):
        raise ValueError("Support scoring requires all four fold paths")
    if not str(concept_id):
        raise ValueError("Support scoring requires a stable concept_id RNG key")
    if sequence_length is None:
        local = _localization_lookup(localization)
        sequence_lengths = {
            len(record["score_by_position"]) for record in local.values()
        }
        if len(sequence_lengths) != 1:
            raise ValueError("Cannot infer one held-out sequence length")
        sequence_length = sequence_lengths.pop()
    if int(sequence_length) < 1:
        raise ValueError("Held-out sequence length must be positive")
    base_seed = int(
        seed + _stable_int(str(concept_id), seed=seed) % 1_000_000_000
    )
    first_path = fold_paths.get("0", fold_paths.get(0))
    if not isinstance(first_path, Mapping):
        raise ValueError("Missing fold path 0")
    r1_selection = first_path["R1"]
    for fold in range(1, N_FOLDS):
        other = fold_paths.get(str(fold), fold_paths.get(fold))
        if not isinstance(other, Mapping) or list(
            other["R1"].get("component_ids", [])
        ) != list(r1_selection.get("component_ids", [])):
            raise ValueError("Global R1 frontier must be identical across folds")
    r1_weight = repaired_weight_read(
        bundle,
        directions,
        r1_selection,
        seed=base_seed,
        component_cache=weight_component_cache,
        cache_namespace=str(concept_id),
    )
    output: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        path = fold_paths.get(str(fold), fold_paths.get(fold))
        if not isinstance(path, Mapping):
            raise ValueError(f"Missing fold path {fold}")
        r23_selection = path["R2_R3"]
        train_path_complete = bool(path.get("training_path_coverage_complete", True))
        if train_path_complete:
            r2_weight = repaired_weight_read(
                bundle,
                directions,
                r23_selection,
                seed=base_seed,
                component_cache=weight_component_cache,
                cache_namespace=str(concept_id),
            )
            r2 = r2_weight_profile(
                r23_selection,
                r2_weight,
                carrying_positions,
                sequence_length=int(sequence_length),
            )
            r3 = r3_profile_for_selection(
                component_profiles,
                r23_selection,
                carrying_positions,
            )
            r2_summary = r2["summaries"]
            r3_summary = r3["summaries"]
            r2_weight_record: dict[str, Any] | None = _compact_weight_read(r2_weight)
        else:
            unavailable_rows = list(path.get("unavailable_training_row_ids", []))
            r2_weight_record = None
            r2 = {
                "status": "UNDEFINED_INCOMPLETE_TRAIN_PATH_COVERAGE",
                "unavailable_training_row_ids": unavailable_rows,
                "profile": None,
            }
            r3 = dict(r2)
            r2_summary = {"sum": None, "peak": None, "carrying": None}
            r3_summary = {"sum": None, "peak": None, "carrying": None}
        scores = {
            "R1": float(r1_weight["equal_family_composite"]),
            "R2_sum": r2_summary.get("sum"),
            "R2_peak": r2_summary.get("peak"),
            "R2_carrying": r2_summary.get("carrying"),
            "R3_sum": r3_summary.get("sum"),
            "R3_peak": r3_summary.get("peak"),
            "R3_carrying": r3_summary.get("carrying"),
        }
        if tuple(scores) != CANDIDATE_NAMES:
            raise RuntimeError("Candidate set drifted from the frozen seven")
        output[str(fold)] = {
            "scores": scores,
            "R1": _compact_weight_read(r1_weight),
            "R2_weight": r2_weight_record,
            "R2_profile": r2,
            "R3_profile": r3,
            "path_component_ids": list(r23_selection["component_ids"]),
            "weight_seed": base_seed,
            "weight_rng_key": str(concept_id),
            "training_path_coverage_complete": train_path_complete,
        }
    return {
        "status": (
            "OK"
            if all(
                record["training_path_coverage_complete"] for record in output.values()
            )
            else "INCOMPLETE_TRAIN_PATH_COVERAGE"
        ),
        "row_id": str(row_id),
        "concept_id": str(concept_id),
        "scores_by_fold_path": output,
        "candidate_names": list(CANDIDATE_NAMES),
        "support_paths": N_FOLDS,
    }


def _finite_float(value: Any) -> float | None:
    """Return one finite numeric scalar, preserving every other value as missing."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nested_value(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _normalize_field_path(path: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(path, str):
        parts = tuple(part for part in path.split(".") if part)
    else:
        parts = tuple(str(part) for part in path)
    if not parts:
        raise ValueError("A ground-truth field path cannot be empty")
    return parts


def _causal_abs_value(
    row: Mapping[str, Any],
    arm: str,
    explicit_path: str | Sequence[str] | None,
) -> float | None:
    """Read one A/B causal magnitude without substituting or dropping failures."""

    arm = arm.upper()
    if arm not in {"A", "B"}:
        raise ValueError("Causal arm must be A or B")
    if explicit_path is not None:
        paths = (_normalize_field_path(explicit_path),)
    elif arm == "A":
        paths = (
            ("ground_truth_A", "causal_abs"),
            ("ground_truths", "A", "causal_abs"),
            ("coordinate_resampling_effect", "causal_abs"),
            ("A", "causal_abs"),
            ("A_causal_abs",),
            ("causal_A",),
        )
    else:
        paths = (
            ("ground_truth_B", "causal_abs"),
            ("ground_truths", "B", "causal_abs"),
            ("masked_source_to_foil_effect", "causal_abs"),
            ("B", "causal_abs"),
            ("B_causal_abs",),
            ("causal_B",),
        )
    found: list[float] = []
    for path in paths:
        value = _finite_float(_nested_value(row, path))
        if value is not None:
            found.append(value)
    if not found:
        return None
    if any(value < 0.0 for value in found):
        raise ValueError(f"Ground-truth {arm} causal_abs must be nonnegative")
    if any(
        not math.isclose(value, found[0], rel_tol=1e-9, abs_tol=1e-12)
        for value in found[1:]
    ):
        raise ValueError(f"Conflicting ground-truth {arm} causal_abs fields")
    return found[0]


def _effect_payload(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    arm = arm.upper()
    keys = (
        ("ground_truth_A", "coordinate_resampling_effect", "A")
        if arm == "A"
        else ("ground_truth_B", "masked_source_to_foil_effect", "B")
    )
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    nested = row.get("ground_truths")
    if isinstance(nested, Mapping) and isinstance(nested.get(arm), Mapping):
        return nested[arm]
    return {}


def _optional_int(value: Any) -> int | None:
    scalar = _finite_float(value)
    if scalar is None or not float(scalar).is_integer():
        return None
    return int(scalar)


def label_verification_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify the declared labels without constructing a verified-only subset.

    Engines pass only when the clean top token is the declared clean target and
    the B-edited top token is the declared counterfactual target.  Dashboards
    pass only when B has ``causal_abs <= 0.5``.  Clean capability is reported as
    a diagnostic and never enters the gate.
    """

    row_list = [dict(row) for row in rows]
    if not row_list:
        raise ValueError("Label verification requires validation rows")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in row_list:
        row_id = str(row["row_id"])
        if row_id in seen:
            raise ValueError(f"Duplicate label-verification row {row_id!r}")
        seen.add(row_id)
        label = int(row["label"])
        class_name = str(
            row.get("class_name", "engine" if label == 1 else "dashboard")
        )
        if (label, class_name) not in ((1, "engine"), (0, "dashboard")):
            raise ValueError(f"Row {row_id!r} has inconsistent class/label provenance")
        b_effect = _effect_payload(row, "B")
        clean_capability_reported = "clean_capable" in row
        clean_capable = (
            bool(row.get("clean_capable")) if clean_capability_reported else None
        )
        reasons: list[str] = []
        details: dict[str, Any]
        if class_name == "engine":
            a_effect = _effect_payload(row, "A")
            clean_top = _optional_int(
                b_effect.get(
                    "clean_top_token_id",
                    a_effect.get(
                        "clean_top_token_id", row.get("clean_top_token_id")
                    ),
                )
            )
            edited_top = _optional_int(
                b_effect.get(
                    "edited_top_token_id", row.get("B_edited_top_token_id")
                )
            )
            clean_target = _optional_int(row.get("clean_target_token_id"))
            counterfactual_target = _optional_int(
                row.get("counterfactual_target_token_id")
            )
            if clean_top is None or clean_target is None or clean_top != clean_target:
                reasons.append("CLEAN_TOP_NOT_DECLARED_TARGET")
            if (
                edited_top is None
                or counterfactual_target is None
                or edited_top != counterfactual_target
            ):
                reasons.append("B_EDITED_TOP_NOT_COUNTERFACTUAL_TARGET")
            details = {
                "clean_top_token_id": clean_top,
                "declared_clean_target_token_id": clean_target,
                "B_edited_top_token_id": edited_top,
                "declared_counterfactual_target_token_id": counterfactual_target,
            }
        else:
            b_causal_abs = _causal_abs_value(row, "B", None)
            if b_causal_abs is None:
                reasons.append("B_CAUSAL_ABS_UNDEFINED")
            elif b_causal_abs > 0.5:
                reasons.append("B_CAUSAL_ABS_ABOVE_0_5")
            details = {
                "B_causal_abs": b_causal_abs,
                "maximum_declared_dashboard_effect": 0.5,
            }
        records.append(
            {
                "row_id": row_id,
                "concept_id": str(row.get("concept_id", "")),
                "label": label,
                "class_name": class_name,
                "status": "PASS" if not reasons else "FAIL",
                "failure_reasons": reasons,
                "clean_capability_reported": clean_capability_reported,
                "clean_capable_diagnostic": clean_capable,
                **details,
            }
        )
    failures = [record for record in records if record["status"] != "PASS"]
    counts_by_class = {
        class_name: sum(record["class_name"] == class_name for record in records)
        for class_name in ("engine", "dashboard")
    }
    return {
        "schema_version": "read-validation-label-verification-v1",
        "status": "OK" if not failures else "FAILED_DECLARED_LABEL_COVERAGE",
        "complete_declared_label_coverage": not failures,
        "all_rows_retained": True,
        "verified_only_subset_created": False,
        "rules": READ_VALIDATION_PROTOCOL["label_verification"],
        "n_rows": len(records),
        "counts_by_class": counts_by_class,
        "expected_frozen_row_counts": {"engine": 155, "dashboard": 8},
        "n_failures": len(failures),
        "failure_row_ids": [record["row_id"] for record in failures],
        "clean_capability_diagnostic": {
            "n_reported": sum(
                record["clean_capability_reported"] for record in records
            ),
            "n_capable": sum(
                record["clean_capable_diagnostic"] is True for record in records
            ),
            "n_incapable": sum(
                record["clean_capable_diagnostic"] is False for record in records
            ),
            "used_as_go_gate": False,
        },
        "rows": records,
    }


def _scores_by_fold_path(row: Mapping[str, Any]) -> Mapping[str | int, Any]:
    payload = row.get("scores_by_fold_path")
    if payload is None:
        support = row.get("support_scores")
        if isinstance(support, Mapping):
            payload = support.get("scores_by_fold_path", support)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Row {row.get('row_id')!r} has no scores_by_fold_path payload"
        )
    return payload


def _fold_candidate_scores(
    row: Mapping[str, Any],
    fold: int,
) -> dict[str, float | None]:
    paths = _scores_by_fold_path(row)
    record = paths.get(str(fold), paths.get(fold))
    if not isinstance(record, Mapping):
        raise ValueError(f"Row {row.get('row_id')!r} is missing support path {fold}")
    scores = record.get("scores", record)
    if not isinstance(scores, Mapping):
        raise ValueError(f"Row {row.get('row_id')!r} path {fold} has no scores")
    return {candidate: _finite_float(scores.get(candidate)) for candidate in CANDIDATE_NAMES}


def _empirical_mid_cdf(value: float, sorted_support: np.ndarray) -> float:
    """Mid-distribution empirical CDF, deterministic in the presence of ties."""

    if sorted_support.ndim != 1 or not len(sorted_support):
        raise ValueError("Empirical CDF support must be a nonempty vector")
    lower = int(np.searchsorted(sorted_support, value, side="left"))
    upper = int(np.searchsorted(sorted_support, value, side="right"))
    return float((lower + 0.5 * (upper - lower)) / len(sorted_support))


def train_cdf_oof_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    a_causal_abs_path: str | Sequence[str] | None = None,
    b_causal_abs_path: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calibrate assigned-fold scores by label-blind training-concept CDFs.

    Every row is scored under every fold path before calling this function.  For
    held-out fold ``f``, support values under path ``f`` are averaged within each
    training concept, and those concept means define a pooled empirical mid-CDF.
    The assigned-path scores of fold-``f`` rows are then transformed by that CDF.
    Missing/nonfinite support is never dropped: it makes the affected calibration
    and downstream concept score undefined.
    """

    row_list = [dict(row) for row in rows]
    if not row_list:
        raise ValueError("Cross-fit calibration requires validation rows")
    label_verification = label_verification_report(row_list)
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    support: dict[str, dict[int, dict[str, float | None]]] = {}
    for row in row_list:
        row_id = str(row["row_id"])
        if row_id in by_id:
            raise ValueError(f"Duplicate validation row ID {row_id!r}")
        label = int(row["label"])
        fold = int(row["fold"])
        if label not in (0, 1) or not 0 <= fold < N_FOLDS:
            raise ValueError(f"Invalid label/fold for validation row {row_id!r}")
        concept = str(row["concept_id"])
        if not concept:
            raise ValueError(f"Validation row {row_id!r} has an empty concept ID")
        by_id[row_id] = row
        by_concept[concept].append(row)
        support[row_id] = {
            path_fold: _fold_candidate_scores(row, path_fold)
            for path_fold in range(N_FOLDS)
        }

    concept_fold: dict[str, int] = {}
    concept_label: dict[str, int] = {}
    for concept, members in by_concept.items():
        folds = {int(row["fold"]) for row in members}
        labels = {int(row["label"]) for row in members}
        if len(folds) != 1:
            raise ValueError(
                f"Concept {concept!r} crosses folds; fold groups must prevent leakage"
            )
        if len(labels) != 1:
            raise ValueError(f"Concept {concept!r} has inconsistent fixed labels")
        concept_fold[concept] = folds.pop()
        concept_label[concept] = labels.pop()

    calibration_by_fold: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        training_concepts = sorted(
            concept for concept in by_concept if concept_fold[concept] != fold
        )
        heldout_concepts = sorted(
            concept for concept in by_concept if concept_fold[concept] == fold
        )
        if not training_concepts or not heldout_concepts:
            raise ValueError(f"Fold {fold} lacks training or held-out concepts")
        candidate_records: dict[str, Any] = {}
        for candidate in CANDIDATE_NAMES:
            train_scores: dict[str, float | None] = {}
            missing_rows: dict[str, list[str]] = {}
            for concept in training_concepts:
                member_values: list[float] = []
                missing = []
                for row in by_concept[concept]:
                    row_id = str(row["row_id"])
                    value = support[row_id][fold][candidate]
                    if value is None:
                        missing.append(row_id)
                    else:
                        member_values.append(value)
                if missing:
                    train_scores[concept] = None
                    missing_rows[concept] = sorted(missing)
                else:
                    train_scores[concept] = float(np.mean(member_values))
            if missing_rows:
                status = "UNDEFINED_INCOMPLETE_TRAIN_SUPPORT"
                sorted_values: list[float] | None = None
            else:
                status = "OK"
                sorted_values = sorted(float(train_scores[c]) for c in training_concepts)
            candidate_records[candidate] = {
                "status": status,
                "training_concept_scores": train_scores,
                "sorted_training_concept_support": sorted_values,
                "missing_training_rows_by_concept": missing_rows,
                "n_training_concepts": len(training_concepts),
            }
        calibration_by_fold[str(fold)] = {
            "training_concept_ids": training_concepts,
            "heldout_concept_ids": heldout_concepts,
            "candidates": candidate_records,
        }

    calibrated_rows: list[dict[str, Any]] = []
    for row_id in sorted(by_id):
        row = by_id[row_id]
        fold = int(row["fold"])
        raw_scores = support[row_id][fold]
        calibrated: dict[str, float | None] = {}
        statuses: dict[str, str] = {}
        for candidate in CANDIDATE_NAMES:
            raw = raw_scores[candidate]
            fit = calibration_by_fold[str(fold)]["candidates"][candidate]
            if raw is None:
                calibrated[candidate] = None
                statuses[candidate] = "UNDEFINED_HELDOUT_SCORE"
            elif fit["status"] != "OK":
                calibrated[candidate] = None
                statuses[candidate] = str(fit["status"])
            else:
                calibration_support = np.asarray(
                    fit["sorted_training_concept_support"], dtype=float
                )
                calibrated[candidate] = _empirical_mid_cdf(raw, calibration_support)
                statuses[candidate] = "OK"
        calibrated_rows.append(
            {
                "row_id": row_id,
                "concept_id": str(row["concept_id"]),
                "label": int(row["label"]),
                "fold": fold,
                "assigned_fold_path": fold,
                "raw_assigned_path_scores": raw_scores,
                "oof_train_cdf_scores": calibrated,
                "candidate_status": statuses,
                "A_causal_abs": _causal_abs_value(
                    row, "A", a_causal_abs_path
                ),
                "B_causal_abs": _causal_abs_value(
                    row, "B", b_causal_abs_path
                ),
            }
        )
    calibrated_by_id = {row["row_id"]: row for row in calibrated_rows}

    concept_rows: list[dict[str, Any]] = []
    for concept in sorted(by_concept):
        members = sorted(by_concept[concept], key=lambda row: str(row["row_id"]))
        calibrated_members = [calibrated_by_id[str(row["row_id"])] for row in members]
        candidate_scores: dict[str, float | None] = {}
        candidate_missing_rows: dict[str, list[str]] = {}
        for candidate in CANDIDATE_NAMES:
            values = [row["oof_train_cdf_scores"][candidate] for row in calibrated_members]
            missing = [
                row["row_id"]
                for row, value in zip(calibrated_members, values, strict=True)
                if value is None
            ]
            if missing:
                candidate_scores[candidate] = None
                candidate_missing_rows[candidate] = missing
            else:
                candidate_scores[candidate] = float(np.mean(values))
        a_values = [row["A_causal_abs"] for row in calibrated_members]
        b_values = [row["B_causal_abs"] for row in calibrated_members]
        a_value = (
            float(np.mean(a_values)) if all(value is not None for value in a_values) else None
        )
        b_value = (
            float(np.mean(b_values)) if all(value is not None for value in b_values) else None
        )
        fold_groups = sorted(
            {str(row["fold_group"]) for row in members if row.get("fold_group") is not None}
        )
        if len(fold_groups) != 1:
            raise ValueError(
                f"Concept {concept!r} must have exactly one dependency cluster"
            )
        pair_ids = sorted(
            {
                str(pair)
                for row in members
                for pair in (row.get("pair_id"), row.get("bootstrap_pair_id"))
                if pair is not None
            }
        )
        class_names = {
            str(row.get("class_name")) for row in members if row.get("class_name")
        }
        if len(class_names) > 1:
            raise ValueError(f"Concept {concept!r} has inconsistent class names")
        concept_rows.append(
            {
                "concept_id": concept,
                "label": concept_label[concept],
                "class_name": (
                    next(iter(class_names))
                    if class_names
                    else ("engine" if concept_label[concept] == 1 else "dashboard")
                ),
                "fold": concept_fold[concept],
                "fold_group": fold_groups[0] if fold_groups else None,
                "cluster_id": fold_groups[0],
                "bootstrap_cluster_id": fold_groups[0],
                "bootstrap_pair_ids": pair_ids,
                "row_ids": [str(row["row_id"]) for row in members],
                "n_rows": len(members),
                "candidate_scores": candidate_scores,
                "candidate_missing_row_ids": candidate_missing_rows,
                "ground_truths": {
                    "A_coordinate_resampling_primary": a_value,
                    "B_masked_source_to_foil_clamped_swap_secondary": b_value,
                },
                "A_causal_abs": a_value,
                "B_causal_abs": b_value,
            }
        )

    coverage = {
        candidate: {
            "n_concepts": len(concept_rows),
            "n_defined": sum(
                row["candidate_scores"][candidate] is not None for row in concept_rows
            ),
            "missing_concept_ids": [
                row["concept_id"]
                for row in concept_rows
                if row["candidate_scores"][candidate] is None
            ],
        }
        for candidate in CANDIDATE_NAMES
    }
    for record in coverage.values():
        record["complete"] = record["n_defined"] == record["n_concepts"]
    ground_truth_coverage = {
        arm: {
            "n_concepts": len(concept_rows),
            "n_defined": sum(row[key] is not None for row in concept_rows),
            "missing_concept_ids": [
                row["concept_id"] for row in concept_rows if row[key] is None
            ],
        }
        for arm, key in (("A", "A_causal_abs"), ("B", "B_causal_abs"))
    }
    for record in ground_truth_coverage.values():
        record["complete"] = record["n_defined"] == record["n_concepts"]
    complete = all(record["complete"] for record in coverage.values()) and all(
        record["complete"] for record in ground_truth_coverage.values()
    )
    concept_counts_by_class = {
        class_name: sum(row["class_name"] == class_name for row in concept_rows)
        for class_name in ("engine", "dashboard")
    }
    cluster_counts_by_class = {
        class_name: len(
            {
                row["cluster_id"]
                for row in concept_rows
                if row["class_name"] == class_name
            }
        )
        for class_name in ("engine", "dashboard")
    }
    return {
        "schema_version": "read-validation-oof-cdf-v1",
        "status": "OK" if complete else "INCOMPLETE_COVERAGE",
        "candidate_names": list(CANDIDATE_NAMES),
        "n_rows": len(row_list),
        "n_concepts": len(concept_rows),
        "concept_counts_by_class": concept_counts_by_class,
        "expected_frozen_concept_counts": {"engine": 75, "dashboard": 4},
        "cluster_counts_by_class": cluster_counts_by_class,
        "method": (
            "assigned-fold raw score transformed by pooled training-concept "
            "empirical mid-CDF; held-out rows then mean-aggregated by concept"
        ),
        "labels_read_during_calibration": False,
        "missing_values_dropped": False,
        "calibration_by_fold": calibration_by_fold,
        "row_scores": calibrated_rows,
        "concept_rows": concept_rows,
        "candidate_coverage": coverage,
        "ground_truth_coverage": ground_truth_coverage,
        "label_verification": label_verification,
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("AUC labels and scores must be aligned vectors")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("AUC requires both fixed provenance labels")
    if not np.isfinite(scores).all():
        raise ValueError("AUC scores must be finite")
    return float(roc_auc_score(labels, scores))


def _pairwise_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Fast exact binary AUC for the repeated bootstrap calculation."""

    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Bootstrap AUC requires both strata")
    differences = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)) / differences.size)


def stratified_cluster_auc_ci(
    concept_rows: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    """Pooled OOF AUC with the frozen stratified dependency-cluster bootstrap."""

    if candidate not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown READ candidate {candidate!r}")
    if int(n_bootstrap) != N_BOOTSTRAP or int(seed) != SEED:
        raise ValueError(
            f"Frozen bootstrap requires n={N_BOOTSTRAP}, seed={SEED}"
        )
    rows = [dict(row) for row in concept_rows]
    if not rows:
        raise ValueError("AUC requires concept-aggregated OOF rows")
    concept_ids = [str(row["concept_id"]) for row in rows]
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("AUC input must contain exactly one row per concept")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    if any(label not in (0, 1) for label in labels):
        raise ValueError("AUC labels must be fixed zeros or ones")
    values = [
        _finite_float(
            row.get("candidate_scores", {}).get(candidate)
            if isinstance(row.get("candidate_scores"), Mapping)
            else None
        )
        for row in rows
    ]
    missing = [
        concept
        for concept, value in zip(concept_ids, values, strict=True)
        if value is None
    ]
    cluster_ids = [
        str(row.get("cluster_id", row.get("bootstrap_cluster_id", "")))
        for row in rows
    ]
    if any(not cluster for cluster in cluster_ids):
        raise ValueError("Every concept must retain one dependency cluster_id")
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        cluster_indices[cluster].append(index)
    clusters_by_label: dict[int, list[str]] = {0: [], 1: []}
    for cluster in sorted(cluster_indices):
        cluster_labels = {int(labels[index]) for index in cluster_indices[cluster]}
        if len(cluster_labels) != 1:
            raise ValueError(f"Bootstrap cluster {cluster!r} crosses label strata")
        clusters_by_label[cluster_labels.pop()].append(cluster)
    base = {
        "candidate": candidate,
        "n_concepts": len(rows),
        "n_positive_concepts": int(np.count_nonzero(labels == 1)),
        "n_negative_concepts": int(np.count_nonzero(labels == 0)),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "bootstrap_unit": (
            "dependency cluster: engine source/foil connected component or "
            "dashboard language"
        ),
        "bootstrap_stratification": "fixed provenance label",
        "n_dependency_clusters": len(cluster_indices),
        "cluster_counts_by_label": {
            str(label): len(clusters_by_label[label]) for label in (0, 1)
        },
        "engine_pair_ids_retained_in_concept_metadata": True,
        "missing_concept_ids": missing,
        "complete_concept_coverage": not missing,
    }
    if missing:
        return {
            **base,
            "status": "UNDEFINED_INCOMPLETE_CONCEPT_COVERAGE",
            "auc": None,
            "ci95_low": None,
            "ci95_high": None,
            "passes_go_threshold": False,
        }
    if set(labels.tolist()) != {0, 1}:
        return {
            **base,
            "status": "UNDEFINED_SINGLE_LABEL",
            "auc": None,
            "ci95_low": None,
            "ci95_high": None,
            "passes_go_threshold": False,
        }
    scores = np.asarray(values, dtype=float)
    auc = _binary_auc(labels, scores)
    if any(not clusters_by_label[label] for label in (0, 1)):
        raise ValueError("Stratified bootstrap requires clusters in both labels")
    rng = np.random.default_rng(SEED)
    bootstrap_aucs = np.empty(N_BOOTSTRAP, dtype=float)
    for draw in range(N_BOOTSTRAP):
        indices: list[int] = []
        for label in (0, 1):
            clusters = clusters_by_label[label]
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            for cluster in sampled:
                indices.extend(cluster_indices[str(cluster)])
        selected = np.asarray(indices, dtype=int)
        bootstrap_aucs[draw] = _pairwise_auc(labels[selected], scores[selected])
    ci_low, ci_high = np.quantile(bootstrap_aucs, [0.025, 0.975])
    passes = bool(auc >= 0.70 and float(ci_low) > 0.5)
    return {
        **base,
        "status": "OK",
        "auc": auc,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "ci_method": "two-sided 95% percentile interval",
        "auc_threshold": 0.70,
        "ci_low_threshold_strictly_above": 0.5,
        "passes_go_threshold": passes,
    }


def pooled_oof_auc_table(
    concept_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute seven fixed-label AUCs and duplicate their A/B report rows."""

    results = {
        candidate: stratified_cluster_auc_ci(concept_rows, candidate)
        for candidate in CANDIDATE_NAMES
    }
    identity_note = READ_VALIDATION_PROTOCOL["decision"]["auc_ground_truth_note"]
    duplicated_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for ground_truth in (
            "A_coordinate_resampling_primary",
            "B_masked_source_to_foil_clamped_swap_secondary",
        ):
            duplicated_rows.append(
                {
                    **results[candidate],
                    "ground_truth_report_row": ground_truth,
                    "auc_identity_across_A_and_B": True,
                    "identity_note": identity_note,
                }
            )
    return {
        "schema_version": "read-validation-pooled-oof-auc-v1",
        "status": (
            "OK"
            if all(result["status"] == "OK" for result in results.values())
            else "INCOMPLETE_CANDIDATE_COVERAGE"
        ),
        "candidate_results": results,
        "ground_truth_rows": duplicated_rows,
        "fixed_label_auc_identity_note": identity_note,
        "ground_truth_values_used_in_auc": False,
        "inferential_look_count": 1,
        "A_B_rows_are_presentation_duplicates_not_extra_chances": True,
    }


def _strict_spearman(
    left: Sequence[float | None],
    right: Sequence[float | None],
    concept_ids: Sequence[str],
) -> dict[str, Any]:
    missing = [
        concept
        for concept, x_value, y_value in zip(
            concept_ids, left, right, strict=True
        )
        if x_value is None or y_value is None
    ]
    if missing:
        return {
            "status": "UNDEFINED_INCOMPLETE_CONCEPT_COVERAGE",
            "n_concepts": len(concept_ids),
            "missing_concept_ids": missing,
            "rho": None,
            "p_value": None,
        }
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {
            "status": "UNDEFINED_CONSTANT_OR_TOO_FEW_VALUES",
            "n_concepts": len(concept_ids),
            "missing_concept_ids": [],
            "rho": None,
            "p_value": None,
        }
    result = spearmanr(x, y)
    rho = _finite_float(result.statistic)
    p_value = _finite_float(result.pvalue)
    return {
        "status": "OK" if rho is not None and p_value is not None else "UNDEFINED",
        "n_concepts": len(concept_ids),
        "missing_concept_ids": [],
        "rho": rho,
        "p_value": p_value,
    }


def _strict_pearson(
    left: Sequence[float | None],
    right: Sequence[float | None],
    concept_ids: Sequence[str],
) -> dict[str, Any]:
    missing = [
        concept
        for concept, x_value, y_value in zip(
            concept_ids, left, right, strict=True
        )
        if x_value is None or y_value is None
    ]
    if missing:
        return {
            "status": "UNDEFINED_INCOMPLETE_CONCEPT_COVERAGE",
            "n_concepts": len(concept_ids),
            "missing_concept_ids": missing,
            "r": None,
        }
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {
            "status": "UNDEFINED_CONSTANT_OR_TOO_FEW_VALUES",
            "n_concepts": len(concept_ids),
            "missing_concept_ids": [],
            "r": None,
        }
    value = _finite_float(np.corrcoef(x, y)[0, 1])
    return {
        "status": "OK" if value is not None else "UNDEFINED",
        "n_concepts": len(concept_ids),
        "missing_concept_ids": [],
        "r": value,
    }


def secondary_correlation_report(
    concept_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report pooled concept-level Spearman vs A/B and direct A/B correlation."""

    rows = [dict(row) for row in concept_rows]
    if not rows:
        raise ValueError("Correlation report requires concept rows")
    concept_ids = [str(row["concept_id"]) for row in rows]
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("Correlation input must contain one row per concept")
    a_values = [_finite_float(row.get("A_causal_abs")) for row in rows]
    b_values = [_finite_float(row.get("B_causal_abs")) for row in rows]
    candidate_results: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        candidate_values = [
            _finite_float(
                row.get("candidate_scores", {}).get(candidate)
                if isinstance(row.get("candidate_scores"), Mapping)
                else None
            )
            for row in rows
        ]
        candidate_results[candidate] = {
            "vs_A_coordinate_resampling": _strict_spearman(
                candidate_values, a_values, concept_ids
            ),
            "vs_B_masked_source_to_foil": _strict_spearman(
                candidate_values, b_values, concept_ids
            ),
        }
    a_b_spearman = _strict_spearman(a_values, b_values, concept_ids)
    a_b_pearson = _strict_pearson(a_values, b_values, concept_ids)
    statuses = [
        result[arm]["status"]
        for result in candidate_results.values()
        for arm in ("vs_A_coordinate_resampling", "vs_B_masked_source_to_foil")
    ]
    statuses.extend((a_b_spearman["status"], a_b_pearson["status"]))
    return {
        "schema_version": "read-validation-correlations-v1",
        "status": "OK" if all(status == "OK" for status in statuses) else "INCOMPLETE_OR_UNDEFINED",
        "aggregation": "pooled OOF concept rows; no pairwise deletion",
        "candidate_spearman": candidate_results,
        "A_B_correlation": {
            "spearman": a_b_spearman,
            "pearson": a_b_pearson,
        },
    }


def read_go_no_go_decision(
    auc_table: Mapping[str, Any],
    label_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the frozen AUC/CI gate; this is not a science conclusion."""

    results = auc_table.get("candidate_results")
    if not isinstance(results, Mapping):
        raise ValueError("Decision requires pooled_oof_auc_table output")
    missing = [candidate for candidate in CANDIDATE_NAMES if candidate not in results]
    if missing:
        raise ValueError("AUC table is missing frozen candidates: " + ", ".join(missing))
    statistically_passing = [
        candidate
        for candidate in CANDIDATE_NAMES
        if results[candidate].get("status") == "OK"
        and results[candidate].get("complete_concept_coverage") is True
        and results[candidate].get("passes_go_threshold") is True
    ]
    label_coverage_complete = bool(
        label_verification.get("complete_declared_label_coverage") is True
    )
    passing = statistically_passing if label_coverage_complete else []
    if passing:
        winner = max(
            passing,
            key=lambda candidate: (
                float(results[candidate]["auc"]),
                float(results[candidate]["ci95_low"]),
                -CANDIDATE_NAMES.index(candidate),
            ),
        )
        decision = "GO"
        one_line = (
            f"READ GO: {winner} reaches pooled OOF AUC "
            f"{results[winner]['auc']:.3f} with 95% CI lower bound "
            f"{results[winner]['ci95_low']:.3f}."
        )
    elif statistically_passing:
        winner = None
        decision = "NO-GO"
        one_line = (
            "READ NO-GO: the AUC/CI gate is met, but declared engine/dashboard "
            "label verification is incomplete and failures were retained."
        )
    else:
        winner = None
        decision = "NO-GO"
        one_line = (
            "READ NO-GO: no complete-coverage candidate reaches AUC >= 0.70 "
            "with the 95% bootstrap CI lower bound strictly above 0.50."
        )
    return {
        "decision": decision,
        "one_line": one_line,
        "winning_candidate": winner,
        "passing_candidates": passing,
        "statistically_passing_candidates": statistically_passing,
        "complete_declared_label_coverage": label_coverage_complete,
        "gate": {
            "auc_at_least": 0.70,
            "ci95_low_strictly_above": 0.5,
            "complete_concept_coverage_required": True,
            "complete_declared_label_verification_required": True,
        },
        "inferential_look_count": 1,
        "A_B_presentation_duplicates_create_no_additional_chance": True,
        "scope_note": (
            "method-validation Go/No-Go only; not a Written-vs-Read or causal "
            "science conclusion"
        ),
    }


def build_model_free_validation_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    a_causal_abs_path: str | Sequence[str] | None = None,
    b_causal_abs_path: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the auditable, JSON-ready model-free end of the validation."""

    calibration = train_cdf_oof_calibration(
        rows,
        a_causal_abs_path=a_causal_abs_path,
        b_causal_abs_path=b_causal_abs_path,
    )
    auc = pooled_oof_auc_table(calibration["concept_rows"])
    correlations = secondary_correlation_report(calibration["concept_rows"])
    decision = read_go_no_go_decision(auc, calibration["label_verification"])
    return {
        "schema_version": "read-go-no-go-report-v1",
        "protocol_sha256": READ_VALIDATION_PROTOCOL_SHA256,
        "protocol": READ_VALIDATION_PROTOCOL,
        "calibration": calibration,
        "pooled_oof_auc": auc,
        "secondary_correlations": correlations,
        "label_verification": calibration["label_verification"],
        "go_no_go": decision,
    }

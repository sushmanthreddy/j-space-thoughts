"""Hard-dashboard construction and clean verification for the isolated v6 run.

The frozen engine roster has categorical outputs rather than numeric outputs:
element symbols and capital-city names.  These controls therefore use a fixed
calibration-only fact with the *same semantic relation and answer class* as
each engine category.  The source concept remains in an exact byte-identical
context prefix but cannot determine the fixed anchor answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


HARD_DASHBOARD_TEMPLATES: dict[str, dict[str, str]] = {
    "atomic-number-element-symbol": {
        "template_id": "fixed-calibration-platinum-symbol",
        "suffix": " For comparison, the chemical symbol of platinum is",
        "target": "Pt",
        "target_surface": " Pt",
        "distractor": "Sn",
        "distractor_surface": " Sn",
        "answer_type": "chemical element symbol",
        "relation": "element-to-chemical-symbol",
        "anchor_dependency_group": "atomic-number-element-symbol:platinum<->tin",
        "irrelevance": "The source element cannot determine platinum's symbol.",
    },
    "city-country-capital": {
        "template_id": "fixed-calibration-netherlands-capital",
        "suffix": " For comparison, the capital of the Netherlands is",
        "target": "Amsterdam",
        "target_surface": " Amsterdam",
        "distractor": "Lima",
        "distractor_surface": " Lima",
        "answer_type": "national capital city",
        "relation": "country-to-national-capital",
        "anchor_dependency_group": "city-country-capital:netherlands<->peru",
        "irrelevance": "The source country cannot determine the Netherlands' capital.",
    },
    "us-city-state-capital": {
        "template_id": "fixed-calibration-alabama-capital",
        "suffix": " For comparison, the capital of the US state of Alabama is",
        "target": "Montgomery",
        "target_surface": " Montgomery",
        "distractor": "Atlanta",
        "distractor_surface": " Atlanta",
        "answer_type": "US state capital city",
        "relation": "US-state-to-state-capital",
        "anchor_dependency_group": "us-city-state-capital:alabama<->georgia",
        "irrelevance": "The source state cannot determine Alabama's capital.",
    },
}


def _single_token_id(tokenizer: Any, surface: str) -> int:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected one token for {surface!r}, got {token_ids}")
    return int(token_ids[0])


def _assert_calibration_anchors(rows: Sequence[Mapping[str, Any]]) -> None:
    calibration_groups = {
        str(row["dependency_group"])
        for row in rows
        if row.get("split") == "calibration"
    }
    missing = {
        template["anchor_dependency_group"]
        for template in HARD_DASHBOARD_TEMPLATES.values()
    } - calibration_groups
    if missing:
        raise ValueError(f"Hard-dashboard calibration anchors are absent: {sorted(missing)}")


def build_hard_dashboard_candidates(
    source_rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    """Build fixed-template controls from the frozen VERIFIED source roster.

    Every hard prompt begins with the persisted natural context byte-for-byte.
    Token-level checks prove that the explicit concept and its intervention
    position are unchanged.  No causal or edited quantity is accepted here.
    """

    _assert_calibration_anchors(source_rows)
    verified = [
        row for row in source_rows if row.get("verification_status") == "VERIFIED"
    ]
    if not verified:
        raise ValueError("No frozen VERIFIED pairs are available")

    candidates: list[dict[str, Any]] = []
    for source in verified:
        category = str(source["category"])
        if category not in HARD_DASHBOARD_TEMPLATES:
            raise ValueError(f"No frozen hard-dashboard template for {category!r}")
        template = HARD_DASHBOARD_TEMPLATES[category]
        target_id = _single_token_id(tokenizer, template["target_surface"])
        distractor_id = _single_token_id(tokenizer, template["distractor_surface"])
        if target_id == distractor_id:
            raise ValueError("Hard-dashboard target and distractor must differ")

        hard_prompts: dict[str, str] = {}
        n_tokens: dict[str, int] = {}
        prefix_audits: dict[str, dict[str, Any]] = {}
        for side in ("a", "b"):
            context = str(source[f"context_{side}"])
            prompt = context + template["suffix"]
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            concept_prefix_ids = tokenizer.encode(
                str(source[f"concept_prefix_{side}"]), add_special_tokens=False
            )
            position = int(source[f"intervention_position_{side}"])
            concept_token_id = int(source[f"concept_{side}_token_id"])
            context_prefix_preserved = prompt_ids[: len(context_ids)] == context_ids
            concept_prefix_preserved = (
                prompt_ids[: len(concept_prefix_ids)] == concept_prefix_ids
            )
            concept_token_preserved = (
                0 <= position < len(prompt_ids)
                and int(prompt_ids[position]) == concept_token_id
            )
            if not (
                context_prefix_preserved
                and concept_prefix_preserved
                and concept_token_preserved
            ):
                raise ValueError(
                    f"Hard prompt changed frozen prefix/position for {source['pair_id']} {side}"
                )
            hard_prompts[side] = prompt
            n_tokens[side] = len(prompt_ids)
            prefix_audits[side] = {
                "context_prefix_preserved": context_prefix_preserved,
                "concept_prefix_preserved": concept_prefix_preserved,
                "concept_token_preserved": concept_token_preserved,
                "intervention_position": position,
                "concept_token_id": concept_token_id,
                "context_n_tokens": len(context_ids),
                "hard_prompt_n_tokens": len(prompt_ids),
            }

        candidates.append(
            {
                "pair_id": str(source["pair_id"]),
                "dependency_group": str(source["dependency_group"]),
                "fold": int(source["fold"]),
                "category": category,
                "concept_a": str(source["concept_a"]),
                "concept_b": str(source["concept_b"]),
                "concept_a_token_id": int(source["concept_a_token_id"]),
                "concept_b_token_id": int(source["concept_b_token_id"]),
                "concept_a_surface": str(source["concept_a_surface"]),
                "concept_b_surface": str(source["concept_b_surface"]),
                "intervention_position_a": int(source["intervention_position_a"]),
                "intervention_position_b": int(source["intervention_position_b"]),
                "hard_prompt_a": hard_prompts["a"],
                "hard_prompt_b": hard_prompts["b"],
                "hard_prompt_n_tokens_a": n_tokens["a"],
                "hard_prompt_n_tokens_b": n_tokens["b"],
                "hard_target": template["target"],
                "hard_target_surface": template["target_surface"],
                "hard_target_token_id": target_id,
                "hard_distractor": template["distractor"],
                "hard_distractor_surface": template["distractor_surface"],
                "hard_distractor_token_id": distractor_id,
                "hard_answer_type": template["answer_type"],
                "engine_answer_type_matched": True,
                "hard_relation": template["relation"],
                "hard_template_id": template["template_id"],
                "anchor_dependency_group": template["anchor_dependency_group"],
                "anchor_selected_from_calibration_only": True,
                "concept_irrelevance_contract": template["irrelevance"],
                "prefix_audit_a": prefix_audits["a"],
                "prefix_audit_b": prefix_audits["b"],
                "hard_z_a": float(source["engine_z_a"]),
                "hard_z_b": float(source["engine_z_b"]),
                "written_threshold": float(source["written_threshold"]),
                "written_provenance": (
                    "Frozen engine z reused because the hard prompt preserves the exact "
                    "causal prefix through the explicit concept token."
                ),
            }
        )
    return candidates


@torch.no_grad()
def verify_hard_dashboard_candidates(
    hf_model: torch.nn.Module,
    tokenizer: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Apply the frozen correctness + WRITTEN gate to hard dashboards."""

    if hf_model.training:
        raise ValueError("Hard-dashboard verification requires eval mode")
    if batch_size < 1 or top_k < 1:
        raise ValueError("batch_size and top_k must be positive")
    flattened: list[tuple[int, str, str, int]] = []
    for index, row in enumerate(candidates):
        for side in ("a", "b"):
            flattened.append(
                (
                    index,
                    side,
                    str(row[f"hard_prompt_{side}"]),
                    int(row["hard_target_token_id"]),
                )
            )

    device = next(hf_model.parameters()).device
    clean_records: dict[tuple[int, str], dict[str, Any]] = {}
    for start in range(0, len(flattened), batch_size):
        batch = flattened[start : start + batch_size]
        prompts = [entry[2] for entry in batch]
        tokenizer.padding_side = "right"
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        for batch_index, (row_index, side, _prompt, expected_id) in enumerate(batch):
            positions = attention_mask[batch_index].nonzero(as_tuple=False).flatten()
            final_position = int(positions[-1])
            next_logits = logits[batch_index, final_position]
            top_values, top_ids = next_logits.topk(top_k)
            top_token_id = int(top_ids[0].cpu())
            expected_rank = int(
                (next_logits > next_logits[int(expected_id)]).sum().cpu().item() + 1
            )
            clean_records[(row_index, side)] = {
                "hard_top1": top_token_id == int(expected_id),
                "hard_target_rank": expected_rank,
                "hard_top_token_id": top_token_id,
                "hard_top_token": tokenizer.decode([top_token_id]),
                "hard_top_tokens": [
                    {
                        "token_id": int(token_id),
                        "token": tokenizer.decode([int(token_id)]),
                        "logit": float(value),
                    }
                    for value, token_id in zip(
                        top_values.detach().cpu(), top_ids.detach().cpu(), strict=True
                    )
                ],
            }
        del logits, input_ids, attention_mask

    verified_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        row = dict(candidate)
        reasons: list[str] = []
        for side in ("a", "b"):
            row.update(
                {f"{key}_{side}": value for key, value in clean_records[(index, side)].items()}
            )
            written = float(row[f"hard_z_{side}"]) >= float(row["written_threshold"])
            row[f"hard_written_{side}"] = written
            if not row[f"hard_top1_{side}"]:
                reasons.append(f"HARD_{side.upper()}_TARGET_NOT_TOP1")
            if not written:
                reasons.append(f"HARD_{side.upper()}_CONCEPT_NOT_WRITTEN")
        row["verification_reasons"] = reasons
        row["verification_status"] = (
            "VERIFIED_HARD" if not reasons else "UNVERIFIED_HARD"
        )
        verified_rows.append(row)
    return verified_rows

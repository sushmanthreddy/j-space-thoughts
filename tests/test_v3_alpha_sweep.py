import copy
import hashlib
import json

import pytest
import torch

from src.v3_alpha_sweep import (
    ALPHA_GRID,
    POLICY_ORDER,
    RANDOM_DRAWS,
    _validate_alpha_sweep,
    project_out_transfer,
    select_carrying_positions,
)


def test_project_out_transfer_caps_source_and_preserves_orthogonal_component():
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    clean = torch.tensor([[[2.0, 0.5, 7.0], [1.0, 3.0, 9.0]]])
    edited = project_out_transfer(
        clean,
        clean,
        source,
        target,
        positions=[0],
        strength=1.0,
    )
    assert torch.allclose(edited[0, 0], torch.tensor([0.0, 2.0, 7.0]))
    assert torch.equal(edited[0, 1], clean[0, 1])
    over = project_out_transfer(
        clean,
        clean,
        source,
        target,
        positions=[0],
        strength=2.0,
    )
    assert float(over[0, 0, 0]) == 0.0
    assert float(over[0, 0, 1]) == 3.5
    assert float(over[0, 0, 2]) == 7.0


def test_select_carrying_positions_is_clean_rank_union():
    # Token 1 ranks <=2 at positions 0 (layer 3) and 2 (layer 4).
    readout = {
        3: torch.tensor([[3.0, 4.0, 2.0], [5.0, 0.0, 4.0], [3.0, 1.0, 2.0]]),
        4: torch.tensor([[4.0, 0.0, 3.0], [5.0, 0.0, 4.0], [2.0, 3.0, 1.0]]),
    }
    result = select_carrying_positions(readout, 1, [3, 4], rank_threshold=2)
    assert result["positions"] == [0, 2]
    assert len(result["sha256"]) == 64


def _valid_fail_sweep(raw_path):
    raw_path.write_text("raw", encoding="utf-8")
    manifest = {"rule": "test", "known": {}, "capability": {}, "g_pos": {}}
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for policy in POLICY_ORDER:
        for alpha in ALPHA_GRID:
            gates = {
                "swap_3_of_3": False,
                "capability": True,
                "g_pos": False,
                "random_null": True,
                "absent_null": True,
                "controls_fire": True,
            }
            rows.append(
                {
                    "policy": policy,
                    "alpha": alpha,
                    "selectable": policy == "project_out_transfer",
                    "known_swaps": {"status": "FAIL"},
                    "capability": {
                        "status": "PASS",
                        "mean_delta_nll": 0.0,
                        "mean_abs_delta_nll": 0.0,
                        "per_intervention": [
                            {
                                "mean_delta_nll": 0.0,
                                "mean_abs_delta_nll": 0.0,
                            }
                        ],
                    },
                    "g_pos": {"status": "FAIL"},
                    "random_null": {
                        "status": "PASS",
                        "n_draws_per_item": RANDOM_DRAWS,
                        "rows": [{}, {}, {}],
                    },
                    "absent_null": {"status": "PASS", "rows": [{}, {}, {}]},
                    "gates": gates,
                    "valid": False,
                }
            )
    return {
        "status": "FAIL",
        "g_alpha": "FAIL",
        "alpha_grid": list(ALPHA_GRID),
        "policy_order": list(POLICY_ORDER),
        "selectable_policies": ["project_out_transfer"],
        "firing_controls": "PASS",
        "mask_manifest": manifest,
        "rows": rows,
        "selected_intervention": None,
        "raw_artifact": str(raw_path),
        "raw_artifact_bytes": raw_path.stat().st_size,
        "raw_artifact_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }


def test_validate_alpha_sweep_enforces_full_null_coverage_and_mean_absolute(tmp_path):
    sweep = _valid_fail_sweep(tmp_path / "raw.json")
    _validate_alpha_sweep(sweep)

    missing_null = copy.deepcopy(sweep)
    missing_null["rows"][0]["random_null"] = {
        "status": "NOT_EVALUATED_GSWAP_PREREQUISITE",
        "n_draws_per_item": 0,
        "rows": [],
    }
    missing_null["rows"][0]["gates"]["random_null"] = False
    with pytest.raises(RuntimeError, match="null coverage"):
        _validate_alpha_sweep(missing_null)

    hidden_mean_absolute_failure = copy.deepcopy(sweep)
    capability = hidden_mean_absolute_failure["rows"][0]["capability"]
    capability["mean_abs_delta_nll"] = 0.3
    with pytest.raises(RuntimeError, match="gate values"):
        _validate_alpha_sweep(hidden_mean_absolute_failure)

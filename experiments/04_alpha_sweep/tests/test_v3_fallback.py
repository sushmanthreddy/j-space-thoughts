from src.v3_fallback import build_stage4_payload


def test_stage4_payload_preserves_claim_boundary_and_structural_zero():
    gpos_rows = [
        {
            "key": f"x{index}",
            "primary_weight_read_ratio": 0.8 + index / 100,
            "checks": {
                "low_causal_abs_delta": True,
                "low_primary_weight_read_ratio": False,
            },
        }
        for index in range(8)
    ]
    rows = [
        {
            "policy": "project_out_transfer",
            "alpha": 1.0,
            "known_swaps": {"n_pass": 2},
        },
        {
            "policy": "fractional_swap_carrying_positions",
            "alpha": 1.5,
            "known_swaps": {"n_pass": 3},
            "g_pos": {"n_reproduced": 0, "rows": gpos_rows},
            "random_null": {"status": "PASS"},
            "absent_null": {"status": "PASS"},
        },
        {
            "policy": "fractional_swap_all_positions_reference",
            "alpha": 2.0,
            "capability": {
                "mean_delta_nll": 0.623,
                "mean_abs_delta_nll": 0.669,
            },
        },
    ]
    legacy = {
        "jlens": {"pearson_r": 0.608},
        "identity_j_logit_lens": {"pearson_r": 0.639},
        "n": 155,
        "provenance_commit": "legacy",
    }
    metrics = {
        "calibration_v3": {
            "stage1_5_alpha_sweep": {
                "rows": rows,
                "mask_manifest": {
                    "capability": {
                        "a": {"positions": []},
                        "b": {"positions": []},
                    }
                },
                "raw_artifact": "data/raw.json",
                "raw_artifact_bytes": 10,
                "raw_artifact_sha256": "abc",
                "figure": "results/figures/f.png",
            }
        },
        "repair_v2": {
            "stage4_report": {"legacy_fallback_comparison": legacy}
        },
    }

    payload = build_stage4_payload(metrics)

    assert payload["status"] == "COMPLETE"
    assert payload["observations"]["primary_policy"]["max_swaps"] == 2
    assert payload["observations"]["masked_alpha_1_5"]["swaps"] == 3
    assert payload["observations"]["masked_alpha_1_5"]["low_causal_abs"] == 8
    assert payload["observations"]["masked_alpha_1_5"]["low_weight_read"] == 0
    assert payload["observations"]["capability"] == {
        "empty_masks": 2,
        "total_masks": 2,
        "interpretation": "NO_EDIT_OPPORTUNITY",
    }
    assert payload["claim_boundary"]["hypothesis_status"] == "NOT_TESTED"
    assert not payload["claim_boundary"]["hypothesis_false_established"]
    assert payload["legacy_fallback_comparison"] == legacy

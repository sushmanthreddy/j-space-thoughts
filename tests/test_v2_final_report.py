import copy
import json
from pathlib import Path

import pytest

from src.v2_final_report import (
    LEGACY_SOURCE_COMMIT,
    Stage4PrerequisiteError,
    build_stage4_report,
    persist_stage4,
)


def _metrics() -> dict:
    return {
        "repair_v2": {
            "current_allowed_conclusion": (
                "STAGE4_REPLICATION_FAILURE_NO_HYPOTHESIS_INFERENCE"
            ),
            "gate_ledger": {
                "stage3_science": "SKIPPED_PREREQUISITE",
                "stage4_report": "REQUIRED",
                "upstream_causal_swap": "NOT_RUNNABLE_RELEASE_OMISSION",
            },
            "stage0": {
                "decision": "UPSTREAM_CAUSAL_SWAP_NOT_RUNNABLE_RELEASE_OMISSION",
                "figure": "results/figures/f0.png",
                "upstream_release_audit": {"causal_swap_code_available": False},
            },
            "stage1": {
                "figure": "results/figures/swap.png",
                "g_swap": {
                    "status": "PASS",
                    "n_pass": 3,
                    "n_required": 3,
                },
            },
            "stage1c_concept_finder": {
                "status": "PASS",
                "figure": "results/figures/dir.png",
            },
            "stage1d_read_validation": {
                "status": "PASS",
                "figure": "results/figures/read.png",
            },
            "stage3_notebooks": {
                number: {
                    "status": "SKIPPED_PREREQUISITE",
                    "science_executed": False,
                    "model_inference_run": False,
                }
                for number in ("05", "06", "07")
            },
            "stage2_recalibration": {
                "status": "FAIL",
                "stage3_allowed": False,
                "stage4_required": True,
                "configuration": "raw alpha=2",
                "g_swap_reverification": {"status": "PASS"},
                "controls_fire": {"status": "PASS"},
                "random_pair_null": {"status": "PASS"},
                "absent_coordinate_null": {"status": "PASS"},
                "capability": {
                    "status": "FAIL",
                    "mean_delta_nll": 0.6233231922,
                    "mean_abs_delta_nll": 0.6690807243,
                    "threshold": 0.25,
                    "criterion": "absolute mean below threshold",
                },
                "g_pos": {
                    "status": "FAIL",
                    "n_reproduced": 1,
                    "n_passages": 8,
                    "categories_reproduced": ["Spanish"],
                    "criterion": "6/8 across at least three languages",
                },
                "figure_f3": "results/figures/f3.png",
            },
        }
    }


def _write_figures(root: Path, names: tuple[str, ...]) -> None:
    figure_dir = root / "results" / "figures"
    figure_dir.mkdir(parents=True)
    for name in names:
        (figure_dir / name).write_bytes(b"valid-png-placeholder")


def test_build_stage4_report_freezes_fallback_claims_and_legacy_values(
    tmp_path: Path,
) -> None:
    _write_figures(tmp_path, ("f0.png", "swap.png", "f3.png"))
    source = _metrics()
    before = copy.deepcopy(source)

    report = build_stage4_report(source, root=tmp_path)

    assert source == before
    assert report["classification"] == "OPEN_MODEL_INSTRUMENT_REPLICATION_FAILURE"
    assert report["custom_swap"]["n_pass"] == 3
    assert report["upstream_release"]["status"] == (
        "NOT_RUNNABLE_RELEASE_OMISSION"
    )
    assert report["predictions"] == {
        "P1": "NOT_TESTED",
        "P2": "NOT_TESTED",
        "P3": "NOT_TESTED",
    }
    assert set(report["skipped_notebooks"]) == {
        "05_science_twohop.ipynb",
        "06_science_ambiguity.ipynb",
        "07_scale.ipynb",
    }
    legacy = report["legacy_fallback_comparison"]
    assert legacy["provenance_commit"] == LEGACY_SOURCE_COMMIT
    assert legacy["n"] == 155
    assert legacy["jlens"] == {
        "pearson_r": 0.6083677,
        "ci_low": 0.5167659,
        "ci_high": 0.6927411,
    }
    assert legacy["identity_j_logit_lens"] == {
        "pearson_r": 0.6394085,
        "ci_low": 0.5519587,
        "ci_high": 0.7188587,
    }
    assert report["claim_boundary"]["hypothesis_false_established"] is False
    assert [row["id"] for row in report["valid_figures"]] == [
        "F0",
        "G-SWAP",
        "F3",
    ]


@pytest.mark.parametrize(
    ("ledger_value", "stage4_required"),
    (("ALLOWED", True), ("SKIPPED_PREREQUISITE", False)),
)
def test_build_rejects_nonfallback_state(
    tmp_path: Path, ledger_value: str, stage4_required: bool
) -> None:
    metrics = _metrics()
    metrics["repair_v2"]["gate_ledger"]["stage3_science"] = ledger_value
    metrics["repair_v2"]["stage2_recalibration"][
        "stage4_required"
    ] = stage4_required

    with pytest.raises(Stage4PrerequisiteError):
        build_stage4_report(metrics, root=tmp_path)


def test_build_rejects_missing_executed_skip_record(tmp_path: Path) -> None:
    metrics = _metrics()
    del metrics["repair_v2"]["stage3_notebooks"]["06"]
    with pytest.raises(Stage4PrerequisiteError, match="stage3_notebooks.06"):
        build_stage4_report(metrics, root=tmp_path)


def test_persist_stage4_updates_metrics_and_replaces_existing_tail(
    tmp_path: Path,
) -> None:
    _write_figures(tmp_path, ("f0.png", "swap.png", "dir.png", "read.png", "f3.png"))
    metrics_path = tmp_path / "results" / "metrics.json"
    report_path = tmp_path / "results" / "RESULTS.md"
    metrics_path.write_text(json.dumps(_metrics()), encoding="utf-8")
    report_path.write_text(
        "# Repair-first replication report (v2)\n\n"
        "## Stage 0 — diagnosis\n\nkept zero\n\n"
        "## Stage 1 — repair\n\nkept one\n\n"
        "## Stage 2 — recalibration\n\nkept two\n\n"
        "## Stage 4 — replication-failure fallback\n\nstale tail\n",
        encoding="utf-8",
    )

    persisted = persist_stage4(
        metrics_path=metrics_path,
        report_path=report_path,
        root=tmp_path,
    )

    stage4 = persisted["repair_v2"]["stage4_report"]
    assert stage4["status"] == "COMPLETE"
    assert persisted["repair_v2"]["gate_ledger"]["stage4_report"] == "COMPLETE"
    assert persisted["repair_v2"]["gate_ledger"]["stage3_science"] == (
        "SKIPPED_PREREQUISITE"
    )
    assert persisted["repair_v2"]["current_allowed_conclusion"] == (
        "STAGE4_REPLICATION_FAILURE_NO_HYPOTHESIS_INFERENCE"
    )

    disk_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert disk_metrics["repair_v2"]["stage4_report"] == stage4
    rendered = report_path.read_text(encoding="utf-8")
    assert rendered.count("## Stage 4 — replication-failure fallback") == 1
    assert "stale tail" not in rendered
    assert "kept zero" in rendered and "kept one" in rendered and "kept two" in rendered
    assert "05_science_twohop.ipynb" in rendered
    assert "P1 | **NOT_TESTED**" in rendered
    assert "0.6083677" in rendered and "0.6394085" in rendered
    assert "does not establish that the WRITE-versus-READ hypothesis is" in rendered


def test_persist_guard_failure_does_not_write(tmp_path: Path) -> None:
    metrics = _metrics()
    metrics["repair_v2"]["gate_ledger"]["stage3_science"] = "ALLOWED"
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "RESULTS.md"
    original_metrics = json.dumps(metrics)
    original_report = "## Stage 0\n## Stage 1\n## Stage 2\n"
    metrics_path.write_text(original_metrics, encoding="utf-8")
    report_path.write_text(original_report, encoding="utf-8")

    with pytest.raises(Stage4PrerequisiteError):
        persist_stage4(
            metrics_path=metrics_path,
            report_path=report_path,
            root=tmp_path,
        )

    assert metrics_path.read_text(encoding="utf-8") == original_metrics
    assert report_path.read_text(encoding="utf-8") == original_report

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _v4_metrics() -> dict:
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    return metrics["calibration_v4"]


def test_v4_one_shot_gate_and_claim_boundary() -> None:
    v4 = _v4_metrics()
    protocol = v4["protocol"]
    validation = v4["stage11_readval"]
    report = v4["stage14_report"]

    assert protocol["new_read_estimators_permitted"] == 1
    assert protocol["causal_intervention"]["alpha_resweep"] is False
    assert protocol["path_threshold_abs_delta_m"] == 0.05
    assert validation["g_readval"] == "FAIL"
    assert v4["gate_ledger"] == {
        "g_readval": "FAIL",
        "stage_a_science": "SKIPPED_PREREQUISITE",
        "stage_b_report": "PASS",
    }

    known = validation["known_predictivity"]
    assert known["n"] == 21
    assert known["checks"]["all_21_locked_rows_estimable"] is True
    assert known["checks"]["spearman_rho_at_least_0_4"] is False
    assert known["checks"]["bootstrap_ci_lower_strictly_positive"] is False

    narration = validation["narration_separation"]
    assert narration["n"] == 8
    assert narration["n_finite_behavior_specific_ratios"] == 0
    assert narration["n_joint_pass"] == 0
    assert narration["checks"]["empty_auto_paths_not_counted_as_low"] is True

    assert report["status"] == "COMPLETE"
    assert report["classification"] == "READ_OPERATIONALIZATION_METHODS_LIMITATION"
    assert report["predictions"] == {
        "P1": "NOT_TESTED",
        "P2": "NOT_TESTED",
        "P3": "NOT_TESTED",
    }
    assert report["claim_boundary"]["hypothesis_true_established"] is False
    assert report["claim_boundary"]["hypothesis_false_established"] is False
    assert report["one_shot_compliance"] == {
        "alpha_resweep": False,
        "extra_notebooks": [],
        "new_read_estimators": 1,
        "path_thresholds_tested": [0.05],
    }
    assert "NO_EDIT_OPPORTUNITY" in report["working_instrument"][
        "capability_guardrail"
    ]


def test_v4_notebook_chain_is_executed_and_science_skips_are_model_free() -> None:
    expected = {
        "10_behavior_specific_read.ipynb",
        "11_readval_gate.ipynb",
        "12_science_twohop.ipynb",
        "13_science_ambiguity.ipynb",
        "14_report.ipynb",
    }
    actual = {path.name for path in (ROOT / "notebooks").glob("1[0-4]_*.ipynb")}
    assert actual == expected

    for name in sorted(expected):
        notebook = json.loads((ROOT / "notebooks" / name).read_text())
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert [cell.get("execution_count") for cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        )
        assert all(
            output.get("output_type") != "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )
        assert all(cell.get("id") for cell in notebook["cells"])

    for name in ("12_science_twohop.ipynb", "13_science_ambiguity.ipynb"):
        notebook = json.loads((ROOT / "notebooks" / name).read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        ).lower()
        assert "skipped_prerequisite" in source
        assert "load_model" not in source
        assert "transformers" not in source
        assert "import torch" not in source


def test_v4_report_states_the_methods_limitation_without_a_verdict() -> None:
    report = (ROOT / "results/RESULTS.md").read_text()
    readme = (ROOT / "README.md").read_text()
    required = (
        "READ-operationalization",
        "G-READVAL FAIL",
        "P1, P2, and P3 are NOT TESTED",
        "neither supports nor refutes",
        "NO_AUTO_PATH_DETECTED",
        "NO_EDIT_OPPORTUNITY",
        "no further estimator attempt",
    )
    assert all(phrase in report for phrase in required)
    assert report.index("## Preflight") < report.index("## G-READVAL decision")
    assert report.index("## G-READVAL decision") < report.index(
        "## Stage B — methods limitation"
    )
    assert "no hypothesis verdict" in readme
    assert "P1–P3 were not run" in readme


def test_v4_raw_artifact_metadata_and_available_files_match() -> None:
    report = _v4_metrics()["stage14_report"]
    artifacts = report["raw_artifacts"]
    assert len(artifacts) == 2
    for artifact in artifacts:
        path = ROOT / artifact["path"]
        assert len(artifact["sha256"]) == 64
        assert artifact["bytes"] > 0
        if path.exists():
            assert path.stat().st_size == artifact["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

import json
from pathlib import Path

from src import v2_stage0


def test_upstream_release_constants_match_pinned_checkout():
    assert v2_stage0._sha256(v2_stage0.WALKTHROUGH) == (
        v2_stage0.EXPECTED_WALKTHROUGH_SHA256
    )
    assert v2_stage0._sha256(v2_stage0.PROBE_SWAP) == (
        v2_stage0.EXPECTED_PROBE_SWAP_SHA256
    )


def test_upstream_walkthrough_cells_have_no_mutation_api():
    notebook = json.loads(v2_stage0.WALKTHROUGH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for marker in (
        "register_forward_hook",
        "register_forward_pre_hook",
        "swap_coordinates",
        "ablate_direction",
        "residual_edit",
    ):
        assert marker not in code


def test_stage0_audit_classifies_release_omission():
    audit = v2_stage0.audit_upstream_release()
    assert audit["dependency_commit"] == v2_stage0.EXPECTED_DEPENDENCY_COMMIT
    assert audit["dependency_clean"] is True
    assert audit["walkthrough_capability"] == "READOUT_ONLY"
    assert audit["canonical_swap_runnable_unchanged"] is False
    assert audit["g_swap_status"] == "UNTESTED"
    assert audit["model_mismatch_inference_permitted"] is False
    assert len(audit["spider_probe_rows"]) == 1
    row = audit["spider_probe_rows"][0]
    assert row["answer"] == "8"
    assert row["swap_answer"] == "6"


def test_v2_result_path_is_inside_repository():
    assert Path(v2_stage0.ROOT, "results", "metrics.json").is_relative_to(
        v2_stage0.ROOT
    )

import copy

import torch

from src.v3_reverify import verify_gdir_artifact


def _v2_metrics() -> dict:
    return {
        "stage1c_concept_finder": {
            "status": "PASS",
            "chance_retrieval": 0.025,
            "retrieval": {"top1_at_fixed_layer": {"estimate": 0.55}},
            "explicit_known_answer": {"heldout_top5": {"estimate": 0.8875}},
        }
    }


def test_verify_gdir_artifact_checks_unit_norm_and_metrics(tmp_path):
    artifact = {
        "metadata": {
            "schema_version": "md-repair-v2",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
        },
        "mean_difference": {
            f"concept-{index}": {24: torch.tensor([1.0, 0.0])}
            for index in range(40)
        },
    }
    path = tmp_path / "md.pt"
    torch.save(artifact, path)
    source = _v2_metrics()
    before = copy.deepcopy(source)
    result = verify_gdir_artifact(source, artifact_path=path)
    assert source == before
    assert result["status"] == "PASS"
    assert result["n_concepts"] == 40
    assert result["n_direction_vectors"] == 40


def test_verify_gdir_artifact_fails_nonunit_direction(tmp_path):
    artifact = {
        "metadata": {
            "schema_version": "md-repair-v2",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
        },
        "mean_difference": {
            f"concept-{index}": {24: torch.tensor([2.0, 0.0])}
            for index in range(40)
        },
    }
    path = tmp_path / "md.pt"
    torch.save(artifact, path)
    result = verify_gdir_artifact(_v2_metrics(), artifact_path=path)
    assert result["status"] == "FAIL"
    assert result["checks"]["unit_norm"] is False

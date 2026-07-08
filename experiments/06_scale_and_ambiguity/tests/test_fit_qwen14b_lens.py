from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.fit_qwen14b_lens as fit_module
from src.fit_qwen14b_lens import (
    MODEL_ID,
    WIKITEXT_REVISION,
    checkpoint_provenance_payload,
    exclusive_fit_lock,
    fit_qwen14b_lens,
    prepare_checkpoint_provenance,
    validate_completed_fit_artifacts,
)
from src.model_utils import MODEL_REVISIONS


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _provenance() -> dict:
    return checkpoint_provenance_payload(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISIONS[MODEL_ID],
        wikitext_revision=WIKITEXT_REVISION,
        prompt_sha256=[_hash("first"), _hash("second")],
        source_layers=[2, 3],
        target_layer=4,
        n_prompts=2,
        max_seq_len=32,
        dim_batch=2,
        checkpoint_every=1,
    )


def test_exclusive_fit_lock_rejects_a_second_writer(tmp_path) -> None:
    lock_path = tmp_path / "fit.lock"

    with exclusive_fit_lock(lock_path):
        owner = json.loads(lock_path.read_text())
        assert isinstance(owner["pid"], int)
        with pytest.raises(RuntimeError, match="Another Qwen-14B lens fit"):
            with exclusive_fit_lock(lock_path):
                raise AssertionError("unreachable")

    with exclusive_fit_lock(lock_path):
        pass


def test_checkpoint_sidecar_is_written_before_new_fit_and_exact_on_resume(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    sidecar = tmp_path / "checkpoint.provenance.json"
    expected = _provenance()

    new = prepare_checkpoint_provenance(checkpoint, sidecar, expected)
    assert new["mode"] == "new_fit"
    assert json.loads(sidecar.read_text()) == expected

    checkpoint.write_bytes(b"checkpoint")
    resumed = prepare_checkpoint_provenance(checkpoint, sidecar, expected)
    assert resumed["mode"] == "resume_validated"
    assert resumed["checkpoint_bytes"] == len(b"checkpoint")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_revision", "wrong-revision"),
        ("wikitext_revision", "wrong-dataset"),
        ("prompt_sha256", [_hash("second"), _hash("first")]),
        ("source_layers", [1, 2]),
        ("target_layer", 5),
        ("n_prompts", 3),
        ("max_seq_len", 64),
        ("dim_batch", 4),
        ("checkpoint_every", 2),
    ],
)
def test_checkpoint_resume_fails_closed_on_every_bound_field(
    tmp_path,
    field,
    replacement,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    sidecar = tmp_path / "checkpoint.provenance.json"
    expected = _provenance()
    checkpoint.write_bytes(b"checkpoint")
    actual = dict(expected)
    actual[field] = replacement
    sidecar.write_text(json.dumps(actual), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        prepare_checkpoint_provenance(checkpoint, sidecar, expected)


def test_existing_checkpoint_without_sidecar_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"legacy checkpoint")

    with pytest.raises(RuntimeError, match="without a nonempty matching"):
        prepare_checkpoint_provenance(
            checkpoint,
            tmp_path / "missing.provenance.json",
            _provenance(),
        )


class _FakeLens:
    def __init__(self, n_prompts: int) -> None:
        self.n_prompts = n_prompts

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"complete fake lens")


def _fake_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        model_id=MODEL_ID,
        revision=MODEL_REVISIONS[MODEL_ID],
        lens_model=SimpleNamespace(n_layers=5),
    )


def test_mocked_fit_publishes_atomic_artifacts_and_releases_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    released: list[object] = []
    observed_sidecar: list[dict] = []
    monkeypatch.setattr(fit_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        fit_module,
        "pinned_wikitext_prompts",
        lambda n_prompts: [f"prompt-{index}" for index in range(n_prompts)],
    )
    monkeypatch.setattr(fit_module, "load_model", lambda model_id: _fake_bundle())
    monkeypatch.setattr(
        fit_module,
        "release_model",
        lambda bundle: released.append(bundle),
    )
    monkeypatch.setattr(fit_module.torch.cuda, "is_available", lambda: False)

    def fake_fit(model, prompts, **kwargs):
        del model, prompts
        sidecar = tmp_path / "data/lenses/qwen2.5-14b_fit_ckpt.provenance.json"
        observed_sidecar.append(json.loads(sidecar.read_text()))
        assert kwargs["source_layers"] == [2, 3]
        assert kwargs["target_layer"] == 4
        return _FakeLens(n_prompts=2)

    monkeypatch.setattr(fit_module.jlens, "fit", fake_fit)

    metadata = fit_qwen14b_lens(
        n_prompts=2,
        dim_batch=2,
        max_seq_len=32,
        checkpoint_every=1,
    )

    lens_path = tmp_path / "data/lenses/qwen2.5-14b_jlens_2prompts.pt"
    metadata_path = tmp_path / "data/lenses/qwen2.5-14b_jlens_2prompts.json"
    assert lens_path.read_bytes() == b"complete fake lens"
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata["atomic_final_artifacts"] is True
    assert metadata["checkpoint_resume_mode"] == "new_fit"
    assert observed_sidecar[0]["prompt_sha256"] == [
        _hash("prompt-0"),
        _hash("prompt-1"),
    ]
    assert len(released) == 1
    assert not list((tmp_path / "data/lenses").glob(".*.tmp.*"))


def test_mocked_fit_releases_bundle_after_fit_exception(tmp_path, monkeypatch) -> None:
    released: list[object] = []
    monkeypatch.setattr(fit_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        fit_module,
        "pinned_wikitext_prompts",
        lambda n_prompts: [f"prompt-{index}" for index in range(n_prompts)],
    )
    monkeypatch.setattr(fit_module, "load_model", lambda model_id: _fake_bundle())
    monkeypatch.setattr(
        fit_module,
        "release_model",
        lambda bundle: released.append(bundle),
    )
    monkeypatch.setattr(fit_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        fit_module.jlens,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )

    with pytest.raises(RuntimeError, match="fit failed"):
        fit_qwen14b_lens(
            n_prompts=2,
            dim_batch=2,
            max_seq_len=32,
            checkpoint_every=1,
        )

    assert len(released) == 1
    assert not (tmp_path / "data/lenses/qwen2.5-14b_jlens_2prompts.pt").exists()


def test_completed_artifact_validation_rejects_empty_or_malformed_metadata(
    tmp_path,
) -> None:
    lens_path = tmp_path / "lens.pt"
    metadata_path = tmp_path / "lens.json"
    lens_path.write_bytes(b"lens")
    metadata_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read final lens metadata"):
        validate_completed_fit_artifacts(
            lens_path,
            metadata_path,
            expected_n_prompts=2,
        )

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISIONS[MODEL_ID],
        "wikitext_revision": WIKITEXT_REVISION,
        "n_prompts_requested": 2,
        "n_prompts_fitted": 2,
        "selection": "first 2 train records with >=600 characters",
        "source_layers": [2, 3],
        "target_layer": 4,
        "max_seq_len": 32,
        "dim_batch": 2,
        "checkpoint_every": 1,
        "prompt_sha256": [_hash("first"), _hash("second")],
        "lens_path": str(lens_path),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert validate_completed_fit_artifacts(
        lens_path,
        metadata_path,
        expected_n_prompts=2,
        expected_source_layers=[2, 3],
        expected_target_layer=4,
        expected_max_seq_len=32,
        expected_dim_batch=2,
        expected_checkpoint_every=1,
    ) == metadata

    lens_path.write_bytes(b"")
    with pytest.raises(ValueError, match="missing or empty"):
        validate_completed_fit_artifacts(
            lens_path,
            metadata_path,
            expected_n_prompts=2,
            expected_source_layers=[2, 3],
            expected_target_layer=4,
            expected_max_seq_len=32,
            expected_dim_batch=2,
            expected_checkpoint_every=1,
        )


def test_notebook_statistics_cell_handles_unavailable_statistics() -> None:
    notebook = json.loads(
        (fit_module.ROOT / "notebooks/06_scale_comparison.ipynb").read_text()
    )
    setup_source = "".join(
        source_line
        for cell in notebook["cells"]
        if cell.get("id") == "setup"
        for source_line in cell["source"]
    )
    assert "validate_completed_fit_artifacts" in setup_source
    assert "known first 10 resumed contributions" in setup_source

    statistics_source = "".join(
        source_line
        for cell in notebook["cells"]
        if cell.get("id") == "statistics"
        for source_line in cell["source"]
    )
    comparison = {
        "models": {
            "14B": {
                "methods": {
                    "jlens_raw_wu_j": {"status": "NOT_AVAILABLE", "n": 0},
                    "mean_difference": {
                        "status": "ESTIMATED",
                        "n": 2,
                        "partial_correlations": {
                            "causal_read_given_write": {
                                "status": "NOT_ESTIMABLE"
                            }
                        },
                        "attribution_predicted_vs_real": {
                            "status": "NOT_ESTIMABLE"
                        },
                        "mean_ablation_positive_damage": {
                            "status": "NOT_ESTIMABLE"
                        },
                    },
                }
            }
        },
        "paired_14b_minus_7b": {
            "mean_difference": {"status": "NOT_AVAILABLE", "n_common": 0}
        },
        "p1_interpretation": {"status": "DESCRIPTIVE_ESTIMATES_ONLY"},
    }
    displayed: list[object] = []
    namespace = {
        "comparison": comparison,
        "display": displayed.append,
        "pd": pd,
    }

    exec(compile(statistics_source, "notebook-statistics-cell", "exec"), namespace)

    rows = namespace["rows"]
    assert rows[0]["method status"] == "NOT_AVAILABLE"
    assert rows[0]["partial causal~READ|WRITE"] is None
    assert rows[1]["READ status"] == "NOT_ESTIMABLE"
    assert rows[1]["attribution r"] is None
    assert len(displayed) == 2

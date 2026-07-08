"""Build the six auditable notebooks for the symmetric causal READ run."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path("/home/jovyan/j-space-thoughts")

NOTEBOOKS = {
    30: {
        "slug": "dataset_and_verification",
        "title": "Matched dataset and verification",
        "description": (
            "Build 118 natural reciprocal prompt pairs, freeze calibration-only "
            "layer/WRITTEN selection, enforce clean-answer and WRITTEN gates, and "
            "log every failure as UNVERIFIED."
        ),
    },
    31: {
        "slug": "causal_ground_truth",
        "title": "Symmetric causal ground truth",
        "description": (
            "Compute signed two-direction full-residual interchange C and the "
            "separately labelled J-Lens two-concept-subspace diagnostic for "
            "task-matched engine/dashboard controls."
        ),
    },
    32: {
        "slug": "cheap_read",
        "title": "Gradient-only cheap READ",
        "description": (
            "Compute 16-step midpoint READ_IG, READ_local, and the labelled "
            "known-broken capacity baseline. This notebook cannot load the "
            "causal artifact and performs a static import-isolation audit."
        ),
    },
    33: {
        "slug": "trust_check",
        "title": "Held-out trust check",
        "description": (
            "Join the independently generated causal truth and cheap scores, "
            "evaluate grouped held-out AUC/CIs, create F1--F4, and make the "
            "pre-registered one-line GO/NO-GO decision."
        ),
    },
    34: {
        "slug": "localization",
        "title": "GO-only signed localization",
        "description": (
            "Run signed component restoration and outside-circuit zero-ablation "
            "faithfulness only when Phase 3 is GO; otherwise record an executed "
            "prerequisite skip without loading the model."
        ),
    },
    35: {
        "slug": "report",
        "title": "Final report and completion audit",
        "description": (
            "Audit every required artifact, schema field, figure, executed "
            "notebook, anti-circularity boundary, and RESULTS.md section order."
        ),
    },
}


for number, spec in NOTEBOOKS.items():
    driver = ROOT / f"scripts/run_symmetric_phase{number}.py"
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"# {number} — {spec['title']}\n\n{spec['description']}"
        ),
        nbformat.v4.new_code_cell(
            f"""import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
driver = ROOT / 'scripts/{driver.name}'
exec(compile(driver.read_text(), str(driver), 'exec'), globals(), globals())"""
        ),
    ]
    target = ROOT / f"notebooks/{number}_{spec['slug']}.ipynb"
    nbformat.write(notebook, target)
    print(target)

from __future__ import annotations

import numpy as np
import torch

from src.metrics import logit_difference, partial_correlation, standardized_regression


def test_logit_difference_sign() -> None:
    logits = torch.zeros(2, 3, 5)
    logits[:, -1, 2] = torch.tensor([4.0, 1.0])
    logits[:, -1, 3] = torch.tensor([1.5, 2.0])
    assert torch.equal(logit_difference(logits, 2, 3), torch.tensor([2.5, -1.0]))


def test_partial_correlation_removes_shared_write_signal() -> None:
    rng = np.random.default_rng(7)
    write = rng.normal(size=300)
    read = 0.8 * write + rng.normal(scale=0.4, size=300)
    causal = 2.0 * read + 0.5 * write + rng.normal(scale=0.2, size=300)
    read_partial = partial_correlation(causal, read, write)
    write_partial = partial_correlation(causal, write, read)
    assert read_partial > write_partial
    assert read_partial > 0.9


def test_standardized_regression_returns_named_coefficients() -> None:
    write = np.arange(1, 21, dtype=float)
    read = np.sin(write)
    causal = 0.25 * write + 3.0 * read
    result = standardized_regression(causal, write, read)
    assert result["n"] == 20
    assert set(result["coefficients"]) == {"intercept", "write", "read"}
    assert result["r_squared"] > 0.999


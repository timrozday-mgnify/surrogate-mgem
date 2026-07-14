"""Tests for the verify-in-loop safety net."""

import pytest

from surrogate_mgem.infer import verify_shortlist


def test_verification_reranks_by_true_score():
    # Surrogate ranks A best, but the real evaluator prefers C. Verification of
    # the top-2 surrogate picks (A, B) must return them sorted by true score.
    candidates = ["A", "B", "C"]
    surrogate = [3.0, 2.0, 1.0]
    true = {"A": 0.5, "B": 0.9, "C": 5.0}
    out = verify_shortlist(candidates, surrogate, lambda c: true[c], k=2)
    assert [v.item for v in out] == ["B", "A"]  # C not in shortlist; B beats A on truth
    assert out[0].true_score == 0.9
    assert out[0].surrogate_score == 2.0


def test_only_k_real_evaluations_spent():
    calls = []
    verify_shortlist([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], lambda c: calls.append(c) or float(c), k=2)
    assert len(calls) == 2  # only the top-k are solved, not all candidates


def test_k_clamped_and_empty_cases():
    assert verify_shortlist([], [], lambda c: 0.0, k=3) == []
    assert verify_shortlist(["A"], [1.0], lambda c: 0.0, k=0) == []
    assert len(verify_shortlist(["A", "B"], [1.0, 2.0], lambda c: 0.0, k=99)) == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        verify_shortlist(["A", "B"], [1.0], lambda c: 0.0, k=1)

import numpy as np
import pandas as pd
import pytest

from xai_consistency import metrics as M

IDX = list("abcde")


def s(values):
    return pd.Series(values, index=IDX, dtype=float)


def test_rank_agreement():
    a = s([5, 4, 3, 2, 1])
    assert M.rank_agreement(a, a * 10)["spearman"] == pytest.approx(1)
    assert M.rank_agreement(a, -a)["kendall"] == pytest.approx(-1)
    # шкала не важна, важен только порядок
    assert M.rank_agreement(a, np.exp(a))["spearman"] == pytest.approx(1)
    assert np.isnan(M.rank_agreement(a, s([1, 1, 1, 1, 1]))["spearman"])


def test_top_k_overlap():
    a, b = s([5, 4, 3, 2, 1]), s([5, 1, 4, 2, 3])
    assert M.top_k_overlap(a, b, 1) == 1
    assert M.top_k_overlap(a, b, 2) == 0.5
    assert M.top_k_overlap(a, b, 5) == 1


def test_sign_agreement_ignores_undefined_reference():
    ref = s([1, -1, 0, 1, 0])
    assert M.sign_agreement(s([1, -1, 1, 1, -1]), ref) == 1
    assert M.sign_agreement(s([1, 1, 1, 0, 1]), ref) == 1 / 3
    assert np.isnan(M.sign_agreement(s([1] * 5), s([0] * 5)))


def test_curve_agreement():
    x = np.linspace(-1, 1, 20)
    w = np.ones(20)
    same = M.curve_agreement(x**2, x**2 + 5, w)  # сдвиг не важен
    assert np.isclose(same["ccc"], 1) and np.isclose(same["nrmsd"], 0)
    half = M.curve_agreement(x, 0.5 * x, w)  # та же форма, другой размах
    assert np.isclose(half["pearson"], 1) and np.isclose(half["ccc"], 0.8)
    assert np.isclose(M.curve_agreement(x, -x, w)["ccc"], -1)
    flat = M.curve_agreement(x, np.zeros(20), w)
    assert flat["ccc"] == 0 and np.isnan(flat["pearson"])
    # NaN и пустые интервалы пропускаются
    a = x.copy(); a[3] = np.nan
    assert np.isclose(M.curve_agreement(a, x, w)["ccc"], 1)


def test_fidelity():
    true = s([3, 2, 1, 0, 0])
    good = s([30, 20, 10, 1, 2])
    r = M.fidelity(good, true, ["a", "b", "c"])
    assert r["recall_at_k"] == 1 and r["all_found"] == 1 and np.isclose(r["spearman_relevant"], 1)
    assert np.isclose(r["margin"], 10 / 2)
    bad = s([30, 20, 1, 10, 2])
    r = M.fidelity(bad, true, ["a", "b", "c"])
    assert np.isclose(r["recall_at_k"], 2 / 3) and r["all_found"] == 0 and r["margin"] < 1


def test_stability():
    base = s([5, 4, 3, 2, 1])
    same = pd.DataFrame([base, base * 2, base + 1])
    st = M.stability(same, k=2)
    assert np.isclose(st["mean_spearman"], 1) and st["mean_topk_jaccard"] == 1 and st["mean_rank_sd"] == 0
    rng = np.random.default_rng(0)
    noisy = pd.DataFrame(rng.random((30, 5)), columns=IDX)
    assert M.stability(noisy, k=2)["mean_spearman"] < 0.3


def test_null_pvalues():
    rng = np.random.default_rng(0)
    null = pd.DataFrame(rng.normal(0, 1, (99, 5)), columns=IDX)
    obs = s([10, 0, -10, 0, 0])
    p = M.null_pvalues(obs, null)
    assert p["a"] == 0.01 and p["c"] == 1.0 and 0.2 < p["b"] < 0.8


def test_local_global():
    rng = np.random.default_rng(0)
    g = s([5, 4, 3, 2, 1])
    local = pd.DataFrame(rng.normal(0, 0.1, (50, 5)) + g.to_numpy(), columns=IDX)
    r = M.local_global(local, g, k=1)
    assert np.isclose(r["agg_spearman"], 1) and r["top1_in_global_topk"] == 1
    assert (r["sign_consistency"] == 1).all()


def test_bh_fdr_matches_manual():
    p = pd.Series([0.01, 0.04, 0.03, 0.2, np.nan], index=IDX)
    q = M.bh_fdr(p)
    # m = 4: отсортировано 0.01, 0.03, 0.04, 0.2 → 0.04, 0.0533, 0.0533, 0.2
    assert q["a"] == pytest.approx(0.04)
    assert q["c"] == pytest.approx(0.04 * 4 / 3)
    assert q["b"] == pytest.approx(0.04 * 4 / 3)
    assert q["d"] == pytest.approx(0.2)
    assert np.isnan(q["e"])


def test_null_tail_pvalues_calibrated():
    rng = np.random.default_rng(0)
    null = pd.DataFrame(np.exp(rng.normal(0, 1, (200, 5))), columns=IDX)
    obs = pd.Series(np.exp([0.0, 1.645, 3.0, -1.0, 2.326]), index=IDX)
    p = M.null_tail_pvalues(obs, null)
    assert p["a"] == pytest.approx(0.5, abs=0.08)
    assert p["b"] == pytest.approx(0.05, abs=0.02)
    assert p["c"] < 0.005
    assert p["e"] == pytest.approx(0.01, abs=0.007)

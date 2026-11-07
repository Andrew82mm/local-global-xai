import numpy as np
import pytest

from xai_consistency.synthetic import R2_LEVELS, SCENARIOS, make_synthetic


def test_r2_matches_target():
    for sc in SCENARIOS:
        for r2 in R2_LEVELS:
            d = make_synthetic(sc, n=50_000, r2=r2, seed=3)
            assert abs(d.y.var() - 1) < 0.03, (sc, r2)
            assert abs(d.signal.var() / d.y.var() - r2) < 0.015, (sc, r2)


def test_true_shap_sums_to_signal():
    for sc in SCENARIOS:
        d = make_synthetic(sc, n=2000, r2=0.5, seed=0)
        assert np.allclose(d.true_shap.sum(axis=1), d.signal)


def test_zero_signal_has_no_truth():
    d = make_synthetic("independent", n=1000, r2=0.0, seed=0)
    assert np.allclose(d.signal, 0)
    assert d.relevant == []
    assert (d.true_sign == 0).all()


def test_correlation_structure():
    d = make_synthetic("correlated", n=50_000, r2=0.5, rho=0.9, seed=1)
    c = d.X.corr()
    for j in range(1, 5):
        assert abs(c.loc[f"x{j}", f"x{j + 4}"] - 0.9) < 0.01
    assert abs(c.loc["x1", "x2"]) < 0.02
    # у двойника нет своего эффекта
    assert d.true_importance["x5"] == 0


def test_interaction_split_equally_and_no_main_effect():
    d = make_synthetic("interaction", n=20_000, r2=0.5, seed=0)
    imp = d.true_importance
    assert abs(imp["x1"] - imp["x2"]) / imp["x1"] < 0.05
    grid = np.linspace(-2, 2, 11)
    assert np.allclose(d.true_curve("x1", grid), 0)
    assert d.true_sign["x1"] == 0 and d.true_sign["x3"] == 1


def test_true_curve_is_centered_component():
    d = make_synthetic("nonlinear", n=200_000, r2=0.5, seed=0)
    # среднее истинного главного эффекта по популяции равно нулю
    for f in ["x1", "x2", "x3", "x4"]:
        assert abs(d.true_curve(f, d.X[f].to_numpy()).mean()) < 0.01
    # x² — U-образная кривая
    v = d.true_curve("x1", np.array([-2.0, 0.0, 2.0]))
    assert v[0] > v[1] < v[2] and np.isclose(v[0], v[2])


def test_reproducible_and_seed_dependent():
    a = make_synthetic("nonlinear", n=100, r2=0.2, seed=5)
    b = make_synthetic("nonlinear", n=100, r2=0.2, seed=5)
    c = make_synthetic("nonlinear", n=100, r2=0.2, seed=6)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y)


@pytest.mark.parametrize("kwargs", [{"scenario": "unknown"},
                                    {"scenario": "independent", "r2": 1.5},
                                    {"scenario": "correlated", "n_features": 6}])
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        make_synthetic(**kwargs)

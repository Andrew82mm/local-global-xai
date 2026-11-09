"""Методы на моделях, где ответ известен аналитически."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from xai_consistency import explain as E
from xai_consistency.models import fit_transform, make_pipeline
from xai_consistency.synthetic import make_synthetic


def _linear_model(n=500, seed=0):
    rng = np.random.default_rng(seed)
    Z = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = 2 * Z["a"] - 1 * Z["b"] + 0 * Z["c"]
    return LinearRegression().fit(Z.to_numpy(), y), Z, y.to_numpy()


def test_pdp_and_ale_recover_linear_slope():
    model, Z, _ = _linear_model()
    grids = E.make_grids(Z, Z.columns)
    for ex in (E.partial_dependence(model, Z, grids), E.accumulated_local_effects(model, Z, grids)):
        for f, slope in [("a", 2), ("b", -1), ("c", 0)]:
            c = ex.curves[f]
            fit = np.polyfit(c["x"], c["value"], 1)[0]
            assert np.isclose(fit, slope, atol=1e-8), (ex.method, f)
        assert ex.sign["a"] == 1 and ex.sign["b"] == -1


def test_shap_and_lime_on_linear_model():
    model, Z, _ = _linear_model()
    sh = E.shap_values(model, "ridge", Z.iloc[:100], Z.iloc[:50], "test")
    pred = model.predict(Z.iloc[:50].to_numpy())
    assert np.allclose(sh.info["expected_value"] + sh.local.sum(axis=1), pred)
    lime = E.lime_values(model, Z, Z.iloc[:5], "test", seed=0, num_samples=2000)
    # у линейной модели локальная аппроксимация = сама модель: коэффициент × SD признака
    sd = Z.std(ddof=0)
    assert np.allclose(lime.local.mean(), [2 * sd["a"], -sd["b"], 0], atol=0.02)


def test_lime_categorical_flag_step():
    """Ступенька на флаге: categorical даёт полный эффект перехода, гауссово возмущение — меньше."""
    rng = np.random.default_rng(0)
    n = 600
    flag = (rng.random(n) < 0.2).astype(float)
    Z = pd.DataFrame({"a": rng.normal(size=n), "flag": (flag - flag.mean()) / flag.std()})

    mid = (Z["flag"].min() + Z["flag"].max()) / 2

    class Step:
        def predict(self, Zv):
            Zv = np.asarray(Zv, float)
            return Zv[:, 0] + 3.0 * (Zv[:, 1] > mid)  # порог посередине, как у деревьев

    cat = E.lime_values(Step(), Z, Z.iloc[:20], "test", seed=0, num_samples=3000, categorical=["flag"])
    assert np.allclose(cat.info["flip"]["flag"], 3.0, atol=0.15)  # знак «+» и у объектов с 0, и с 1
    gap = Z["flag"].max() - Z["flag"].min()
    assert np.isclose(cat.importance["flag"], 3.0 * Z["flag"].std(ddof=0) / gap, rtol=0.05)
    gauss = E.lime_values(Step(), Z, Z.iloc[:20], "test", seed=0, num_samples=3000)
    # гауссов сосед пересекает порог редко: наклон ≈ φ(расстояние до порога) · 3, около половины
    assert gauss.importance["flag"] < 0.6 * cat.importance["flag"]
    assert np.isclose(cat.importance["a"], 1.0, atol=0.1)


def test_pi_group_and_zero_for_unused():
    model, Z, y = _linear_model()
    pi = E.permutation_importance(model, Z, y, n_repeats=5, seed=0)
    assert pi.importance["c"] == 0 and pi.importance["a"] > pi.importance["b"] > 0
    grp = E.permutation_importance(model, Z, y, n_repeats=5, seed=0, groups={"ab": ["a", "b"]})
    assert grp.importance["ab"] > pi.importance["a"]


def test_hgb_shap_additivity_on_train_points():
    """Регрессия: без округления до float32 ошибка аддитивности доходила до 0.2."""
    d = make_synthetic("independent", n=1000, r2=0.5, seed=0)
    pipe = make_pipeline("hgb", other_cols=list(d.X.columns), random_state=0)
    Z_train, _ = fit_transform(pipe, d.X.iloc[:800], d.y[:800], d.X.iloc[800:])
    model = pipe.named_steps["model"]
    sh = E.shap_values(model, "hgb", Z_train.iloc[:100], Z_train.iloc[100:300], "train")
    err = sh.info["expected_value"] + sh.local.sum(axis=1) - model.predict(Z_train.iloc[100:300].to_numpy())
    assert np.abs(err).max() < 1e-5


def test_fit_and_explain_format():
    d = make_synthetic("independent", n=300, r2=0.5, seed=0)
    res = E.fit_and_explain("ridge", d.X.iloc[:240], d.y[:240], d.X.iloc[240:], d.y[240:],
                            n_explain=20, lime_samples=500, pi_repeats=2)
    assert set(res.explanations) == {("pi", "test"), ("pdp", "test"), ("ale", "test"), ("shap", "train"),
                                     ("shap", "test"), ("lime", "train"), ("lime", "test")}
    imp = res.importances()
    assert imp.shape == (10, 7) and list(imp.index) == list(d.X.columns)
    assert res[("shap", "test")].local.shape == (20, 10)


@pytest.mark.parametrize("model_name", ["rf", "hgb"])
def test_tree_shap_matches_exact_enumeration(model_name):
    """Интервенционный TreeExplainer = точный перебор коалиций (в т.ч. при взаимодействии)."""
    d = make_synthetic("interaction", n=600, r2=0.5, n_features=6, seed=1)
    pipe = make_pipeline(model_name, other_cols=list(d.X.columns), random_state=0)
    Z_train, Z_test = fit_transform(pipe, d.X.iloc[:500], d.y[:500], d.X.iloc[500:])
    model = pipe.named_steps["model"]
    bg = Z_train.iloc[:40]
    sh = E.shap_values(model, model_name, bg, Z_test.iloc[:8], "test")
    exact = np.array([E.exact_interventional_shap(model, x, bg) for x in Z_test.iloc[:8].to_numpy()])
    assert np.abs(sh.local.to_numpy() - exact).max() < 1e-6


def test_exact_shap_on_product():
    """f = a·b с фоном нулевого среднего: каждому признаку половина произведения."""
    class Prod:
        def predict(self, Z):
            return Z[:, 0] * Z[:, 1]

    rng = np.random.default_rng(0)
    bg = rng.normal(size=(400, 3))
    bg -= bg.mean(axis=0)
    phi = E.exact_interventional_shap(Prod(), np.array([2.0, 3.0, 5.0]), bg)
    # Центрированный фон: v(∅) = E[ab] ≈ 0, v({a}) = a·E b = 0, v({a,b}) = ab, поэтому φ_a = ½(ab − E[ab]) ≈ 3
    assert np.allclose(phi[:2], 3.0, atol=0.15) and abs(phi[2]) < 1e-12


def test_rf_shap_additivity_rational_ratios():
    """Регрессия: доли вида k/n в бутстрепе давали пороги RF почти на значениях данных.

    Без float32_safe_thresholds сумма SHAP расходилась с прогнозом (на реальных
    данных до 0.02)."""
    import shap

    rng = np.random.default_rng(0)
    n = 800
    k, m = rng.integers(0, 6, n), rng.integers(1, 9, n)
    ratio = np.minimum(k / m, 1.0)
    X = pd.DataFrame({"ratio": ratio, "skew": rng.exponential(1, n) ** 3, "x": rng.normal(size=n)})
    y = ratio + 0.3 * X["x"] + rng.normal(0, 1, n)
    pipe = make_pipeline("rf", other_cols=list(X.columns), random_state=0)
    Z_train, Z_test = fit_transform(pipe, X.iloc[:700], y[:700], X.iloc[700:])
    model = pipe.named_steps["model"]
    bg = Z_train.iloc[:100]
    raw = shap.TreeExplainer(model, data=bg.to_numpy(), feature_perturbation="interventional")
    v = raw.shap_values(Z_test.to_numpy(), check_additivity=False)
    raw_err = np.abs(raw.expected_value + v.sum(1) - model.predict(Z_test.to_numpy())).max()
    sh = E.shap_values(model, "rf", bg, Z_test, "test")  # с исправлением
    err = sh.info["expected_value"] + sh.local.sum(axis=1) - model.predict(Z_test.to_numpy())
    assert np.abs(err).max() < 1e-6
    # прогноз внутренней модели shap после правки порогов совпадает с sklearn
    fixed = shap.TreeExplainer(model, data=bg.to_numpy(), feature_perturbation="interventional")
    E.float32_safe_thresholds(fixed)
    assert np.allclose(fixed.model.predict(Z_test.to_numpy()), model.predict(Z_test.to_numpy()), atol=1e-12)
    print("ошибка без исправления:", raw_err)

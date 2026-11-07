"""Методы на моделях, где ответ известен аналитически."""

import numpy as np
import pandas as pd
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

"""Полусинтетика на искусственной таблице с именами реальных признаков (без персональных данных)."""

import numpy as np
import pandas as pd
import pytest

from xai_consistency import explain as E
from xai_consistency.data import FEATURES, FEATURES_FLAG, FEATURES_LOG, FEATURES_OTHER
from xai_consistency.models import fit_transform, make_pipeline
from xai_consistency.semisynthetic import TWIN, FoldTruth, Truth, make_target, to_explain_space


def fake_X(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(index=range(n))
    base = rng.normal(size=n)
    for c in FEATURES:
        if c in FEATURES_FLAG:
            X[c] = (rng.normal(size=n) + 0.5 * base > 0.3).astype(float)
        elif c in FEATURES_LOG:
            X[c] = np.round(np.exp(3 + 0.8 * rng.normal(size=n) + 0.4 * base))
        else:
            X[c] = rng.beta(2, 5, size=n)
    X.loc[: n // 3, "vk_counter_gifts"] = np.nan  # пропуски, как в реальных данных
    return X[FEATURES]


@pytest.fixture(scope="module")
def setup():
    X = fake_X()
    T = to_explain_space(X)
    truth = Truth(T)
    y, signal = make_target(truth, T, r2=0.5, seed=1)
    pipe = make_pipeline("ridge", FEATURES_LOG, FEATURES_OTHER)
    Z_train, Z_test = fit_transform(pipe, X.iloc[:300], y[:300], X.iloc[300:])
    return X, T, truth, y, signal, pipe, Z_train, Z_test


def test_truth_is_standardized_and_r2(setup):
    X, T, truth, y, signal, *_ = setup
    assert truth.f(T).std() == pytest.approx(1)
    assert signal.var() == pytest.approx(0.5)
    assert TWIN[0] in truth.relevant and TWIN[1] not in truth.relevant
    assert truth.sign["vk_counter_audios"] == -1 and truth.sign["vk_counter_photos"] == 0


def test_truth_rejects_missing_features():
    X = fake_X()
    T = to_explain_space(X)
    from xai_consistency.semisynthetic import Term
    with pytest.raises(ValueError):
        Truth(T, terms=(Term("main", ("vk_counter_gifts",)),))


def test_fold_truth_matches_truth_on_raw_values(setup):
    X, T, truth, y, signal, pipe, Z_train, Z_test = setup
    ft = FoldTruth(truth, pipe, Z_test.columns, r2=0.5)
    # Z фолда → T → f совпадает с f, посчитанной прямо по исходным значениям
    assert np.allclose(ft.predict(Z_test.to_numpy()), signal[300:], atol=1e-5)


def test_true_shap_additivity_and_zero_outside_f(setup):
    X, T, truth, y, signal, pipe, Z_train, Z_test = setup
    ft = FoldTruth(truth, pipe, Z_test.columns, r2=0.5)
    bg = Z_train.iloc[:50]
    phi = ft.true_shap(Z_test.iloc[:20].reset_index(drop=True), bg)
    pred = ft.predict(Z_test.iloc[:20].to_numpy())
    assert np.allclose(phi.sum(axis=1), pred - ft.predict(bg.to_numpy()).mean(), atol=1e-8)
    others = [c for c in phi.columns if c not in truth.relevant]
    assert (phi[others] == 0).all().all()


def test_true_function_as_model_pdp_equals_component(setup):
    """Для аддитивной части PDP истинной функции = её главный эффект."""
    X, T, truth, y, signal, pipe, Z_train, Z_test = setup
    ft = FoldTruth(truth, pipe, Z_test.columns, r2=0.5)
    grids = E.make_grids(Z_test, ["vk_counter_friends", "vk_post_avg_likes"])
    pdp = E.partial_dependence(ft, Z_test, grids)
    for f, c in pdp.curves.items():
        v = c["value"] - c["value"].mean()
        t = ft.main_curve(f, c["x"].to_numpy())
        assert np.allclose(v, t - t.mean(), atol=1e-6)

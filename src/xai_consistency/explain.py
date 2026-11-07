"""Обёртки над пятью методами объяснения с единым форматом результата.

Все методы работают с обученной моделью `model` (последний шаг Pipeline) и
матрицами Z после предобработки (DataFrame с именами колонок).

Единый формат: `Explanation`
- importance: глобальная важность по признакам (pd.Series). Единицы у методов
  разные, поэтому сравниваем только ранги и знаки.
- sign: направление эффекта +1/−1/0 (у PI знака нет, там NaN).
- local: локальные объяснения (объекты × признаки): значения SHAP или
  коэффициенты LIME. У глобальных методов None.
- curves: кривые эффекта {признак: DataFrame[x, value, count]} на общей
  квантильной сетке. Есть у PDP, ALE и SHAP (зависимость SHAP).
- info: дополнительные величины (разброс PI, локальный R² LIME и т.п.).
- seconds: время работы.
"""

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy import stats
from sklearn.metrics import r2_score

from .models import base_features, fit_transform, make_pipeline

ALL_METHODS = ("pi", "pdp", "ale", "shap", "lime")
N_BINS = 20


@dataclass
class Explanation:
    method: str
    split: str
    importance: pd.Series
    sign: pd.Series
    local: pd.DataFrame | None = None
    curves: dict | None = None
    info: dict = field(default_factory=dict)
    seconds: float = 0.0


# ---------------------------------------------------------------- сетка

def quantile_bins(z, n_bins=N_BINS):
    """Границы интервалов по квантилям z. None, если различных границ меньше трёх.

    Совпадающие квантили (много одинаковых значений, например нулей) склеиваются,
    поэтому интервалов может быть меньше n_bins.
    """
    edges = np.unique(np.quantile(np.asarray(z, float), np.linspace(0, 1, n_bins + 1)))
    return edges if len(edges) >= 3 else None


def assign_bins(z, edges):
    """Номер интервала для каждого значения: (e_k, e_{k+1}], первый включает e_0."""
    idx = np.searchsorted(edges, np.asarray(z, float), side="left") - 1
    return np.clip(idx, 0, len(edges) - 2)


def make_grids(Z, features, n_bins=N_BINS):
    """Общая сетка для PDP, ALE и зависимости SHAP: {признак: границы}."""
    grids = {}
    for f in features:
        edges = quantile_bins(Z[f], n_bins)
        if edges is not None:
            grids[f] = edges
    return grids


def _curve_frame(edges, values, counts):
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"x": centers, "value": values, "count": counts})


def _weighted_center(values, weights):
    ok = ~np.isnan(values) & (weights > 0)
    return values - np.sum(values[ok] * weights[ok]) / np.sum(weights[ok])


def curve_sign(curve):
    """Знак тренда кривой: знак взвешенной корреляции значения с номером интервала."""
    df = curve.dropna()
    if len(df) < 3 or df["value"].std() == 0:
        return 0
    k = np.arange(len(df), dtype=float)
    w = df["count"].to_numpy(float)
    v = df["value"].to_numpy(float)
    kc, vc = k - np.average(k, weights=w), v - np.average(v, weights=w)
    return int(np.sign(np.sum(w * kc * vc)))


def curve_strength(curve):
    """Взвешенное SD центрированной кривой: размах эффекта в единицах прогноза."""
    df = curve.dropna()
    w = df["count"].to_numpy(float)
    v = _weighted_center(df["value"].to_numpy(float), w)
    return float(np.sqrt(np.sum(w * v**2) / np.sum(w)))


def _from_curves(method, split, curves, names, seconds, info=None):
    imp = pd.Series(np.nan, index=names)
    sign = pd.Series(np.nan, index=names)
    for f, c in curves.items():
        imp[f] = curve_strength(c)
        sign[f] = curve_sign(c)
    return Explanation(method, split, imp, sign, curves=curves, info=info or {}, seconds=seconds)


# ---------------------------------------------------------------- PI

def permutation_importance(model, Z, y, n_repeats=10, seed=0, groups=None, split="test"):
    """Падение R² при перемешивании признака (или группы признаков целиком).

    groups: {имя: [колонки]} для перестановки группы как одного признака (E4).
    Все повторы одного признака склеиваются в одну матрицу: один вызов predict.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    Zv = Z.to_numpy()
    n = len(Zv)
    base = r2_score(y, model.predict(Zv))
    groups = groups or {c: [c] for c in Z.columns}
    cols = {c: i for i, c in enumerate(Z.columns)}
    mean, sd = {}, {}
    for name, members in groups.items():
        idx = [cols[m] for m in members]
        big = np.tile(Zv, (n_repeats, 1))
        for r in range(n_repeats):
            perm = rng.permutation(n)
            big[r * n:(r + 1) * n, idx] = Zv[perm][:, idx]
        pred = model.predict(big).reshape(n_repeats, n)
        drops = np.array([base - r2_score(y, p) for p in pred])
        mean[name], sd[name] = drops.mean(), drops.std()
    imp = pd.Series(mean)
    return Explanation("pi", split, imp, pd.Series(np.nan, index=imp.index),
                       info={"sd": pd.Series(sd), "base_r2": base, "n_repeats": n_repeats},
                       seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------- PDP и ALE

def partial_dependence(model, Z, grids, split="test", ice=False):
    """PDP в центрах квантильных интервалов: подставляем значение всем объектам."""
    t0 = time.perf_counter()
    Zv = Z.to_numpy()
    n = len(Zv)
    curves, ices = {}, {}
    for f, edges in grids.items():
        j = Z.columns.get_loc(f)
        centers = (edges[:-1] + edges[1:]) / 2
        big = np.tile(Zv, (len(centers), 1))
        big[:, j] = np.repeat(centers, n)
        pred = model.predict(big).reshape(len(centers), n)
        counts = np.bincount(assign_bins(Zv[:, j], edges), minlength=len(centers))
        curves[f] = _curve_frame(edges, pred.mean(axis=1), counts)
        if ice:
            ices[f] = pred.T  # объекты × точки сетки
    info = {"ice": ices} if ice else {}
    return _from_curves("pdp", split, curves, list(Z.columns), time.perf_counter() - t0, info)


def accumulated_local_effects(model, Z, grids, split="test"):
    """ALE первого порядка (Apley, Zhu 2020) на тех же квантильных интервалах.

    В интервале (e_k, e_{k+1}] берём только объекты, попавшие в него, и считаем
    среднюю разницу прогноза между правой и левой границей. Накопленная сумма
    даёт кривую в границах, значение в центре интервала — среднее двух границ.
    Центрирование по числу объектов в интервалах.
    """
    t0 = time.perf_counter()
    Zv = Z.to_numpy()
    curves = {}
    for f, edges in grids.items():
        j = Z.columns.get_loc(f)
        k = assign_bins(Zv[:, j], edges)
        lo, hi = Zv.copy(), Zv.copy()
        lo[:, j], hi[:, j] = edges[k], edges[k + 1]
        diff = model.predict(hi) - model.predict(lo)
        nb = len(edges) - 1
        counts = np.bincount(k, minlength=nb)
        sums = np.bincount(k, weights=diff, minlength=nb)
        local = np.divide(sums, counts, out=np.zeros(nb), where=counts > 0)
        at_edges = np.concatenate([[0.0], np.cumsum(local)])
        centers_val = (at_edges[:-1] + at_edges[1:]) / 2
        curves[f] = _curve_frame(edges, _weighted_center(centers_val, counts.astype(float)), counts)
    return _from_curves("ale", split, curves, list(Z.columns), time.perf_counter() - t0)


# ---------------------------------------------------------------- SHAP

def shap_values(model, model_name, Z_background, Z_explain, split, grids=None):
    """Интервенционный SHAP: LinearExplainer для Ridge, TreeExplainer для деревьев.

    Фон — выборка из обучающей части. Важность = среднее |SHAP|, знак — знак
    корреляции Спирмена значения признака с его вкладом. Кривая зависимости —
    среднее значение SHAP в каждом квантильном интервале сетки.
    """
    t0 = time.perf_counter()
    bg = Z_background.to_numpy()
    if model_name == "ridge":
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)))
        values = explainer.shap_values(Z_explain.to_numpy())
    else:
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        values = explainer.shap_values(Z_explain.to_numpy(), check_additivity=True)
    local = pd.DataFrame(values, columns=Z_explain.columns)
    imp = local.abs().mean()
    sign = pd.Series({c: _spearman_sign(Z_explain[c], local[c]) for c in local})
    curves = {}
    for f, edges in (grids or {}).items():
        k = assign_bins(Z_explain[f].to_numpy(), edges)
        nb = len(edges) - 1
        counts = np.bincount(k, minlength=nb)
        sums = np.bincount(k, weights=local[f].to_numpy(), minlength=nb)
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        curves[f] = _curve_frame(edges, mean, counts)
    return Explanation("shap", split, imp, sign, local=local, curves=curves,
                       info={"expected_value": float(np.ravel(explainer.expected_value)[0])},
                       seconds=time.perf_counter() - t0)


def _spearman_sign(x, v):
    if np.std(x) == 0 or np.std(v) == 0:
        return 0
    return int(np.sign(stats.spearmanr(x, v)[0]))


# ---------------------------------------------------------------- LIME

def lime_values(model, Z_train, Z_explain, split, seed=0, num_samples=5000,
                kernel_width_factor=0.75):
    """LIME без дискретизации, соседи сэмплируются вокруг объекта.

    Ширина ядра = kernel_width_factor · sqrt(d) (0.75 — значение по умолчанию в lime).
    Генератор случайных чисел задаётся для каждого объекта отдельно (seed + номер),
    поэтому результат не зависит от порядка и состава объясняемых объектов.

    local — коэффициенты локальной линейной модели (на 1 SD признака в обучении).
    info['contrib'] — вклад = коэффициент · (z − среднее)/SD, аналог SHAP.
    info['local_r2'] — R² локальной модели на взвешенных соседях.
    """
    t0 = time.perf_counter()
    d = Z_train.shape[1]
    explainer = LimeTabularExplainer(
        Z_train.to_numpy(), mode="regression", feature_names=list(Z_train.columns),
        discretize_continuous=False, sample_around_instance=True,
        kernel_width=kernel_width_factor * np.sqrt(d), feature_selection="none",
        random_state=seed,
    )
    coefs, r2s, preds = [], [], []
    for i, row in enumerate(Z_explain.to_numpy()):
        explainer.random_state = np.random.RandomState(seed * 100_003 + i)
        exp = explainer.explain_instance(row, model.predict, num_features=d, num_samples=num_samples)
        w = dict(exp.as_map()[1])
        coefs.append([w.get(j, 0.0) for j in range(d)])
        r2s.append(exp.score)
        preds.append(exp.local_pred[0] if np.ndim(exp.local_pred) else exp.local_pred)
    local = pd.DataFrame(coefs, columns=Z_explain.columns)
    scaled = (Z_explain.to_numpy() - explainer.scaler.mean_) / explainer.scaler.scale_
    contrib = local * scaled
    imp = local.abs().mean()
    sign = np.sign(local.mean()).astype(int)
    return Explanation("lime", split, imp, sign, local=local,
                       info={"contrib": contrib, "local_r2": pd.Series(r2s), "local_pred": np.array(preds),
                             "num_samples": num_samples, "kernel_width_factor": kernel_width_factor},
                       seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------- всё вместе

@dataclass
class ModelExplanations:
    """Результат fit_and_explain: модель, качество и объяснения по (метод, часть)."""
    model_name: str
    r2_train: float
    r2_test: float
    features: list
    grids: dict
    explanations: dict  # {(method, split): Explanation}
    seconds_fit: float

    def __getitem__(self, key):
        return self.explanations[key]

    def importances(self):
        """Таблица важностей: признаки × (метод, часть)."""
        return pd.DataFrame({f"{m}_{s}": e.importance for (m, s), e in self.explanations.items()})

    def timings(self):
        return pd.Series({f"{m}_{s}": e.seconds for (m, s), e in self.explanations.items()})


def fit_and_explain(model_name, X_train, y_train, X_test, y_test, log_cols=(), other_cols=None,
                    indicator_cols=(), curve_features=None, methods=ALL_METHODS, n_explain=200,
                    n_background=100, n_bins=N_BINS, pi_repeats=10, lime_samples=5000,
                    lime_kernel_factor=0.75, seed=0):
    """Обучает Pipeline и считает объяснения.

    Где считаем (docs/methodology.md, раздел 6):
    - PI, PDP, ALE — на тесте;
    - SHAP и LIME — отдельно на обучении и на тесте (до n_explain объектов в каждой части).
    Сетка для кривых строится по квантилям теста и общая для PDP, ALE и SHAP.
    """
    if other_cols is None:
        other_cols = [c for c in X_train.columns if c not in log_cols]
    t0 = time.perf_counter()
    pipe = make_pipeline(model_name, log_cols, other_cols, indicator_cols, random_state=seed)
    Z_train, Z_test = fit_transform(pipe, X_train, y_train, X_test, indicator_cols)
    model = pipe.named_steps["model"]
    seconds_fit = time.perf_counter() - t0
    names = list(Z_train.columns)
    curve_features = base_features(names) if curve_features is None else list(curve_features)
    grids = make_grids(Z_test, curve_features, n_bins)

    rng = np.random.default_rng(seed)
    sample = {
        "train": Z_train.iloc[np.sort(rng.choice(len(Z_train), min(n_explain, len(Z_train)), replace=False))],
        "test": Z_test.iloc[np.sort(rng.choice(len(Z_test), min(n_explain, len(Z_test)), replace=False))],
    }
    sample = {k: v.reset_index(drop=True) for k, v in sample.items()}
    background = Z_train.iloc[rng.choice(len(Z_train), min(n_background, len(Z_train)), replace=False)]

    out = {}
    if "pi" in methods:
        out[("pi", "test")] = permutation_importance(model, Z_test, y_test, pi_repeats, seed)
    if "pdp" in methods:
        out[("pdp", "test")] = partial_dependence(model, Z_test, grids)
    if "ale" in methods:
        out[("ale", "test")] = accumulated_local_effects(model, Z_test, grids)
    for split in ("train", "test"):
        if "shap" in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[("shap", split)] = shap_values(model, model_name, background, sample[split], split, grids)
        if "lime" in methods:
            out[("lime", split)] = lime_values(model, Z_train, sample[split], split, seed,
                                               lime_samples, lime_kernel_factor)
    return ModelExplanations(
        model_name=model_name,
        r2_train=r2_score(y_train, model.predict(Z_train.to_numpy())),
        r2_test=r2_score(y_test, model.predict(Z_test.to_numpy())),
        features=names, grids=grids, explanations=out, seconds_fit=seconds_fit,
    )


def null_importances(model_name, X_train, y_train, X_test, y_test, n_perm=50, seed=0,
                     n_jobs=1, **kwargs):
    """Нулевой эталон: те же модель и объяснения на перемешанной цели.

    Цель перемешивается отдельно в обучении и в тесте, связь с X пропадает.
    Возвращает {"метод_часть": DataFrame (перестановки × признаки)} и R² на тесте.
    """
    from joblib import Parallel, delayed

    def one(i):
        r = np.random.default_rng(seed + 10_000 + i)
        res = fit_and_explain(model_name, X_train, r.permutation(y_train), X_test,
                              r.permutation(y_test), seed=seed + i, **kwargs)
        return res.importances(), res.r2_test

    runs = Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(n_perm))
    tables = {col: pd.DataFrame([imp[col] for imp, _ in runs]).reset_index(drop=True)
              for col in runs[0][0].columns}
    return tables, np.array([r2 for _, r2 in runs])

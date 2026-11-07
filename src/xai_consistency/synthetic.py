"""Синтетические данные с известной истиной.

y = sqrt(R²) · f(X) + sqrt(1 − R²) · ε,  где Var f = 1 и ε ~ N(0, 1).

Так Var y = 1, а доля объяснимой дисперсии в генеральной совокупности ровно R²
(лучшая возможная модель на тесте даёт R² около заданного). При R² = 0 цель
является чистым шумом, и вся «важность» — подгонка под шум.

f — сумма слагаемых. Каждое слагаемое центрировано и нормировано на единичное
стандартное отклонение, потом умножено на свой вес. Поэтому доли дисперсии
пропорциональны квадратам весов (для независимых признаков).

Истина для сравнения:
- true_shap — точные интервенционные значения Шепли для sqrt(R²)·f. Для
  аддитивного слагаемого g(x_j) это g(x_j) − E g; произведение x_j·x_k
  независимых признаков с нулевым средним делится пополам между ними.
- true_importance — среднее |true_shap| по объектам, в единицах y.
- true_curve — истинный главный эффект признака (центрированный). Для признака,
  входящего только в произведение, он равен нулю: в среднем эффект гасится.
- true_sign — знак монотонного тренда, 0 если тренда нет (x², x1·x2, шум).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCENARIOS = ("independent", "correlated", "interaction", "nonlinear")
R2_LEVELS = (0.5, 0.2, 0.08, 0.02, 0.0)

_FUNCS = {
    "linear": lambda x: x,
    "square": lambda x: x**2,
    "tanh": lambda x: np.tanh(2 * x),
    "step": lambda x: (x > 0.5).astype(float),
}
_REF_N = 200_000  # размер эталонной выборки для нормировки слагаемых
_REF_SEED = 12345


@dataclass
class Term:
    kind: str  # "main" или "prod"
    features: tuple  # (j,) или (j, k), номера с нуля
    func: str = "linear"
    weight: float = 1.0


@dataclass
class SyntheticData:
    X: pd.DataFrame
    y: np.ndarray
    signal: np.ndarray  # sqrt(R²)·f без шума
    true_shap: pd.DataFrame
    relevant: list
    true_sign: pd.Series
    scenario: str
    r2: float
    params: dict = field(default_factory=dict)
    _terms: list = field(default_factory=list, repr=False)
    _scales: list = field(default_factory=list, repr=False)

    @property
    def true_importance(self):
        return self.true_shap.abs().mean()

    def true_curve(self, feature, x):
        """Истинный главный эффект признака в точках x, центрированный по популяции."""
        j = list(self.X.columns).index(feature)
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        for term, (mean, coef) in zip(self._terms, self._scales):
            if term.kind == "main" and term.features[0] == j:
                out += coef * (_FUNCS[term.func](x) - mean)
        return out


def _scenario_terms(scenario):
    if scenario in ("independent", "correlated"):
        return [Term("main", (j,), "linear", w) for j, w in enumerate([1.0, 0.8, 0.6, 0.4])]
    if scenario == "interaction":
        return [Term("prod", (0, 1), weight=1.0), Term("main", (2,), "linear", 0.8),
                Term("main", (3,), "linear", 0.6)]
    if scenario == "nonlinear":
        return [Term("main", (0,), "square", 1.0), Term("main", (1,), "tanh", 0.8),
                Term("main", (2,), "step", 0.6), Term("main", (3,), "linear", 0.4)]
    raise ValueError(f"Неизвестный сценарий {scenario!r}, есть {SCENARIOS}")


def _covariance(scenario, p, rho):
    cov = np.eye(p)
    if scenario == "correlated":
        # У каждого истинного признака x1..x4 есть «двойник» x5..x8 без своего эффекта
        if p < 8:
            raise ValueError("Для сценария correlated нужно не меньше 8 признаков")
        for j in range(4):
            cov[j, j + 4] = cov[j + 4, j] = rho
    return cov


def _term_raw(term, X):
    if term.kind == "main":
        return _FUNCS[term.func](X[:, term.features[0]])
    j, k = term.features
    return X[:, j] * X[:, k]


def make_synthetic(scenario, n=1000, r2=0.5, rho=0.5, n_features=10, seed=0):
    """Одна выборка из сценария. Признаки x1..xp ~ N(0, Σ)."""
    if not 0 <= r2 <= 1:
        raise ValueError("r2 должен быть от 0 до 1")
    terms = _scenario_terms(scenario)
    cov = _covariance(scenario, n_features, rho)
    chol = np.linalg.cholesky(cov)
    for t in terms:
        if t.kind == "prod" and abs(cov[t.features]) > 0:
            raise ValueError("Произведение задано только для независимых признаков")

    # Нормировка на эталонной выборке: константы не зависят от seed и n
    ref = np.random.default_rng(_REF_SEED).standard_normal((_REF_N, n_features)) @ chol.T
    ref_parts = []
    raw_scales = []
    for t in terms:
        g = _term_raw(t, ref)
        mean, sd = g.mean(), g.std()
        raw_scales.append((mean, t.weight / sd))
        ref_parts.append(t.weight * (g - mean) / sd)
    f_sd = np.sum(ref_parts, axis=0).std()
    amp = np.sqrt(r2) / f_sd
    scales = [(mean, coef * amp) for mean, coef in raw_scales]

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)) @ chol.T
    cols = [f"x{j + 1}" for j in range(n_features)]
    shap_true = np.zeros((n, n_features))
    for t, (mean, coef) in zip(terms, scales):
        part = coef * (_term_raw(t, X) - mean)
        if t.kind == "main":
            shap_true[:, t.features[0]] += part
        else:
            # Независимые признаки с нулевым средним: вклад x_j·x_k делится поровну
            for j in t.features:
                shap_true[:, j] += part / 2
    signal = shap_true.sum(axis=1)
    y = signal + np.sqrt(1 - r2) * rng.standard_normal(n)

    relevant = sorted({cols[j] for t in terms for j in t.features}, key=cols.index) if r2 > 0 else []
    sign = pd.Series(0, index=cols, dtype=int)
    for t in terms:
        if t.kind == "main" and t.func in ("linear", "tanh", "step") and r2 > 0:
            sign[cols[t.features[0]]] = 1
    return SyntheticData(
        X=pd.DataFrame(X, columns=cols), y=y, signal=signal,
        true_shap=pd.DataFrame(shap_true, columns=cols), relevant=relevant, true_sign=sign,
        scenario=scenario, r2=r2,
        params={"n": n, "rho": rho if scenario == "correlated" else None,
                "n_features": n_features, "seed": seed},
        _terms=terms, _scales=scales,
    )

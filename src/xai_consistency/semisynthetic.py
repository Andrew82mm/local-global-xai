"""Полусинтетика (E2): настоящие признаки, цель задаём сами.

Истинная функция f задаётся в пространстве объяснения: после log1p счётчиков и
стандартизации. Константы стандартизации (среднее и SD) берутся по всем строкам
данных и фиксированы. Так f — это фиксированная функция исходных признаков и не
зависит от разбиения. Модель же объясняется в координатах Z своего фолда:
StandardScaler обучается на обучающей части. `FoldTruth` переводит Z фолда
обратно в исходные значения (после log1p) и считает f в этих координатах.

Состав f (docs/e2_results.md, раздел «Истинная функция»; веса фиксированы до прогонов):
- vk_counter_friends — линейный, +;
- vk_post_avg_likes — насыщающийся, tanh(1.5·z), +;
- has_university — бинарный флаг, +;
- vk_counter_audios — линейный, −. Его «двойник» vk_can_see_audio (r = 0.88) в f не входит;
- vk_counter_photos × vk_wall_posts_total — взаимодействие двух умеренно
  коррелированных признаков (r = 0.54).

y = sqrt(R²)·f_std + sqrt(1 − R²)·ε, где f_std — f, стандартизованная по всем
строкам, ε ~ N(0, 1). Доля дисперсии сигнала по всем строкам равна R² в среднем.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURES_LOG
from .explain import exact_interventional_shap


@dataclass(frozen=True)
class Term:
    kind: str  # "main" или "prod"
    features: tuple
    func: str = "linear"  # linear, tanh, flag
    weight: float = 1.0  # знак веса задаёт направление эффекта


TERMS = (
    Term("main", ("vk_counter_friends",), "linear", 1.0),
    Term("main", ("vk_post_avg_likes",), "tanh", 0.8),
    Term("main", ("has_university",), "flag", 0.6),
    Term("main", ("vk_counter_audios",), "linear", -0.6),
    Term("prod", ("vk_counter_photos", "vk_wall_posts_total"), weight=0.8),
)
TWIN = ("vk_counter_audios", "vk_can_see_audio")  # (в f, не в f)
R2_LEVELS = (0.2, 0.08, 0.02, 0.0)


def to_explain_space(X):
    """Исходные признаки → значения до стандартизации (log1p для счётчиков)."""
    T = X.copy()
    cols = [c for c in FEATURES_LOG if c in T]
    T[cols] = np.log1p(T[cols])
    return T


class Truth:
    """Истинная функция на значениях T (после log1p, до стандартизации)."""

    def __init__(self, T_full, terms=TERMS):
        self.terms = terms
        used = sorted({f for t in terms for f in t.features})
        if T_full[used].isna().any().any():
            raise ValueError("В f допускаются только признаки без пропусков")
        self.mu = T_full[used].mean()
        self.sd = T_full[used].std(ddof=0)
        raw = [self._raw(t, T_full) for t in terms]
        self.term_mean = [r.mean() for r in raw]
        self.term_sd = [r.std() for r in raw]
        f = sum(t.weight * (r - m) / s for t, r, m, s in zip(terms, raw, self.term_mean, self.term_sd))
        self.f_sd = f.std()
        self.relevant = used
        self.sign = {f: 0 for f in used}
        for t in terms:
            if t.kind == "main":
                self.sign[t.features[0]] = int(np.sign(t.weight))

    def _z(self, T, f):
        return (np.asarray(T[f], float) - self.mu[f]) / self.sd[f]

    def _raw(self, term, T):
        if term.kind == "prod":
            a, b = term.features
            return self._z(T, a) * self._z(T, b)
        f = term.features[0]
        if term.func == "flag":
            return np.round(np.asarray(T[f], float))
        z = self._z(T, f)
        return np.tanh(1.5 * z) if term.func == "tanh" else z

    def coef(self, i):
        """Множитель слагаемого i в f_std: weight / (SD слагаемого · SD f)."""
        return self.terms[i].weight / (self.term_sd[i] * self.f_sd)

    def term_values(self, T, i):
        """Вклад слагаемого i в f_std (центрирован по всем строкам)."""
        return self.coef(i) * (self._raw(self.terms[i], T) - self.term_mean[i])

    def f(self, T):
        return sum(self.term_values(T, i) for i in range(len(self.terms)))


def make_target(truth, T_full, r2, seed):
    """y и сигнал sqrt(R²)·f_std для одной реализации шума."""
    rng = np.random.default_rng(seed)
    signal = np.sqrt(r2) * truth.f(T_full)
    return signal + np.sqrt(1 - r2) * rng.standard_normal(len(T_full)), signal


def fold_affine(pipe):
    """Параметры стандартизации фолда: {признак: (среднее, SD)} в пространстве T."""
    ct = pipe.named_steps["prep"].named_steps["columns"]
    out = {}
    for name, trans, cols in ct.transformers_:
        if name not in ("log", "other"):
            continue
        scaler = trans.named_steps["scale"]
        for c, m, s in zip(cols, scaler.mean_, scale_or_one(scaler.scale_)):
            out[c] = (m, s)
    return out


def scale_or_one(scale):
    return np.where(scale == 0, 1.0, scale)


class FoldTruth:
    """Истинная функция sqrt(R²)·f_std в координатах Z фолда.

    Имеет predict(Z) как модель, поэтому к ней применимы те же PDP, ALE и SHAP.
    """

    def __init__(self, truth, pipe, names, r2):
        self.truth = truth
        self.names = list(names)
        self.amp = np.sqrt(r2)
        self.affine = fold_affine(pipe)

    def to_T(self, Z, cols):
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[None, :]
        return pd.DataFrame({c: Z[:, i] * self.affine[c][1] + self.affine[c][0] for i, c in enumerate(cols)})

    def predict(self, Z):
        Z = np.asarray(Z, float)
        idx = [self.names.index(f) for f in self.truth.relevant]
        return self.amp * self.truth.f(self.to_T(Z[:, idx], self.truth.relevant))

    def main_curve(self, feature, z):
        """Истинный главный эффект признака (сумма его аддитивных слагаемых) в точках z фолда."""
        T = self.to_T(np.asarray(z, float)[:, None], [feature])
        out = np.zeros(len(T))
        for i, t in enumerate(self.truth.terms):
            if t.kind == "main" and t.features[0] == feature:
                out += self.amp * self.truth.term_values(T, i)
        return out

    def true_shap(self, Z_explain, Z_background):
        """Точные интервенционные значения Шепли по тому же фону, что у объяснителя.

        Аддитивное слагаемое: φ_j = c·(g(x_j) − среднее по фону g). Произведение —
        перебором коалиций его двух признаков (совместные строки фона).
        """
        phi = pd.DataFrame(0.0, index=range(len(Z_explain)), columns=Z_explain.columns)
        for i, t in enumerate(self.truth.terms):
            cols = list(t.features)
            if t.kind == "main":
                x = self.truth.term_values(self.to_T(Z_explain[cols], cols), i)
                b = self.truth.term_values(self.to_T(Z_background[cols], cols), i)
                phi[cols[0]] += self.amp * (x - b.mean())
            else:
                term_model = _TermModel(self, i, cols)
                bg = Z_background[cols].to_numpy()
                vals = np.array([exact_interventional_shap(term_model, x, bg) for x in Z_explain[cols].to_numpy()])
                phi[cols] += vals
        return phi


class _TermModel:
    """Одно слагаемое f как функция своих признаков в координатах Z (для перебора коалиций)."""

    def __init__(self, fold_truth, i, cols):
        self.ft, self.i, self.cols = fold_truth, i, cols

    def predict(self, Z):
        T = self.ft.to_T(Z, self.cols)
        return self.ft.amp * self.ft.truth.term_values(T, self.i)

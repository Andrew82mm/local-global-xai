"""Метрики согласованности объяснений (docs/plan.md, раздел «Метрики»).

Важности разных методов измерены в разных единицах (падение R², единицы y,
коэффициент на 1 SD), поэтому сравниваются только ранги и знаки. Кривые PDP,
ALE и зависимости SHAP все в единицах прогноза, для них сравнивается и форма,
и размах (docs/methodology.md, раздел 5).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _align(a, b, features=None):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    idx = a.index.intersection(b.index) if features is None else pd.Index(features)
    df = pd.DataFrame({"a": a.reindex(idx), "b": b.reindex(idx)}).dropna()
    return df["a"], df["b"]


# ---------------------------------------------------------------- ранги

def rank_agreement(a, b, features=None):
    """ρ Спирмена и τ Кендалла (tau-b, учитывает совпадения) между двумя важностями."""
    a, b = _align(a, b, features)
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return {"spearman": np.nan, "kendall": np.nan}
    return {"spearman": float(stats.spearmanr(a, b)[0]), "kendall": float(stats.kendalltau(a, b)[0])}


def top_k(s, k):
    """k признаков с наибольшей важностью. При равенстве — порядок в индексе."""
    s = pd.Series(s, dtype=float).dropna()
    return list(s.sort_values(ascending=False, kind="stable").index[:k])


def top_k_overlap(a, b, k, features=None):
    """Доля общих признаков в top-k двух важностей: |A ∩ B| / k."""
    a, b = _align(a, b, features)
    return len(set(top_k(a, k)) & set(top_k(b, k))) / k


# ---------------------------------------------------------------- знаки

def sign_agreement(sign, reference, features=None):
    """Доля признаков, где знак совпадает с эталонным.

    Считается только по признакам с определённым эталонным знаком (≠ 0), то есть
    с монотонным эффектом. Знак 0 у проверяемого метода считается несовпадением.
    """
    s, r = pd.Series(sign, dtype=float), pd.Series(reference, dtype=float)
    idx = r.index if features is None else pd.Index(features)
    r = r.reindex(idx)
    r = r[(r != 0) & r.notna()]
    if len(r) == 0:
        return np.nan
    return float((s.reindex(r.index) == r).mean())


# ---------------------------------------------------------------- кривые

def curve_agreement(a, b, weights):
    """Сравнение двух кривых на общей сетке.

    Обе кривые центрируются с весами (число объектов в интервале). Возвращает:
    - ccc: коэффициент согласия Лина центрированных кривых,
      2·cov / (var_a + var_b). 1 — кривые совпадают, 0 — не связаны или одна
      плоская, −1 — зеркальны. Чувствителен и к форме, и к размаху.
    - pearson: корреляция, только форма (размах не важен).
    - amp_a, amp_b: взвешенное SD кривых, размах эффекта в единицах прогноза.
    - nrmsd: RMS разницы, делённое на среднее из двух размахов.
    """
    a, b, w = (np.asarray(v, float) for v in (a, b, weights))
    ok = ~np.isnan(a) & ~np.isnan(b) & (w > 0)
    a, b, w = a[ok], b[ok], w[ok] / w[ok].sum()
    if len(a) < 3:
        return {"ccc": np.nan, "pearson": np.nan, "amp_a": np.nan, "amp_b": np.nan, "nrmsd": np.nan}
    a, b = a - np.sum(w * a), b - np.sum(w * b)
    va, vb, cov = np.sum(w * a * a), np.sum(w * b * b), np.sum(w * a * b)
    amp_a, amp_b = np.sqrt(va), np.sqrt(vb)
    rmsd = np.sqrt(np.sum(w * (a - b) ** 2))
    return {
        "ccc": 2 * cov / (va + vb) if va + vb > 0 else np.nan,
        "pearson": cov / (amp_a * amp_b) if amp_a * amp_b > 0 else np.nan,
        "amp_a": amp_a, "amp_b": amp_b,
        "nrmsd": rmsd / ((amp_a + amp_b) / 2) if amp_a + amp_b > 0 else np.nan,
    }


def compare_curves(curves_a, curves_b, weights_from="a"):
    """Сравнение кривых двух методов по всем общим признакам: таблица по признакам."""
    rows = {}
    for f in set(curves_a) & set(curves_b):
        ca, cb = curves_a[f], curves_b[f]
        w = (ca if weights_from == "a" else cb)["count"]
        rows[f] = curve_agreement(ca["value"], cb["value"], w)
    return pd.DataFrame(rows).T


def true_curves(curves, truth_fn):
    """Истинные кривые на той же сетке: truth_fn(признак, x) → значения."""
    return {f: c.assign(value=truth_fn(f, c["x"].to_numpy())) for f, c in curves.items()}


# ---------------------------------------------------------------- верность истине

def fidelity(importance, true_importance, relevant, k=None):
    """Верность истине на синтетике.

    - spearman: ρ с истинной важностью по всем признакам. У шумовых признаков
      истинная важность одинакова (0), поэтому даже идеальный метод не получит 1;
    - spearman_relevant: ρ только среди истинных признаков (правильный ли порядок);
    - recall_at_k: доля истинных признаков в top-k, k = число истинных;
    - all_found: все ли истинные признаки в top-k;
    - margin: минимальная важность истинного признака, делённая на максимальную
      у шумового. Больше 1 — все истинные выше всего шума.
    """
    k = len(relevant) if k is None else k
    nan = {"spearman": np.nan, "spearman_relevant": np.nan, "recall_at_k": np.nan,
           "all_found": np.nan, "margin": np.nan}
    if not relevant:
        return nan
    imp = pd.Series(importance, dtype=float)
    found = set(top_k(imp, k)) & set(relevant)
    noise = imp.drop(index=list(relevant), errors="ignore")
    top_noise = noise.max() if len(noise) else np.nan
    margin = imp.reindex(relevant).min() / top_noise if top_noise and top_noise > 0 else np.nan
    return {"spearman": rank_agreement(imp, true_importance)["spearman"],
            "spearman_relevant": rank_agreement(imp, true_importance, relevant)["spearman"],
            "recall_at_k": len(found) / len(relevant), "all_found": float(len(found) == len(relevant)),
            "margin": float(margin)}


# ---------------------------------------------------------------- устойчивость

def stability(runs, k=5):
    """Устойчивость важностей между повторами (seed, фолды, bootstrap).

    runs: DataFrame повторы × признаки.
    - mean_spearman: средний ρ Спирмена по всем парам повторов;
    - mean_topk_jaccard: среднее |A∩B|/|A∪B| top-k по парам;
    - mean_rank_sd: среднее по признакам SD ранга между повторами;
    - rank_sd: SD ранга для каждого признака (Series).
    """
    runs = runs.dropna(axis=1, how="all")
    pairs = list(combinations(range(len(runs)), 2))
    rhos, jac = [], []
    for i, j in pairs:
        a, b = runs.iloc[i], runs.iloc[j]
        rhos.append(rank_agreement(a, b)["spearman"])
        ta, tb = set(top_k(a, k)), set(top_k(b, k))
        jac.append(len(ta & tb) / len(ta | tb))
    ranks = runs.rank(axis=1, ascending=False)
    rank_sd = ranks.std(axis=0)
    return {"mean_spearman": float(np.nanmean(rhos)) if rhos else np.nan,
            "mean_topk_jaccard": float(np.mean(jac)) if jac else np.nan,
            "mean_rank_sd": float(rank_sd.mean()), "rank_sd": rank_sd}


# ---------------------------------------------------------------- нулевой эталон

def null_pvalues(observed, null):
    """Эмпирическое p: доля перестановок, где важность не меньше наблюдаемой.

    p = (1 + #{null ≥ obs}) / (1 + n_perm), по каждому признаку отдельно.
    """
    obs = pd.Series(observed, dtype=float)
    null = null.reindex(columns=obs.index)
    ge = (null.to_numpy() >= obs.to_numpy()[None, :]).sum(axis=0)
    return pd.Series((1 + ge) / (1 + len(null)), index=obs.index)


def null_tail_pvalues(observed, null, log=True):
    """p по нормальной аппроксимации нулевого распределения каждого признака.

    Зачем: у эмпирического p разрешение 1/(1 + n_perm), и при 100 перестановках
    BH по 34 признакам не может отвергнуть одну гипотезу (нужно p ≤ 0.05/34).
    log=True: нормальное распределение для log важности (положительные меры:
    mean |SHAP|, размах кривой). log=False: на исходной шкале (PI бывает < 0).
    p = 1 − Φ((obs − среднее) / SD), односторонний: важность больше нулевой.
    """
    obs = pd.Series(observed, dtype=float)
    null = null.reindex(columns=obs.index).astype(float)
    if log:
        floor = np.nanmin(np.where(null > 0, null, np.nan))
        null, obs = np.log(null.clip(lower=floor)), np.log(obs.clip(lower=floor))
    z = (obs - null.mean()) / null.std().replace(0, np.nan)
    return pd.Series(stats.norm.sf(z), index=obs.index)


def bh_fdr(pvalues):
    """q-значения Бенджамини–Хохберга (NaN пропускаются)."""
    p = pd.Series(pvalues, dtype=float)
    ok = p.dropna()
    m = len(ok)
    if m == 0:
        return p
    order = np.argsort(ok.to_numpy())
    ranked = ok.to_numpy()[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = pd.Series(np.nan, index=p.index)
    out[ok.index[order]] = q
    return out


def null_zscore(observed, null):
    """(важность − среднее нулевых) / SD нулевых: насколько выше шума."""
    obs = pd.Series(observed, dtype=float)
    null = null.reindex(columns=obs.index)
    return (obs - null.mean()) / null.std().replace(0, np.nan)


# ---------------------------------------------------------------- локальное и глобальное

def local_global(local, global_importance, k=5):
    """Согласие локальных объяснений с глобальной важностью.

    local: объекты × признаки (SHAP или вклады LIME).
    - agg_spearman: ρ между средним |локальным| и глобальной важностью;
    - object_spearman_median, _q25, _q75: ρ для каждого объекта отдельно
      (|локальный вклад| против глобальной важности), квантили по объектам;
    - top1_in_global_topk: доля объектов, у которых главный локальный признак
      входит в глобальный top-k;
    - sign_consistency: по признакам, доля объектов со знаком вклада,
      совпадающим с преобладающим. 0.5 — вклады разных знаков поровну.
    """
    g = pd.Series(global_importance, dtype=float).reindex(local.columns)
    absl = local.abs()
    per_obj = [rank_agreement(row, g)["spearman"] for _, row in absl.iterrows()]
    gtop = set(top_k(g, k))
    top1 = absl.idxmax(axis=1)
    sgn = np.sign(local)
    major = np.sign(sgn.sum(axis=0)).replace(0, 1)
    consistency = (sgn == major).mean(axis=0)
    return {"agg_spearman": rank_agreement(absl.mean(), g)["spearman"],
            "object_spearman_median": float(np.nanmedian(per_obj)),
            "object_spearman_q25": float(np.nanpercentile(per_obj, 25)),
            "object_spearman_q75": float(np.nanpercentile(per_obj, 75)),
            "top1_in_global_topk": float(top1.isin(gtop).mean()),
            "sign_consistency": consistency}


def agreement_table(importances, features=None, k=5):
    """Попарная согласованность важностей: ρ, τ и пересечение top-k для всех пар колонок."""
    rows = []
    for a, b in combinations(importances.columns, 2):
        r = rank_agreement(importances[a], importances[b], features)
        rows.append({"a": a, "b": b, **r, f"top{k}": top_k_overlap(importances[a], importances[b], k, features)})
    return pd.DataFrame(rows)

# %% [markdown]
# # 07. E2: полусинтетика на реальных признаках
#
# Признаки настоящие (34 очищенных, все 1000 строк), цель задаём сами:
# y = sqrt(R²)·f_std + шум. Истинная функция f и дизайн зафиксированы до прогонов
# в `docs/e2_results.md` и `src/xai_consistency/semisynthetic.py`.
#
# R² = 0.2, 0.08 (уровень `extra`), 0.02, 0; Ridge, RF, HGB; 30 повторов
# (новый шум, разбиение и seed модели). Всего 360 прогонов.
#
# **Персональные данные:** выводятся только агрегаты по строкам и повторам.
# Сырые результаты (тоже только агрегаты по объектам) — `results/e2_*.csv`, кеш
# дописывается, пересчёт: `E2_RECOMPUTE=1`.
#
# Запуск: `uv run python notebooks/07_e2_semisynthetic.py` (около 10 минут без кеша).

# %%
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from xai_consistency import explain as E  # noqa: E402
from xai_consistency import metrics as M  # noqa: E402
from xai_consistency.data import CONTINUOUS, FEATURES_LOG, FEATURES_OTHER, load_real  # noqa: E402
from xai_consistency.models import MODELS  # noqa: E402
from xai_consistency.semisynthetic import R2_LEVELS, TERMS, TWIN, FoldTruth, Truth, make_target, to_explain_space  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 300)

SMOKE = os.environ.get("E2_SMOKE") == "1"
N_SEEDS = 1 if SMOKE else 30
TEST_SIZE = 0.2
N_JOBS = min(6, os.cpu_count() or 1)
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
CACHE = {k: RESULTS / f"e2{'_smoke' if SMOKE else ''}_{k}.csv" for k in ("methods", "features", "pairs", "curves", "runs")}
METHOD_COLORS = {"pi": "#2a78d6", "shap": "#eb6834", "lime": "#1baf7a", "pdp": "#eda100", "ale": "#e87ba4"}
TEST_METHODS = ["pi_test", "shap_test", "lime_test", "pdp_test", "ale_test"]
K = 6  # число истинных признаков

X, _ = load_real("extra")  # цель extra не используется, только признаки
T_FULL = to_explain_space(X)
TRUTH = Truth(T_FULL)
RELEVANT = TRUTH.relevant
INTER = list(TERMS[-1].features)
WATCH = RELEVANT + [TWIN[1]]  # истинные признаки и двойник
print("Истинные признаки:", RELEVANT, "| двойник вне f:", TWIN[1])

# %% [markdown]
# ## 1. Один прогон

# %%
def shap_curves(phi, Z, grids):
    """Зависимость SHAP (среднее значение в интервале сетки) — для истинных значений Шепли."""
    out = {}
    for f, edges in grids.items():
        k = E.assign_bins(Z[f].to_numpy(), edges)
        nb = len(edges) - 1
        counts = np.bincount(k, minlength=nb)
        sums = np.bincount(k, weights=phi[f].to_numpy(), minlength=nb)
        out[f] = pd.DataFrame({"x": (edges[:-1] + edges[1:]) / 2,
                               "value": np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), "count": counts})
    return out


def apley_ale(ft, feature, other, z):
    """Предсказание Apley и Zhu (раздел 5.3) для главного эффекта ALE слагаемого c·z_a·z_b:
    ½·ρ·c·(z_a² − 1), ρ — корреляция z_a и z_b по всем строкам."""
    i = len(TERMS) - 1
    rho = np.corrcoef(TRUTH._z(T_FULL, feature), TRUTH._z(T_FULL, other))[0, 1]
    T = ft.to_T(np.asarray(z, float)[:, None], [feature])
    za = TRUTH._z(T, feature)
    return 0.5 * rho * ft.amp * TRUTH.coef(i) * (za**2 - 1)


def run_one(r2, model_name, seed):
    t0 = time.perf_counter()
    y, _ = make_target(TRUTH, T_FULL, r2, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_train, y_train, X_test, y_test, log_cols=FEATURES_LOG,
                            other_cols=FEATURES_OTHER, curve_features=CONTINUOUS, seed=seed)
    key = {"r2": r2, "model": model_name, "seed": seed}
    ft = FoldTruth(TRUTH, res.pipe, res.features, r2)
    Z_expl = res.samples["test"]
    phi = ft.true_shap(Z_expl, res.background)
    true_imp = phi.abs().mean()
    relevant = RELEVANT if r2 > 0 else []
    true_sign = pd.Series(0, index=res.features, dtype=int)
    if r2 > 0:
        for f, s in TRUTH.sign.items():
            true_sign[f] = s

    imp = res.importances()
    methods, feats = [], []
    for (m, part), ex in res.explanations.items():
        name = f"{m}_{part}"
        avail = ex.importance.dropna()
        row = {**key, "method": name, "seconds": ex.seconds,
               **M.fidelity(avail, true_imp.reindex(avail.index), [f for f in relevant if f in avail.index], k=K)}
        if m != "pi":
            have = ex.sign.dropna().index
            row["sign"] = M.sign_agreement(ex.sign, true_sign, features=have)
        methods.append(row)
        ranks = avail.rank(ascending=False)
        for f in WATCH:
            feats.append({**key, "method": name, "feature": f,
                          "rank": ranks.get(f, np.nan), "n_features": len(avail),
                          "in_top6": float(ranks[f] <= K) if f in ranks else np.nan,
                          "importance": avail.get(f, np.nan)})
        feats.append({**key, "method": name, "feature": "twin/audios",
                      "importance": avail.get(TWIN[1], np.nan) / avail.get(TWIN[0], np.nan)})

    pairs = []
    cont = [c for c in CONTINUOUS if c in imp.index]
    for a, b in combinations(imp.columns, 2):
        pairs.append({**key, "set": "continuous20", "a": a, "b": b, **M.rank_agreement(imp[a], imp[b], cont),
                      "top6": M.top_k_overlap(imp[a], imp[b], K, cont)})
        if not (a.startswith(("pdp", "ale")) or b.startswith(("pdp", "ale"))):
            pairs.append({**key, "set": "all34", "a": a, "b": b, **M.rank_agreement(imp[a], imp[b]),
                          "top6": M.top_k_overlap(imp[a], imp[b], K)})

    # Кривые: модель и истинная функция теми же методами на той же сетке
    grids = res.grids
    sets = {"pdp": res[("pdp", "test")].curves, "ale": res[("ale", "test")].curves,
            "shap": res[("shap", "test")].curves,
            "pdp_true": E.partial_dependence(ft, res.Z_test, grids).curves,
            "ale_true": E.accumulated_local_effects(ft, res.Z_test, grids).curves,
            "shap_true": shap_curves(phi, Z_expl, grids)}
    sets["main_true"] = {f: c.assign(value=ft.main_curve(f, c["x"].to_numpy())) for f, c in sets["pdp"].items()}
    sets["apley_ale"] = {f: c.assign(value=apley_ale(ft, f, INTER[1 - INTER.index(f)], c["x"].to_numpy()))
                         for f, c in sets["pdp"].items() if f in INTER}
    compare = [("pdp", "ale"), ("pdp", "shap"), ("ale", "shap"), ("pdp", "pdp_true"), ("ale", "ale_true"),
               ("shap", "shap_true"), ("pdp", "main_true"), ("ale", "main_true"), ("shap", "main_true"),
               ("pdp_true", "ale_true"), ("ale_true", "apley_ale")]
    kind = {f: ("main" if f in RELEVANT and f not in INTER else "interaction" if f in INTER else "noise")
            for f in CONTINUOUS}
    curves = []
    for a, b in compare:
        cmp = M.compare_curves(sets[a], sets[b])
        for f, r in cmp.iterrows():
            curves.append({**key, "a": a, "b": b, "feature": f, "kind": kind[f], **r.to_dict()})

    run = {**key, "r2_train": res.r2_train, "r2_test": res.r2_test,
           "true_importance_sum": float(true_imp.sum())}
    pi_global = res[("pi", "test")].importance
    for name, local in [("shap", res[("shap", "test")].local), ("lime", res[("lime", "test")].info["contrib"])]:
        lg = M.local_global(local, pi_global, k=K)
        run[f"{name}_local_vs_pi_obj_rho"] = lg["object_spearman_median"]
        run[f"{name}_local_top1_in_pi_top6"] = lg["top1_in_global_topk"]
    # Локальные: согласие SHAP модели с истинными SHAP по объектам (агрегат)
    if r2 > 0:
        sh = res[("shap", "test")].local
        rho_obj = [M.rank_agreement(sh.iloc[i].abs(), phi.iloc[i].abs())["spearman"] for i in range(len(sh))]
        run["shap_vs_true_obj_rho"] = float(np.nanmedian(rho_obj))
    run["seconds"] = time.perf_counter() - t0
    return methods, feats, pairs, curves, run


# %% [markdown]
# ## 2. Все прогоны (или кеш)

# %%
all_jobs = [(r2, m, seed) for r2 in R2_LEVELS for m in MODELS for seed in range(N_SEEDS)]
cached = os.environ.get("E2_RECOMPUTE") != "1" and all(p.exists() for p in CACHE.values())
tables = {k: pd.read_csv(p) for k, p in CACHE.items()} if cached else {}
done = set(tables["runs"][["r2", "model", "seed"]].itertuples(index=False, name=None)) if cached else set()
jobs = [j for j in all_jobs if j not in done]
print(f"Прогонов в плане: {len(all_jobs)}, в кеше: {len(all_jobs) - len(jobs)}, считаем: {len(jobs)}")
if jobs:
    jobs.sort(key=lambda j: {"rf": 0, "hgb": 1, "ridge": 2}[j[1]])
    t0 = time.perf_counter()
    out = Parallel(n_jobs=N_JOBS)(delayed(run_one)(*j) for j in jobs)
    print(f"{len(jobs)} прогонов за {(time.perf_counter() - t0) / 60:.1f} мин на {N_JOBS} процессах")
    names = ["methods", "features", "pairs", "curves"]
    new = {n: pd.DataFrame([r for o in out for r in o[i]]) for i, n in enumerate(names)}
    new["runs"] = pd.DataFrame([o[4] for o in out])
    for k, df in new.items():
        tables[k] = pd.concat([tables[k], df], ignore_index=True) if k in tables else df
        tables[k].to_csv(CACHE[k], index=False)
methods, feats, pairs, curves, runs = (tables[k] for k in ("methods", "features", "pairs", "curves", "runs"))
print("Строк:", {k: len(v) for k, v in tables.items()})
print("Среднее время прогона, с:", runs.groupby("model")["seconds"].mean().round(1).to_dict())


def by_r2(df, value, cols, index=("model",)):
    return df.pivot_table(index=[*index, "r2"], columns="method", values=value)[cols]


# %% [markdown]
# ## 3. Качество моделей

# %%
print(runs.pivot_table(index="r2", columns="model", values="r2_test", aggfunc=["mean", "std"]).round(3))

# %% [markdown]
# ## 4. Находят ли методы истинные признаки
#
# У PI, SHAP и LIME выбор из 34 признаков, истинных 6, случайный recall@6 ≈ 0.18.
# У PDP и ALE выбор из 20 непрерывных: флаг has_university им недоступен, поэтому
# recall считается по 5 доступным истинным признакам в их top-6, случайный уровень
# 6/20 = 0.3. Сравнивать recall PDP и ALE с остальными напрямую нельзя.

# %%
pos = methods[methods["r2"] > 0]
print("recall@6:")
print(by_r2(pos, "recall_at_k", TEST_METHODS).round(2))
print("\nВсе истинные найдены (PI, SHAP, LIME):")
print(by_r2(pos, "all_found", ["pi_test", "shap_test", "lime_test"]).round(2))
print("\nmargin > 1 (все истинные выше всего шума), доля повторов:")
pos = pos.assign(margin_gt1=(pos["margin"] > 1).astype(float).where(pos["margin"].notna()))
print(by_r2(pos, "margin_gt1", TEST_METHODS).round(2))

# %%
fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.4), sharey=True)
for ax, m in zip(axes, MODELS):
    sub = methods[(methods["model"] == m) & (methods["r2"] > 0)]
    for meth in TEST_METHODS:
        s = sub[sub["method"] == meth].groupby("r2")["recall_at_k"].mean()
        ax.plot(s.index, s.values, marker="o", ms=4, lw=2, color=METHOD_COLORS[meth.split("_")[0]], label=meth)
    ax.axhline(K / 34, color="#888888", lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xticks([0.02, 0.08, 0.2])
    ax.set_xticklabels(["0.02", "0.08", "0.2"])
    ax.set_ylim(0, 1.02)
    ax.set_title(m)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("recall@6")
axes[0].legend(frameon=False, fontsize=8)
fig.supxlabel("R² сигнала")
fig.suptitle("E2: доля истинных признаков в top-6 (пунктир — случайный уровень для 34 признаков)")
fig.tight_layout()
fig.savefig(FIGURES / "e2_recall.png", dpi=110)
plt.show()

# %% [markdown]
# ## 5. Каждый истинный признак и двойник
#
# Доля повторов, где признак в top-6 метода. У двойника `vk_can_see_audio` истинная
# важность 0, поэтому для него это доля **ложных** срабатываний.

# %%
f6 = feats[feats["feature"].isin(WATCH) & feats["method"].isin(["pi_test", "shap_test", "lime_test"])]
print(f6.pivot_table(index=["feature", "model"], columns=["method", "r2"], values="in_top6").round(2).to_string())

# %%
print("PDP и ALE (только непрерывные признаки):")
f6c = feats[feats["feature"].isin([f for f in WATCH if f in CONTINUOUS]) & feats["method"].isin(["pdp_test", "ale_test"])]
print(f6c.pivot_table(index=["feature", "model"], columns=["method", "r2"], values="in_top6").round(2).to_string())

# %%
print("Важность двойника / важность vk_counter_audios (истина: 0), медиана по повторам:")
tw = feats[(feats["feature"] == "twin/audios") & feats["method"].isin(["pi_test", "shap_test", "lime_test"])]
print(tw.pivot_table(index="model", columns=["method", "r2"], values="importance", aggfunc="median").round(2).to_string())

# %% [markdown]
# ## 6. Знаки

# %%
print("Совпадение знаков с истиной (монотонные эффекты; у PDP и ALE без флага):")
print(by_r2(methods[methods["r2"] > 0], "sign", ["shap_test", "lime_test", "pdp_test", "ale_test", "shap_train", "lime_train"]).round(2))

# %% [markdown]
# ## 7. Кривые
#
# - модель против истинного главного эффекта (main) — как в E1;
# - модель против **того же метода, применённого к истинной функции** (pdp–pdp_true и т.д.);
# - истинная функция: PDP против ALE у признаков взаимодействия и сравнение ALE
#   с предсказанием Apley и Zhu ½·ρ·c·(z² − 1).

# %%
cpos = curves[curves["r2"] > 0]
print("CCC с истинным главным эффектом (признаки main):")
print(cpos[(cpos["b"] == "main_true") & (cpos["kind"] == "main")].pivot_table(
    index=["model", "r2"], columns="a", values="ccc").round(2))

# %%
print("CCC модели с тем же методом на истинной функции, по типу признака:")
own = cpos[cpos["b"].isin(["pdp_true", "ale_true", "shap_true"]) & (cpos["a"] + "_true" == cpos["b"])]
print(own.pivot_table(index=["model", "r2"], columns=["kind", "a"], values="ccc").round(2))

# %%
print("Истинная функция, признаки взаимодействия: размах PDP и ALE, их CCC, ALE против Apley–Zhu")
tr = cpos[cpos["kind"] == "interaction"]
pa = tr[(tr["a"] == "pdp_true") & (tr["b"] == "ale_true")].groupby(["feature", "r2"])[["amp_a", "amp_b", "ccc"]].mean()
pa.columns = ["размах PDP_true", "размах ALE_true", "CCC PDP–ALE"]
ap = tr[(tr["a"] == "ale_true") & (tr["b"] == "apley_ale")].groupby(["feature", "r2"])[["ccc", "pearson"]].mean()
ap.columns = ["CCC ALE–Apley", "r ALE–Apley"]
print(pa.join(ap).round(3).to_string())

# %%
print("Модель: CCC PDP–ALE по типу признака (в E1 при ρ = 0.9 было 0.92–0.99):")
mm = cpos[(cpos["a"] == "pdp") & (cpos["b"] == "ale")]
print(mm.pivot_table(index=["model", "r2"], columns="kind", values="ccc").round(2))
print("\nМодель, признаки взаимодействия: размах PDP и ALE")
print(mm[mm["kind"] == "interaction"].pivot_table(index=["model", "r2"], columns="feature", values=["amp_a", "amp_b"]).round(3))

# %%
print("Согласие кривых методов модели между собой (CCC), main / шум:")
bt = curves[curves["a"].isin(["pdp", "ale"]) & curves["b"].isin(["ale", "shap"]) & curves["kind"].isin(["main", "noise"])]
print(bt.assign(pair=bt["a"] + "–" + bt["b"]).pivot_table(index=["model", "r2"], columns=["kind", "pair"], values="ccc").round(2))

# %% [markdown]
# ## 8. Согласованность методов между собой
#
# На общих 20 непрерывных признаках. Две группы, как в E1: внутри
# SHAP, LIME, PDP, ALE и PI против остальных.

# %%
tp = pairs[(pairs["set"] == "continuous20") & pairs["a"].isin(TEST_METHODS) & pairs["b"].isin(TEST_METHODS)]
tp = tp.assign(group=np.where(tp["a"].eq("pi_test") | tp["b"].eq("pi_test"), "PI — остальные", "внутри SHAP, LIME, PDP, ALE"))
grp = tp.groupby(["model", "group", "r2"])["spearman"].mean().unstack("r2")
print(grp.round(2))
print("\nВсе 34 признака, PI–SHAP, PI–LIME, SHAP–LIME (тест):")
a34 = pairs[(pairs["set"] == "all34") & pairs["a"].isin(TEST_METHODS) & pairs["b"].isin(TEST_METHODS)]
print(a34.assign(pair=a34["a"] + "–" + a34["b"]).pivot_table(index=["model", "pair"], columns="r2", values="spearman").round(2))

# %%
fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.4), sharey=True)
for ax, m in zip(axes, MODELS):
    for g, color in [("внутри SHAP, LIME, PDP, ALE", "#eb6834"), ("PI — остальные", "#2a78d6")]:
        sub = tp[(tp["model"] == m) & (tp["group"] == g)]
        per_run = sub.groupby(["seed", "r2"])["spearman"].mean().reset_index()
        st = per_run.groupby("r2")["spearman"].agg(mean="mean", q10=lambda v: v.quantile(0.1), q90=lambda v: v.quantile(0.9))
        ax.fill_between(st.index, st["q10"], st["q90"], color=color, alpha=0.15, lw=0)
        ax.plot(st.index, st["mean"], marker="o", ms=4, lw=2, color=color, label=g)
    ax.axhline(0, color="#888888", lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xticks([0, 0.02, 0.08, 0.2])
    ax.set_xticklabels(["0", "0.02", "0.08", "0.2"])
    ax.set_title(m)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ρ Спирмена важностей (20 непрерывных)")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.supxlabel("R² сигнала")
fig.suptitle("E2: две группы методов (линия — среднее, полоса — 10–90% повторов)")
fig.tight_layout()
fig.savefig(FIGURES / "e2_agreement_groups.png", dpi=110)
plt.show()

# %% [markdown]
# ## 9. Обучение против теста, локальное против глобального

# %%
tt = pairs[(pairs["set"] == "all34") & (((pairs["a"] == "shap_train") & (pairs["b"] == "shap_test")) |
                                        ((pairs["a"] == "lime_train") & (pairs["b"] == "lime_test")))]
print("ρ обучение–тест:")
print(tt.pivot_table(index=["a", "model"], columns="r2", values="spearman").round(2))
cols = ["shap_local_vs_pi_obj_rho", "shap_local_top1_in_pi_top6", "shap_vs_true_obj_rho"]
print("\nЛокальное: медиана по объектам ρ(|SHAP|, PI), главный признак объекта в top-6 PI, ρ(|SHAP|, |истинный SHAP|):")
print(runs.groupby(["model", "r2"])[cols].mean().round(2))

# %% [markdown]
# ## 10. Дополнение: локальные объяснения и аддитивная замена взаимодействия
#
# Добавлено после первого разбора результатов (правки к выводам 4 и 8).
# Пересчитываются только SHAP, PDP и ALE. С тем же seed `fit_and_explain` даёт те же
# разбиение, модель, объясняемые объекты и фон, поэтому SHAP совпадают с основными
# прогонами (проверка: ρ по всем 34 признакам равна сохранённой `shap_vs_true_obj_rho`).
#
# **Локальное (вывод 8).** У 28 шумовых признаков истинный |φ| = 0, это сплошные связи,
# и ρ по 34 признакам в основном показывает, подняты ли 6 истинных над шумом. Поэтому
# по каждому объекту отдельно считаем:
# - ρ только по 6 истинным признакам;
# - пересечение top-3 модели и истины по всем 34 признакам и долю истинных в top-6 модели;
# - базовую линию «глобальный порядок»: та же ρ по 6 истинным, но вместо |SHAP| объекта
#   берётся средний |SHAP| модели (одинаковый для всех объектов). Если локальные объяснения
#   несут информацию об объекте сверх глобального порядка, их ρ должна быть выше.
#
# **Аддитивная замена (вывод 4, гипотеза).** Для стандартных нормальных z_a, z_b с
# корреляцией ρ лучшая аддитивная проекция z_a·z_b — это a·(z_a² + z_b²) + const,
# a = ρ/(1+ρ²). Главный эффект ALE истины ½ρ(z_a² − 1), отношение 2/(1+ρ²). Признаки
# не нормальные, поэтому проекцию считаем численно: HGB с `interaction_cst` (каждое
# дерево по одному признаку) на самой f без шума, на всех 1000 строках. Проверяем:
# (а) кривые моделей ближе к проекции, чем к ALE истины;
# (б) прогнозы обычной модели почти аддитивны по паре photos × wall_posts:
#     H² Фридмана, доля дисперсии прогноза от взаимодействия пары, у HGB ещё и сравнение
#     с аддитивной HGB на тех же данных.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from xai_consistency.models import make_model  # noqa: E402
from xai_consistency.semisynthetic import fold_affine  # noqa: E402

CACHE_ADD = {k: RESULTS / f"e2{'_smoke' if SMOKE else ''}_{k}.csv" for k in ("local", "projection")}
FEATURE_ORDER = list(X.columns)
T_FILLED = T_FULL.fillna(T_FULL.median())  # пропуски только у gifts и views, в f их нет


def additive_hgb(**kw):
    """HGB без взаимодействий: каждое дерево строится по одному признаку."""
    p = {"max_iter": 3000, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20,
         "early_stopping": False, "random_state": 0, **kw}
    return HistGradientBoostingRegressor(**p)


def fit_projection(features, target):
    m = additive_hgb(interaction_cst=[[i] for i in range(len(features))])
    m.fit(T_FILLED[features].to_numpy(), target)
    return m


F_FULL = TRUTH.f(T_FULL)
PROD_I = len(TERMS) - 1
PROD_FULL = TRUTH.term_values(T_FULL, PROD_I)
PROJ_ALL = fit_projection(FEATURE_ORDER, F_FULL)  # проекция всей f на аддитивные функции 34 признаков
PROJ_PAIR = fit_projection(INTER, PROD_FULL)       # проекция одного произведения на функции двух его признаков


def r2_of(pred, target):
    return 1 - np.var(target - pred) / np.var(target)


# Насколько аддитивна f: доля дисперсии, которую объясняет лучшая аддитивная функция
add_part = sum(TRUTH.term_values(T_FULL, i) for i in range(PROD_I))
print(f"R² аддитивной проекции f (34 признака): {r2_of(PROJ_ALL.predict(T_FILLED.to_numpy()), F_FULL):.3f}")
print(f"  для сравнения, R² одной аддитивной части f (без произведения): {r2_of(add_part, F_FULL):.3f}")
print(f"R² проекции произведения на функции двух его признаков: "
      f"{r2_of(PROJ_PAIR.predict(T_FILLED[INTER].to_numpy()), PROD_FULL):.3f}")

# Коэффициент при z² у проекции и у ALE истины (по всем строкам)
rho_pair = np.corrcoef(TRUTH._z(T_FULL, INTER[0]), TRUTH._z(T_FULL, INTER[1]))[0, 1]
c_prod = TRUTH.coef(PROD_I)
print(f"\nρ(photos, wall_posts) = {rho_pair:.3f}; нормальная теория: a/c = ρ/(1+ρ²) = {rho_pair / (1 + rho_pair**2):.3f}, "
      f"ALE: ½ρ = {rho_pair / 2:.3f}, отношение {2 / (1 + rho_pair**2):.2f}")
for j, f in enumerate(INTER):
    grid = np.quantile(T_FILLED[f], np.linspace(0.02, 0.98, 41))
    Tg = pd.DataFrame(np.tile(T_FILLED[INTER].median().to_numpy(), (len(grid), 1)), columns=INTER)
    Tg[f] = grid
    shape = PROJ_PAIR.predict(Tg.to_numpy())
    zg = (grid - TRUTH.mu[f]) / TRUTH.sd[f]
    slope = np.polyfit(zg**2, shape, 1)[0] / c_prod
    print(f"  {f}: коэффициент при z² у численной проекции / c = {slope:.3f}")


class FoldProjection:
    """Аддитивная проекция sqrt(R²)·f в координатах Z фолда (как FoldTruth)."""

    def __init__(self, pipe, names, r2):
        self.names, self.amp, self.aff = list(names), np.sqrt(r2), fold_affine(pipe)
        self.idx = [self.names.index(c) for c in FEATURE_ORDER]

    def predict(self, Z):
        Z = np.asarray(Z, float)[:, self.idx]
        T = np.column_stack([Z[:, i] * self.aff[c][1] + self.aff[c][0] for i, c in enumerate(FEATURE_ORDER)])
        return self.amp * PROJ_ALL.predict(T)


def h_statistic(model, Z, a, b):
    """H² Фридмана для пары (a, b) по объектам Z и доля дисперсии прогноза от их взаимодействия.

    PD считаются в точках самих объектов, все PD центрированы:
    H² = Σ(PD_ab − PD_a − PD_b)² / Σ PD_ab²; share = Σ(PD_ab − PD_a − PD_b)² / Σ(f̂ − mean f̂)².
    """
    Zv = np.asarray(Z, float)
    n = len(Zv)
    ia, ib = Z.columns.get_loc(a), Z.columns.get_loc(b)

    def pd_at(cols):
        big = np.tile(Zv, (n, 1))
        for c in cols:
            big[:, c] = np.repeat(Zv[:, c], n)  # блок i: всем объектам подставлено значение объекта i
        v = model.predict(big).reshape(n, n).mean(axis=1)
        return v - v.mean()

    pab, pa, pb = pd_at([ia, ib]), pd_at([ia]), pd_at([ib])
    inter = np.sum((pab - pa - pb) ** 2)
    pred = model.predict(Zv)
    return {"H2": inter / np.sum(pab**2), "share": inter / np.sum((pred - pred.mean()) ** 2)}


def run_addendum(r2, model_name, seed):
    y, _ = make_target(TRUTH, T_FULL, r2, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_train, y_train, X_test, y_test, log_cols=FEATURES_LOG,
                            other_cols=FEATURES_OTHER, curve_features=CONTINUOUS, seed=seed,
                            methods=("shap", "pdp", "ale"))
    key = {"r2": r2, "model": model_name, "seed": seed}
    ft = FoldTruth(TRUTH, res.pipe, res.features, r2)
    Z_expl = res.samples["test"]
    phi = ft.true_shap(Z_expl, res.background).abs()
    sh = res[("shap", "test")].local.abs()
    glob = sh.mean()
    true6 = RELEVANT
    rows = []
    for i in range(len(sh)):
        s, t = sh.iloc[i], phi.iloc[i]
        top_t3 = set(M.top_k(t, 3))
        rows.append({"rho_all": M.rank_agreement(s, t)["spearman"],
                     "rho_true6": M.rank_agreement(s, t, true6)["spearman"],
                     "rho_true6_global": M.rank_agreement(glob, t, true6)["spearman"],
                     "top3": len(set(M.top_k(s, 3)) & top_t3) / 3,
                     "top3_global": len(set(M.top_k(glob, 3)) & top_t3) / 3,
                     "true_in_top6": len(set(M.top_k(s, K)) & set(true6)) / K})
    loc = pd.DataFrame(rows)
    local = {**key, **{f"{c}_median": loc[c].median() for c in loc}, **{f"{c}_mean": loc[c].mean() for c in loc}}

    # Кривые: модель против проекции и против ALE истины
    proj = FoldProjection(res.pipe, res.features, r2)
    grids = res.grids
    sets = {"pdp": res[("pdp", "test")].curves, "ale": res[("ale", "test")].curves,
            "ale_true": E.accumulated_local_effects(ft, res.Z_test, grids).curves,
            "proj": E.partial_dependence(proj, res.Z_test, grids).curves}
    sets["main_true"] = {f: c.assign(value=ft.main_curve(f, c["x"].to_numpy())) for f, c in sets["pdp"].items()}
    proj_rows = []
    for a in ("pdp", "ale"):
        for b in ("ale_true", "proj", "main_true"):
            cmp = M.compare_curves(sets[a], sets[b])
            for f in INTER + ["vk_counter_friends", "vk_post_avg_likes", "vk_counter_audios"]:
                if b == "main_true" and f in INTER:
                    continue
                proj_rows.append({**key, "row": "curve", "a": a, "b": b, "feature": f, **cmp.loc[f].to_dict()})
    # Проекция против ALE истины (то, что модель «должна» выучить, если выучит только аддитивное)
    cmp = M.compare_curves(sets["proj"], sets["ale_true"])
    for f in INTER:
        proj_rows.append({**key, "row": "curve", "a": "proj", "b": "ale_true", "feature": f, **cmp.loc[f].to_dict()})

    # Насколько прогноз аддитивен по паре photos × wall_posts
    model = res.pipe.named_steps["model"]
    hs = {"model": h_statistic(model, Z_expl, *INTER), "truth": h_statistic(ft, Z_expl, *INTER)}
    extra = {}
    if model_name == "hgb":
        Z_train = pd.DataFrame(res.pipe.named_steps["prep"].transform(X_train), columns=res.features)
        add = make_model("hgb", random_state=seed).set_params(interaction_cst=[[i] for i in range(len(res.features))])
        add.fit(Z_train.to_numpy(), y_train)
        p, pa = model.predict(res.Z_test.to_numpy()), add.predict(res.Z_test.to_numpy())
        extra = {"add_vs_full_r2": 1 - np.var(p - pa) / np.var(p),
                 "r2_test_full": r2_of(p, y_test), "r2_test_add": r2_of(pa, y_test)}
    proj_rows.append({**key, "row": "additivity", "H2_model": hs["model"]["H2"], "share_model": hs["model"]["share"],
                      "H2_truth": hs["truth"]["H2"], "share_truth": hs["truth"]["share"], **extra})
    return local, proj_rows


# %%
add_jobs = [(r2, m, seed) for r2 in R2_LEVELS if r2 > 0 for m in MODELS for seed in range(N_SEEDS)]
if os.environ.get("E2_RECOMPUTE") != "1" and all(p.exists() for p in CACHE_ADD.values()):
    local_tab, proj_tab = (pd.read_csv(CACHE_ADD[k]) for k in ("local", "projection"))
else:
    add_jobs.sort(key=lambda j: {"rf": 0, "hgb": 1, "ridge": 2}[j[1]])
    t0 = time.perf_counter()
    out = Parallel(n_jobs=N_JOBS)(delayed(run_addendum)(*j) for j in add_jobs)
    print(f"Дополнение: {len(add_jobs)} прогонов за {(time.perf_counter() - t0) / 60:.1f} мин на {N_JOBS} процессах")
    local_tab = pd.DataFrame([o[0] for o in out])
    proj_tab = pd.DataFrame([r for o in out for r in o[1]])
    local_tab.to_csv(CACHE_ADD["local"], index=False)
    proj_tab.to_csv(CACHE_ADD["projection"], index=False)

# Проверка: SHAP те же, что в основных прогонах
chk = local_tab.merge(runs[["r2", "model", "seed", "shap_vs_true_obj_rho"]], on=["r2", "model", "seed"])
print("Макс. расхождение ρ по 34 признакам с основными прогонами:",
      float((chk["rho_all_median"] - chk["shap_vs_true_obj_rho"]).abs().max()))

# %%
print("Локальное, по объекту (медиана по объектам, среднее по 30 повторам):")
lc = ["rho_all_median", "rho_true6_median", "rho_true6_global_median", "top3_mean", "top3_global_mean", "true_in_top6_mean"]
print(local_tab.groupby(["model", "r2"])[lc].mean().round(2).to_string())
print("Случайный уровень: ρ 0, top-3 по 34 признакам 3/34 ≈ 0.09, истинных в top-6 6/34 ≈ 0.18")
print("\nДоля повторов, где ρ по 6 истинным у локальных выше глобальной базовой линии:")
print((local_tab["rho_true6_median"] > local_tab["rho_true6_global_median"]).groupby(
    [local_tab["model"], local_tab["r2"]]).mean().unstack().round(2))

# %%
cv = proj_tab[proj_tab["row"] == "curve"]
print("Кривые признаков взаимодействия: модель против проекции и против ALE истины (среднее по повторам)")
ci = cv[cv["feature"].isin(INTER)]
print(ci.pivot_table(index=["model", "r2"], columns=["a", "b"], values="ccc").round(2).to_string())
print("\nТо же, корреляция (только форма):")
print(ci.pivot_table(index=["model", "r2"], columns=["a", "b"], values="pearson").round(2).to_string())
print("\nРазмах кривой модели / размах эталона (взаимодействие) и для калибровки — главные эффекты:")
amp = cv.assign(ratio=cv["amp_a"] / cv["amp_b"], kind=np.where(cv["feature"].isin(INTER), "inter", "main"))
print(amp[amp["a"].isin(["pdp"])].pivot_table(index=["model", "r2"], columns=["kind", "b"], values="ratio",
                                              aggfunc="median").round(2).to_string())
print("\nДоля повторов, где CCC модели с проекцией выше, чем с ALE истины:")
w = ci[ci["b"].isin(["proj", "ale_true"]) & ci["a"].isin(["pdp", "ale"])].pivot_table(
    index=["model", "r2", "seed", "a", "feature"], columns="b", values="ccc")
print((w["proj"] > w["ale_true"]).groupby(["model", "r2", "a"]).mean().unstack("a").round(2))

# %%
ad = proj_tab[proj_tab["row"] == "additivity"]
print("Аддитивность прогноза по паре photos × wall_posts (среднее по повторам):")
print(ad.groupby(["model", "r2"])[["H2_model", "share_model", "H2_truth", "share_truth"]].mean().round(3).to_string())
print("\nHGB против аддитивной HGB на тех же данных:")
print(ad[ad["model"] == "hgb"].groupby("r2")[["add_vs_full_r2", "r2_test_full", "r2_test_add"]].mean().round(3))

# %% [markdown]
# ## Выводы
#
# Подробно с числами и сравнением с E1: `docs/e2_results.md`.
#
# 1. **Порог из E1 не переносится как есть.** На 34 реальных признаках полный набор
#    из 6 истинных почти не находится даже при R² = 0.2 (0–13%). Обнаружение
#    определяется долей сигнала отдельного признака: friends находится уже при 0.08,
#    слабые признаки (9% дисперсии f) — при 0.2 и не всегда.
# 2. **Раскол подтвердился**: SHAP, LIME, PDP и ALE согласны (ρ 0.81–0.94) даже
#    без сигнала, PI с ними нет.
# 3. **Знаки надёжнее рангов**, как в E1.
# 4. **Двойник:** HGB его не использует, RF частично, Ridge почти поровну. Методы
#    это видят: свойство модели, а не метода.
# 5. **PDP и ALE истинной функции** на взаимодействии с r = 0.54 расходятся точно
#    по Apley и Zhu, но у моделей снова совпадают: модели почти аддитивны по этой паре
#    (взаимодействие даёт 0.1–0.3% дисперсии прогноза против 15% у истины). Размах их
#    аддитивной замены меньше полной аддитивной проекции, ближе к ALE истины (раздел 10).
# 7. **Локально** порядок среди истинных признаков у объекта лучше глобального, но top-3
#    объекта по его SHAP совпадает с истинным хуже, чем глобальный top-3 модели.
# 6. **LIME** у деревьев недооценивает флаг has_university. Связь асимметрии с ложной
#    важностью — только слабое наблюдение.

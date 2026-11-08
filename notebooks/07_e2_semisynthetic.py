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
#    по Apley и Zhu, но у моделей снова совпадают: деревья выучивают аддитивную замену.
# 6. **LIME** у деревьев недооценивает флаг has_university. Асимметрия признаков
#    методы не ломает.

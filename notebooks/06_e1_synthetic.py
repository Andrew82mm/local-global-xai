# %% [markdown]
# # 06. E1: синтетика с известной истиной при разном уровне сигнала
#
# Вопрос: **при каком уровне сигнала методы ещё находят правильные признаки,
# знаки и форму эффекта, и насколько они согласны между собой?**
#
# Дизайн (docs/plan.md, E1; настройки методов — docs/methodology.md):
# - сценарии: independent, correlated (ρ = 0.5 и 0.9), interaction (x1·x2), nonlinear;
# - R² = 0.5, 0.2, 0.08, 0.02, 0 (доля дисперсии сигнала в генеральной совокупности);
# - модели Ridge, RF, HGB; n = 1000 (800 обучение, 200 тест), 10 признаков, 4 истинных;
# - 10 повторов: в каждом новая выборка, разбиение и seed модели.
#
# Всего 750 прогонов «обучить модель + все пять методов». Сырые результаты
# сохраняются в `results/e1_*.csv`, повторный запуск берёт их из кеша
# (пересчёт: удалить файлы или `E1_RECOMPUTE=1`).
#
# Запуск: `uv run python notebooks/06_e1_synthetic.py` (около 15 минут без кеша).

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
from xai_consistency.models import MODELS, fit_transform, make_pipeline  # noqa: E402
from xai_consistency.synthetic import R2_LEVELS, make_synthetic  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 300)

N = 1000
TEST_SIZE = 0.2
SMOKE = os.environ.get("E1_SMOKE") == "1"  # быстрая проба кода: 1 повтор, отдельный кеш
N_SEEDS = 1 if SMOKE else 10
N_JOBS = min(6, os.cpu_count() or 1)
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
CONFIGS = [("independent", None), ("correlated", 0.5), ("correlated", 0.9),
           ("interaction", None), ("nonlinear", None)]
LABELS = {("independent", None): "independent", ("correlated", 0.5): "correlated ρ=0.5",
          ("correlated", 0.9): "correlated ρ=0.9", ("interaction", None): "interaction",
          ("nonlinear", None): "nonlinear"}
SCEN_ORDER = list(LABELS.values())
CACHE = {k: RESULTS / f"e1{'_smoke' if SMOKE else ''}_{k}.csv" for k in ("methods", "pairs", "curves", "runs")}
# Цвета по методу фиксированы (категориальная палитра, порядок слотов)
METHOD_COLORS = {"pi": "#2a78d6", "shap": "#eb6834", "lime": "#1baf7a", "pdp": "#eda100", "ale": "#e87ba4"}
TEST_METHODS = ["pi_test", "shap_test", "lime_test", "pdp_test", "ale_test"]
GRID2 = np.linspace(-2, 2, 17)

# %% [markdown]
# ## 1. Один прогон
#
# Для каждого прогона сохраняются:
# - **methods**: по каждому методу — верность истине (recall@k, все ли найдены,
#   margin, ρ среди истинных), совпадение знаков, время;
# - **pairs**: попарная согласованность важностей методов (ρ Спирмена, τ Кендалла, top-4);
# - **curves**: по каждому признаку и паре кривых (PDP, ALE, SHAP-тест, истина) —
#   CCC, корреляция и размахи;
# - **runs**: R² модели, для interaction — доля выученного взаимодействия ĉ / c,
#   согласие локального SHAP и LIME с глобальной картиной.

# %%
def feature_kind(data, f):
    """main — есть истинный главный эффект; interaction — только через x1·x2; noise."""
    if f not in data.relevant:
        return "noise"
    return "main" if np.std(data.true_curve(f, GRID2)) > 1e-9 else "interaction"


def learned_interaction_ratio(model, Z, c_true):
    """ĉ / c: коэффициент чистого взаимодействия x1·x2 в двумерном PDP модели (ноутбук 05)."""
    Zv = Z.to_numpy()
    j1, j2 = Z.columns.get_loc("x1"), Z.columns.get_loc("x2")
    pd12 = np.zeros((len(GRID2), len(GRID2)))
    for i, a in enumerate(GRID2):
        big = np.tile(Zv, (len(GRID2), 1))
        big[:, j1] = a
        big[:, j2] = np.repeat(GRID2, len(Zv))
        pd12[i] = model.predict(big).reshape(len(GRID2), len(Zv)).mean(axis=1)
    h = pd12 - pd12.mean(axis=1, keepdims=True) - pd12.mean(axis=0, keepdims=True) + pd12.mean()
    A, B = np.meshgrid(GRID2, GRID2, indexing="ij")
    return float(np.sum(h * A * B) / np.sum((A * B) ** 2) / c_true)


def run_one(scenario, rho, r2, model_name, seed):
    t0 = time.perf_counter()
    label = LABELS[(scenario, rho)]
    data = make_synthetic(scenario, n=N, r2=r2, rho=rho or 0.5, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_train, y_train, X_test, y_test, seed=seed)
    key = {"scenario": label, "r2": r2, "model": model_name, "seed": seed}
    kinds = {f: feature_kind(data, f) for f in data.X.columns}

    methods = []
    for (m, part), ex in res.explanations.items():
        row = {**key, "method": f"{m}_{part}", "seconds": ex.seconds,
               **M.fidelity(ex.importance, data.true_importance, data.relevant)}
        row["sign"] = M.sign_agreement(ex.sign, data.true_sign) if m != "pi" else np.nan
        methods.append(row)

    imp = res.importances()
    pairs = [{**key, "a": a, "b": b, **M.rank_agreement(imp[a], imp[b]),
              "top4": M.top_k_overlap(imp[a], imp[b], 4)}
             for a, b in combinations(imp.columns, 2)]

    curve_sets = {"pdp": res[("pdp", "test")].curves, "ale": res[("ale", "test")].curves,
                  "shap": res[("shap", "test")].curves}
    curve_sets["truth"] = M.true_curves(curve_sets["pdp"], data.true_curve)
    curves = []
    for a, b in combinations(curve_sets, 2):
        cmp = M.compare_curves(curve_sets[a], curve_sets[b])
        for f, r in cmp.iterrows():
            curves.append({**key, "a": a, "b": b, "feature": f, "kind": kinds[f], **r.to_dict()})

    run = {**key, "r2_train": res.r2_train, "r2_test": res.r2_test}
    if scenario == "interaction" and r2 > 0:
        # модель и Z восстанавливаем тем же Pipeline (детерминирован при том же seed)
        pipe = make_pipeline(model_name, other_cols=list(data.X.columns), random_state=seed)
        _, Z_test = fit_transform(pipe, X_train, y_train, X_test)
        model = pipe.named_steps["model"]
        run["interaction_learned"] = learned_interaction_ratio(model, Z_test, data._scales[0][1])
    pi_global = res[("pi", "test")].importance
    for name, local in [("shap", res[("shap", "test")].local), ("lime", res[("lime", "test")].info["contrib"])]:
        lg = M.local_global(local, pi_global, k=4)
        run[f"{name}_local_vs_pi_obj_rho"] = lg["object_spearman_median"]
        run[f"{name}_local_top1_in_pi_top4"] = lg["top1_in_global_topk"]
        lg_self = M.local_global(local, local.abs().mean(), k=4)
        run[f"{name}_local_vs_own_global_obj_rho"] = lg_self["object_spearman_median"]
    run["seconds"] = time.perf_counter() - t0
    return methods, pairs, curves, run


# %% [markdown]
# ## 2. Все прогоны (или кеш)

# %%
recompute = os.environ.get("E1_RECOMPUTE") == "1" or not all(p.exists() for p in CACHE.values())
if recompute:
    jobs = [(sc, rho, r2, m, seed) for sc, rho in CONFIGS for r2 in R2_LEVELS for m in MODELS
            for seed in range(N_SEEDS)]
    # RF самый медленный: ставим его первым, чтобы процессы не простаивали в конце
    jobs.sort(key=lambda j: {"rf": 0, "hgb": 1, "ridge": 2}[j[3]])
    t0 = time.perf_counter()
    out = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(run_one)(*j) for j in jobs)
    print(f"{len(jobs)} прогонов за {(time.perf_counter() - t0) / 60:.1f} мин на {N_JOBS} процессах")
    tables = {
        "methods": pd.DataFrame([r for o in out for r in o[0]]),
        "pairs": pd.DataFrame([r for o in out for r in o[1]]),
        "curves": pd.DataFrame([r for o in out for r in o[2]]),
        "runs": pd.DataFrame([o[3] for o in out]),
    }
    for k, df in tables.items():
        df.to_csv(CACHE[k], index=False)
else:
    tables = {k: pd.read_csv(p) for k, p in CACHE.items()}
    print("Загружено из кеша:", {k: len(v) for k, v in tables.items()})
methods, pairs, curves, runs = (tables[k] for k in ("methods", "pairs", "curves", "runs"))
print("Среднее время прогона, с:", runs.groupby("model")["seconds"].mean().round(1).to_dict())


def by_r2(df, value, index=("scenario", "model"), columns="method", agg="mean"):
    t = df.pivot_table(index=[*index, "r2"], columns=columns, values=value, aggfunc=agg)
    return t.reindex(SCEN_ORDER, level=0)


# %% [markdown]
# ## 3. Качество моделей
#
# Лучшая возможная модель даёт на тесте R² около заданного. При R² = 0 гибкие
# модели уходят ниже нуля (подгонка под шум).

# %%
q = runs.pivot_table(index=["scenario", "r2"], columns="model", values="r2_test", aggfunc="mean")
print(q.reindex(SCEN_ORDER, level=0).round(3))
if "interaction_learned" in runs:
    print("\nДоля выученного взаимодействия ĉ / c (interaction), среднее по повторам:")
    print(runs.dropna(subset=["interaction_learned"]).pivot_table(
        index="r2", columns="model", values="interaction_learned").round(2))

# %% [markdown]
# ## 4. Находят ли методы истинные признаки
#
# **all_found** — доля повторов, где все 4 истинных признака в top-4.
# **recall@4** — доля истинных признаков в top-4 (среднее). Случайный порядок даёт 0.4.

# %%
print("all_found, тест:")
print(by_r2(methods[methods["r2"] > 0], "all_found")[TEST_METHODS].round(2))

# %%
print("recall@4, тест:")
print(by_r2(methods[methods["r2"] > 0], "recall_at_k")[TEST_METHODS].round(2))

# %%
print("recall@4, SHAP и LIME на обучении против теста:")
print(by_r2(methods[methods["r2"] > 0], "recall_at_k")[["shap_train", "shap_test", "lime_train", "lime_test"]].round(2))

# %%
def small_multiples(df, value, title, fname, ylim=(0, 1.02), hline=None, methods_list=TEST_METHODS):
    fig, axes = plt.subplots(len(SCEN_ORDER), len(MODELS), figsize=(11, 12), sharex=True, sharey=True)
    for i, sc in enumerate(SCEN_ORDER):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            sub = df[(df["scenario"] == sc) & (df["model"] == m)]
            for meth in methods_list:
                s = sub[sub["method"] == meth].groupby("r2")[value].mean()
                base = meth.split("_")[0]
                ax.plot(s.index, s.values, marker="o", ms=4, lw=2, color=METHOD_COLORS[base],
                        ls="-" if meth.endswith("test") else "--", label=meth)
            if hline is not None:
                ax.axhline(hline, color="#888888", lw=1, ls=":")
            ax.set_xscale("symlog", linthresh=0.02)
            ax.set_xticks([0.02, 0.08, 0.2, 0.5])
            ax.set_xticklabels(["0.02", "0.08", "0.2", "0.5"])
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(sc, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("R² сигнала")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=110)
    plt.show()


small_multiples(methods[methods["r2"] > 0], "recall_at_k",
                "E1: доля истинных признаков в top-4 (пунктир — случайный порядок)", "e1_recall.png", hline=0.4)

# %% [markdown]
# ## 5. Отделены ли истинные признаки от шума
#
# **margin > 1** — все истинные признаки важнее любого шумового. Доля повторов.

# %%
methods["margin_gt1"] = (methods["margin"] > 1).astype(float).where(methods["margin"].notna())
print(by_r2(methods[methods["r2"] > 0], "margin_gt1")[TEST_METHODS].round(2))

# %% [markdown]
# ## 6. Знаки эффектов
#
# Только признаки с монотонным истинным эффектом (у PI знака нет).

# %%
print(by_r2(methods[methods["r2"] > 0], "sign")[["shap_test", "lime_test", "pdp_test", "ale_test", "shap_train", "lime_train"]].round(2))

# %% [markdown]
# ## 7. Форма эффекта
#
# CCC кривой метода с истинным главным эффектом, только признаки с главным
# эффектом (kind = main).

# %%
truth_cmp = curves[(curves["b"] == "truth") & (curves["kind"] == "main") & (curves["r2"] > 0)]
print("CCC с истиной, среднее по признакам с эффектом и повторам:")
print(truth_cmp.pivot_table(index=["scenario", "model", "r2"], columns="a", values="ccc")
      .reindex(SCEN_ORDER, level=0)[["pdp", "ale", "shap"]].round(2))

# %%
print("Согласие кривых методов между собой (CCC), признаки с эффектом / шумовые:")
between = curves[curves["b"] != "truth"].assign(pair=lambda d: d["a"] + "–" + d["b"])
print(between[between["kind"].isin(["main", "noise"])].pivot_table(
    index=["scenario", "r2"], columns=["kind", "pair"], values="ccc").reindex(SCEN_ORDER, level=0).round(2))

# %%
print("Размах кривых (взвешенное SD) у признаков с эффектом и шумовых, HGB:")
amp = curves[(curves["b"] == "truth") & (curves["model"] == "hgb")]
print(amp.pivot_table(index=["scenario", "r2"], columns=["kind", "a"], values="amp_a")
      .reindex(SCEN_ORDER, level=0).round(3))

# %% [markdown]
# ## 8. Согласованность методов между собой
#
# Средний ρ Спирмена по всем 10 признакам между важностями пяти методов на тесте
# (10 пар). При R² = 0 истины нет, но согласие методов всё равно считается: так
# видно, согласны ли методы, когда объяснять нечего.

# %%
test_pairs = pairs[pairs["a"].isin(TEST_METHODS) & pairs["b"].isin(TEST_METHODS)]
agree = test_pairs.groupby(["scenario", "model", "r2"])[["spearman", "top4"]].mean()
print(agree.unstack("r2").reindex(SCEN_ORDER, level=0).round(2))

# %%
pair_names = test_pairs.assign(pair=lambda d: d["a"].str.replace("_test", "") + "–" + d["b"].str.replace("_test", ""))
print("ρ Спирмена по парам методов, среднее по сценариям и моделям:")
print(pair_names.pivot_table(index="pair", columns="r2", values="spearman").round(2))

# %%
fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.4), sharey=True)
scen_colors = dict(zip(SCEN_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
for ax, m in zip(axes, MODELS):
    for sc in SCEN_ORDER:
        s = agree.loc[(sc, m), "spearman"]
        ax.plot(s.index, s.values, marker="o", ms=4, lw=2, color=scen_colors[sc], label=sc)
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xticks([0, 0.02, 0.08, 0.2, 0.5])
    ax.set_xticklabels(["0", "0.02", "0.08", "0.2", "0.5"])
    ax.set_title(m)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("средний ρ между методами")
axes[0].legend(frameon=False, fontsize=8)
fig.supxlabel("R² сигнала")
fig.suptitle("E1: согласованность пяти методов (тест) против уровня сигнала")
fig.tight_layout()
fig.savefig(FIGURES / "e1_agreement.png", dpi=110)
plt.show()

# %% [markdown]
# Методы делятся на две группы. SHAP, LIME, PDP и ALE описывают, **как прогноз
# зависит от признака**, и согласны между собой почти при любом сигнале. PI на тесте
# показывает, **сколько верного выучено** (ноутбук 05): при слабом сигнале он
# расходится с остальными. Ниже ρ внутри первой группы и ρ PI с ней.

# %%
model_desc = ["shap_test", "lime_test", "pdp_test", "ale_test"]
grp = test_pairs.assign(group=np.where(test_pairs["a"].eq("pi_test") | test_pairs["b"].eq("pi_test"),
                                       "PI — остальные", "внутри SHAP, LIME, PDP, ALE"))
grp_mean = grp.groupby(["model", "group", "r2"])["spearman"].agg(["mean", "std"])
print(grp_mean["mean"].unstack("r2").round(2))

fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.4), sharey=True)
for ax, m in zip(axes, MODELS):
    for g, color in [("внутри SHAP, LIME, PDP, ALE", "#eb6834"), ("PI — остальные", "#2a78d6")]:
        sub = grp[(grp["model"] == m) & (grp["group"] == g)]
        per_run = sub.groupby(["scenario", "seed", "r2"])["spearman"].mean().reset_index()
        stats_ = per_run.groupby("r2")["spearman"].agg(["mean", lambda v: v.quantile(0.1), lambda v: v.quantile(0.9)])
        stats_.columns = ["mean", "q10", "q90"]
        ax.fill_between(stats_.index, stats_["q10"], stats_["q90"], color=color, alpha=0.15, lw=0)
        ax.plot(stats_.index, stats_["mean"], marker="o", ms=4, lw=2, color=color, label=g)
    ax.axhline(0, color="#888888", lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xticks([0, 0.02, 0.08, 0.2, 0.5])
    ax.set_xticklabels(["0", "0.02", "0.08", "0.2", "0.5"])
    ax.set_title(m)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ρ Спирмена важностей")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.supxlabel("R² сигнала")
fig.suptitle("E1: две группы методов (линия — среднее, полоса — 10–90% прогонов, все сценарии)")
fig.tight_layout()
fig.savefig(FIGURES / "e1_agreement_groups.png", dpi=110)
plt.show()

# %% [markdown]
# ## 9. Обучение против теста при отсутствии сигнала
#
# При R² = 0 любая важность — подгонка под шум. SHAP и LIME на обучении и на тесте:
# насколько различаются величины важностей, и согласны ли ранги.

# %%
zero = methods[methods["r2"] == 0]
print("Согласие SHAP обучение–тест и LIME обучение–тест (ρ), по R²:")
tt = pairs[((pairs["a"] == "shap_train") & (pairs["b"] == "shap_test")) |
           ((pairs["a"] == "lime_train") & (pairs["b"] == "lime_test"))]
print(tt.pivot_table(index=["a", "model"], columns="r2", values="spearman").round(2))

# %% [markdown]
# ## 10. Локальное против глобального
#
# Медиана по объектам ρ Спирмена между |локальным вкладом| объекта и глобальной
# важностью (PI на тесте или собственное среднее |вклад|).

# %%
lg_cols = ["shap_local_vs_pi_obj_rho", "lime_local_vs_pi_obj_rho", "shap_local_vs_own_global_obj_rho",
           "lime_local_vs_own_global_obj_rho", "shap_local_top1_in_pi_top4"]
print(runs.groupby(["scenario", "model", "r2"])[lg_cols].mean().unstack("r2").reindex(SCEN_ORDER, level=0)
      [["shap_local_vs_pi_obj_rho", "shap_local_top1_in_pi_top4"]].round(2))

# %% [markdown]
# ## Выводы
#
# Подробно с числами: `docs/e1_results.md`.
#
# 1. **Порог обнаружения около R² = 0.08–0.2.** При R² ≥ 0.2 методы обычно находят
#    все истинные признаки. При 0.08 (уровень `extra`) полный набор находится лишь в
#    30–80% повторов, при 0.02 — в 0–30% (recall@4 0.52–0.78 при случайном 0.4).
# 2. **Две группы методов.** SHAP, LIME, PDP и ALE согласны между собой даже при
#    R² = 0 (ρ ≈ 0.8 у деревьев), потому что описывают одну модель вместе с подгонкой
#    под шум. PI на тесте с ними не согласен при отсутствии сигнала (ρ ≈ 0.1). Согласие
#    методов не доказывает, что сигнал есть.
# 3. **Знаки** находятся надёжнее всего (≥ 0.88 уже при R² = 0.02).
# 4. **Кривые** PDP, ALE и SHAP почти совпадают друг с другом, в том числе у шумовых
#    признаков. PDP и ALE **не** разошлись при ρ = 0.9: истина аддитивна (гипотеза,
#    проверить в E4).
# 5. **Провалы модели** (Ridge и x², x1·x2; RF и двойники при ρ = 0.9) одинаково
#    «проваливают» все методы.
# 6. **Взаимодействие** лучше всех находит PI, механизм в ноутбуке 05.

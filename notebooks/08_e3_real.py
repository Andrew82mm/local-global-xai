# %% [markdown]
# # 08. E3: реальные данные, `extra` и `emo`
#
# Те же 34 признака без пола, те же три модели, пять методов и настройки, что в E1 и E2.
# Дизайн и ожидания записаны до прогонов в `docs/e3_results.md`.
#
# - **Нулевой эталон:** перестановки цели на фиксированном разбиении 80/20 (seed 0), протокол
#   тот же, что у наблюдаемой модели (`null_importances`: цель перемешивается отдельно
#   в обучении и в тесте, seed модели свой у каждой перестановки). Каждый признак сравнивается
#   со своим нулевым распределением.
# - **Устойчивость:** повторная 5-кратная CV (5×5 = 25 фолдов), у каждого фолда свой seed модели.
#
# **Персональные данные:** только агрегаты по признакам и повторам. Локальные объяснения
# не сохраняются и не выводятся. Цель стандартизуется по всем строкам (среднее 0, SD 1),
# важности поэтому в единицах SD черты.
#
# Кеш: `results/e3_*.csv`, дописывается по блокам (черта, модель). Пересчёт: `E3_RECOMPUTE=1`.
# Оценка времени: `E3_TIMING=1`. Запуск: `uv run python notebooks/08_e3_real.py`.

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
from sklearn.model_selection import RepeatedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from xai_consistency import explain as E  # noqa: E402
from xai_consistency import metrics as M  # noqa: E402
from xai_consistency.data import CONTINUOUS, FEATURES_LOG, FEATURES_OTHER, load_real  # noqa: E402
from xai_consistency.models import MODELS  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 400)

TRAITS = ("extra", "emo")
SMOKE = os.environ.get("E3_SMOKE") == "1"
N_PERM = 6 if SMOKE else int(os.environ.get("E3_PERM", 100))
CV_SPLITS, CV_REPEATS = 5, (1 if SMOKE else 5)
TEST_SIZE, SPLIT_SEED = 0.2, 0
K = 6  # top-k, как в E2
ALPHA = 0.05
N_JOBS = min(6, os.cpu_count() or 1)
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
CACHE = {k: RESULTS / f"e3{'_smoke' if SMOKE else ''}_{k}.csv" for k in ("importance", "runs")}
METHODS = ["pi_test", "shap_test", "lime_test", "pdp_test", "ale_test", "shap_train", "lime_train"]
TEST_METHODS = METHODS[:5]
GROUP = ["shap_test", "lime_test", "pdp_test", "ale_test"]
METHOD_COLORS = {"pi": "#2a78d6", "shap": "#eb6834", "lime": "#1baf7a", "pdp": "#eda100", "ale": "#e87ba4"}

DATA = {}
for t in TRAITS:
    X, y = load_real(t)
    DATA[t] = (X, (y - y.mean()) / y.std())
FEATURES = list(DATA[TRAITS[0]][0].columns)
print("Строк:", len(DATA["extra"][0]), "признаков:", len(FEATURES), "из них непрерывных:", len(CONTINUOUS))

# %% [markdown]
# ## 1. Один прогон: наблюдаемый, перестановка или фолд CV

# %%
def split_for(trait, kind, idx):
    """Разбиение и seed модели для прогона. kind: obs, null, cv."""
    X, y = DATA[trait]
    if kind in ("obs", "null"):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        if kind == "null":  # как в explain.null_importances
            r = np.random.default_rng(SPLIT_SEED + 10_000 + idx)
            y_tr, y_te = r.permutation(y_tr), r.permutation(y_te)
            return X_tr, X_te, y_tr, y_te, SPLIT_SEED + idx
        return X_tr, X_te, y_tr, y_te, SPLIT_SEED
    folds = list(RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=0).split(X))
    tr, te = folds[idx]
    return X.iloc[tr], X.iloc[te], y[tr], y[te], 100 + idx


def run_job(trait, model_name, kind, idx):
    t0 = time.perf_counter()
    X_tr, X_te, y_tr, y_te, seed = split_for(trait, kind, idx)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, log_cols=FEATURES_LOG,
                            other_cols=FEATURES_OTHER, curve_features=CONTINUOUS, seed=seed)
    key = {"trait": trait, "model": model_name, "kind": kind, "idx": idx}
    rows = []
    for (m, part), ex in res.explanations.items():
        for f in res.features:
            rows.append({**key, "method": f"{m}_{part}", "feature": f,
                         "importance": ex.importance.get(f, np.nan), "sign": ex.sign.get(f, np.nan)})
    run = {**key, "seed": seed, "r2_train": res.r2_train, "r2_test": res.r2_test,
           "seconds": time.perf_counter() - t0}
    return rows, run


def jobs_for(trait, model_name):
    return ([(trait, model_name, "obs", 0)] + [(trait, model_name, "null", i) for i in range(N_PERM)]
            + [(trait, model_name, "cv", i) for i in range(CV_SPLITS * CV_REPEATS)])


# %% [markdown]
# ## 2. Оценка времени (`E3_TIMING=1`)

# %%
if os.environ.get("E3_TIMING") == "1":
    n = 2 * N_JOBS
    t0 = time.perf_counter()
    out = Parallel(n_jobs=N_JOBS)(delayed(run_job)("extra", "rf", "null", i) for i in range(n))
    wall = time.perf_counter() - t0
    per = np.mean([o[1]["seconds"] for o in out])
    total_jobs = len(TRAITS) * len(MODELS) * len(jobs_for("extra", "rf"))
    print(f"extra × rf: {n} прогонов за {wall:.0f} с на {N_JOBS} процессах, {per:.1f} с на прогон в процессе")
    # В E2 под той же нагрузкой: Ridge 6 с, RF 21 с, HGB 21 с — Ridge примерно втрое быстрее
    est = wall / n * total_jobs * (1 + 1 + 6 / 21) / 3
    print(f"Всего прогонов {total_jobs}, оценка {est / 60:.0f} мин")
    sys.exit(0)

# %% [markdown]
# ## 3. Все прогоны (или кеш)

# %%
def read_cache(path):
    """Строка "null" (kind перестановки) не должна читаться как NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


recompute = os.environ.get("E3_RECOMPUTE") == "1"
tables = {k: read_cache(p) for k, p in CACHE.items()} if not recompute and all(p.exists() for p in CACHE.values()) else {}
done = set(tables["runs"][["trait", "model", "kind", "idx"]].itertuples(index=False, name=None)) if tables else set()
t_all = time.perf_counter()
for model_name in ("rf", "hgb", "ridge"):
    for trait in TRAITS:
        jobs = [j for j in jobs_for(trait, model_name) if j not in done]
        if not jobs:
            continue
        t0 = time.perf_counter()
        out = Parallel(n_jobs=N_JOBS)(delayed(run_job)(*j) for j in jobs)
        print(f"{trait} × {model_name}: {len(jobs)} прогонов за {(time.perf_counter() - t0) / 60:.1f} мин")
        new = {"importance": pd.DataFrame([r for o in out for r in o[0]]), "runs": pd.DataFrame([o[1] for o in out])}
        for k, df in new.items():
            tables[k] = pd.concat([tables[k], df], ignore_index=True) if k in tables else df
            tables[k].to_csv(CACHE[k], index=False)
print(f"Расчёт: {(time.perf_counter() - t_all) / 60:.1f} мин")
imp, runs = tables["importance"], tables["runs"]
print("Прогонов:", runs.groupby(["trait", "model", "kind"]).size().unstack().to_string())
print("Среднее время прогона в процессе, с:", runs.groupby("model")["seconds"].mean().round(1).to_dict())

# Широкие таблицы: (черта, модель, kind, idx, метод) × признаки
# pivot, а не pivot_table(dropna=False): тот строит полное произведение уровней индекса,
# и у cv появились бы пустые строки с idx перестановок
IDX = ["trait", "model", "kind", "idx", "method"]
wide = imp.pivot(index=IDX, columns="feature", values="importance").reindex(columns=FEATURES).sort_index()
signs = imp.pivot(index=IDX, columns="feature", values="sign").reindex(columns=FEATURES).sort_index()
n_rows = wide.groupby(level=["trait", "model", "kind", "method"]).size()
assert n_rows.xs("cv", level="kind").eq(CV_SPLITS * CV_REPEATS).all() and n_rows.xs("null", level="kind").eq(N_PERM).all()


def get(trait, model_name, kind, method):
    """Таблица повторы × признаки (только признаки, у которых метод определён)."""
    t = wide.loc[(trait, model_name, kind)].xs(method, level="method")
    return t.dropna(axis=1, how="all")


# %% [markdown]
# ## 4. Качество моделей: наблюдаемый R² против перестановок и на CV

# %%
rows = []
for (trait, model_name), g in runs.groupby(["trait", "model"]):
    obs = g.loc[g["kind"] == "obs", "r2_test"].iloc[0]
    null = g.loc[g["kind"] == "null", "r2_test"].to_numpy()
    cv = g.loc[g["kind"] == "cv", "r2_test"]
    rows.append({"trait": trait, "model": model_name, "R² тест (разбиение 0)": obs,
                 "нуль: среднее": null.mean(), "нуль: 95%": np.quantile(null, 0.95),
                 "p": (1 + (null >= obs).sum()) / (1 + len(null)),
                 "CV: среднее": cv.mean(), "CV: SD по фолдам": cv.std(),
                 "CV: доля фолдов > 0": (cv > 0).mean()})
quality = pd.DataFrame(rows).set_index(["trait", "model"])
print(quality.round(3).to_string())

# %% [markdown]
# ## 5. Важность каждого признака против его собственного нуля
#
# Для каждого (черта, модель, метод, признак): процентиль наблюдаемой важности среди
# перестановок, эмпирическое p = (1 + #{нуль ≥ набл.})/(1 + n), BH по признакам метода
# (34 у PI, SHAP, LIME; 20 непрерывных у PDP и ALE). Основной критерий «выделился» —
# q < 0.05 по хвостовому p (нормальная аппроксимация нуля, у PI на исходной шкале,
# у остальных на log): у эмпирического p при 100 перестановках не хватает разрешения
# (минимум 1/101, а BH для одного открытия из 34 требует p ≤ 0.0015).

# %%
def null_test(trait, model_name, method):
    obs = get(trait, model_name, "obs", method).iloc[0]
    null = get(trait, model_name, "null", method)[obs.index]
    p_emp = M.null_pvalues(obs, null)
    p_tail = M.null_tail_pvalues(obs, null, log=not method.startswith("pi"))
    return pd.DataFrame({"obs": obs, "null_mean": null.mean(), "null_q95": null.quantile(0.95),
                         "percentile": (null.to_numpy() < obs.to_numpy()[None, :]).mean(axis=0) * 100,
                         "p_emp": p_emp, "q_emp": M.bh_fdr(p_emp), "p_tail": p_tail, "q_tail": M.bh_fdr(p_tail),
                         "z": M.null_zscore(obs, null)})


tests = pd.concat({(t, m, meth): null_test(t, m, meth) for t in TRAITS for m in MODELS for meth in METHODS},
                  names=["trait", "model", "method", "feature"]).reset_index()

# %% [markdown]
# ### 5.1 Калибровка хвостового p на самих перестановках
#
# Каждая перестановка по очереди считается «наблюдаемой», нуль — остальные. При верной
# аппроксимации доли p < 0.05 и p < 0.01 близки к 0.05 и 0.01.

# %%
def calibration(trait, model_name, method):
    null = get(trait, model_name, "null", method)
    ps = []
    for i in range(len(null)):
        rest = null.drop(index=null.index[i])
        ps.append(M.null_tail_pvalues(null.iloc[i], rest, log=not method.startswith("pi")).to_numpy())
    ps = np.concatenate(ps)
    return {"p<0.05": np.mean(ps < 0.05), "p<0.01": np.mean(ps < 0.01), "p<0.001": np.mean(ps < 0.001)}


cal = pd.DataFrame({(t, m, meth): calibration(t, m, meth) for t in TRAITS for m in MODELS for meth in TEST_METHODS}).T
cal.index.names = ["trait", "model", "method"]
print("Доля перестановок с хвостовым p ниже порога (ожидается 0.05, 0.01, 0.001):")
print(cal.groupby(level="method").agg(["mean", "max"]).round(3).to_string())

# %%
print("Число признаков, которые выделились (q_tail < 0.05), и при эмпирическом p:")
cnt = tests.assign(tail=tests["q_tail"] < ALPHA, emp=tests["q_emp"] < ALPHA, p05=tests["p_emp"] < ALPHA,
                   above_max=tests["percentile"] == 100)
print(cnt.pivot_table(index=["trait", "model"], columns="method", values="tail", aggfunc="sum")[METHODS].to_string())
print("\nq_emp < 0.05:")
print(cnt.pivot_table(index=["trait", "model"], columns="method", values="emp", aggfunc="sum")[METHODS].to_string())
print("\nНаблюдаемая важность выше всех перестановок (число признаков):")
print(cnt.pivot_table(index=["trait", "model"], columns="method", values="above_max", aggfunc="sum")[METHODS].to_string())
print("\nНескорректированное p_emp < 0.05 (число признаков; ожидание при нуле 34·0.05 = 1.7, у PDP и ALE 1.0):")
print(cnt.pivot_table(index=["trait", "model"], columns="method", values="p05", aggfunc="sum")[METHODS].to_string())

# %%
print("Какие признаки выделились (q_tail < 0.05): число комбинаций модель × метод (тест, 5 методов × 3 модели = 15):")
hits = tests[(tests["q_tail"] < ALPHA) & tests["method"].isin(TEST_METHODS)]
print(hits.groupby(["trait", "feature"]).size().sort_values(ascending=False).to_string())

# %%
print("Признаки, выделившиеся хотя бы раз: z относительно нуля и процентиль (тест):")
top_feats = {t: hits.loc[hits["trait"] == t, "feature"].value_counts().index.tolist() for t in TRAITS}
for t in TRAITS:
    if not top_feats[t]:
        print(f"{t}: ни одного")
        continue
    sub = tests[(tests["trait"] == t) & tests["feature"].isin(top_feats[t]) & tests["method"].isin(TEST_METHODS)]
    print(f"\n{t}: z")
    print(sub.pivot_table(index=["feature", "model"], columns="method", values="z")[TEST_METHODS].round(1).to_string())
    print(f"{t}: q_tail")
    print(sub.pivot_table(index=["feature", "model"], columns="method", values="q_tail")[TEST_METHODS].round(3).to_string())

# %%
print("Минимальное q_tail по признакам у каждой комбинации (насколько близко к порогу):")
print(tests.pivot_table(index=["trait", "model"], columns="method", values="q_tail", aggfunc="min")[METHODS].round(3).to_string())

# %% [markdown]
# ### 5.2 Масштаб модели (Ridge)
#
# RidgeCV на перемешанной цели выбирает большой α и сжимает все коэффициенты, на настоящей
# цели α меньше. Тогда абсолютная важность шумовых признаков у наблюдаемой модели выше,
# чем у нулевых, и признак «отличается от нуля» только из-за масштаба. Проверка: тот же
# тест на доле важности (важность / сумма по признакам метода).

# %%
def null_test_share(trait, model_name, method):
    obs = get(trait, model_name, "obs", method).iloc[0]
    null = get(trait, model_name, "null", method)[obs.index]
    if method.startswith("pi"):
        return None
    obs, null = obs / obs.sum(), null.div(null.sum(axis=1), axis=0)
    p = M.null_tail_pvalues(obs, null)
    return pd.DataFrame({"q_tail_share": M.bh_fdr(p)})


share = pd.concat({(t, m, meth): r for t in TRAITS for m in MODELS for meth in METHODS
                   if (r := null_test_share(t, m, meth)) is not None},
                  names=["trait", "model", "method", "feature"]).reset_index()
print("Число выделившихся признаков по доле важности (q < 0.05):")
print(share.assign(hit=share["q_tail_share"] < ALPHA).pivot_table(
    index=["trait", "model"], columns="method", values="hit", aggfunc="sum").to_string())
ridge_alpha = {}
for t in TRAITS:
    from xai_consistency.models import make_pipeline
    X_tr, X_te, y_tr, y_te, _ = split_for(t, "obs", 0)
    pipe = make_pipeline("ridge", FEATURES_LOG, FEATURES_OTHER).fit(X_tr, y_tr)
    alphas_null = []
    for i in range(20):
        _, _, y_tr_n, _, _ = split_for(t, "null", i)
        alphas_null.append(make_pipeline("ridge", FEATURES_LOG, FEATURES_OTHER).fit(X_tr, y_tr_n).named_steps["model"].alpha_)
    ridge_alpha[t] = {"alpha наблюдаемой": pipe.named_steps["model"].alpha_, "alpha нуля, медиана (20 перест.)": np.median(alphas_null)}
print(pd.DataFrame(ridge_alpha).T.round(1))

# %% [markdown]
# ### 5.3 Ridge: нуль с α, зафиксированным на наблюдаемом
#
# Добавлено после прогонов. Нулевые RidgeCV выбирают α = 10⁴ (верхняя граница сетки),
# наблюдаемая у `extra` — около 264. Тест по доле важности для Ridge оказался слабым
# (не находит даже друзей). Честнее сравнивать с нулём той же модели: перестановки
# с α, равным наблюдаемому. У `emo` наблюдаемый α тоже 10⁴, там пересчёт не нужен.

# %%
CACHE_FIXED = RESULTS / f"e3{'_smoke' if SMOKE else ''}_ridge_fixed_alpha.csv"


def run_fixed_alpha(trait, idx, alpha):
    """Нулевой прогон Ridge с фиксированным α (подмена make_model внутри процесса)."""
    from sklearn.linear_model import RidgeCV

    from xai_consistency import models as Mo
    original = Mo.make_model
    Mo.make_model = lambda name, random_state=42: RidgeCV(alphas=[alpha])
    try:
        rows, _ = run_job(trait, "ridge", "null", idx)
    finally:
        Mo.make_model = original  # процессы joblib переиспользуются
    return rows


if CACHE_FIXED.exists() and not recompute:
    fixed = read_cache(CACHE_FIXED)
else:
    a_obs = ridge_alpha["extra"]["alpha наблюдаемой"]
    t0 = time.perf_counter()
    out = Parallel(n_jobs=N_JOBS)(delayed(run_fixed_alpha)("extra", i, a_obs) for i in range(N_PERM))
    print(f"Ridge с α = {a_obs:.1f}: {N_PERM} перестановок за {(time.perf_counter() - t0) / 60:.1f} мин")
    fixed = pd.DataFrame([r for o in out for r in o])
    fixed.to_csv(CACHE_FIXED, index=False)
fw = fixed.pivot(index=["idx", "method"], columns="feature", values="importance").reindex(columns=FEATURES)
rows = {}
for meth in METHODS:
    obs = get("extra", "ridge", "obs", meth).iloc[0]
    null = fw.xs(meth, level="method")[obs.index]
    p = M.null_tail_pvalues(obs, null, log=not meth.startswith("pi"))
    rows[meth] = M.bh_fdr(p)
q_fixed = pd.DataFrame(rows)
print("extra × Ridge, нуль с фиксированным α: признаки с q_tail < 0.05")
print({m: q_fixed.index[q_fixed[m] < ALPHA].tolist() for m in METHODS})
print("q_tail для признаков, выделившихся хотя бы у одного метода:")
print(q_fixed[(q_fixed < ALPHA).any(axis=1)].round(3).to_string())

# %% [markdown]
# У Ridge mean |SHAP| ∝ |β|, а нуль для |β| близок к полунормальному. У его логарифма длинный
# левый хвост, SD на log-шкале большая, и хвостовой тест теряет мощность. Нормальная
# аппроксимация на исходной шкале, наоборот, слишком либеральна (калибровка ниже). Поэтому
# для Ridge с фиксированным α смотрим и на признаки выше всех перестановок.

# %%
for meth in ["shap_test", "pi_test"]:
    obs = get("extra", "ridge", "obs", meth).iloc[0]
    null = fw.xs(meth, level="method")[obs.index]
    p_raw = M.null_tail_pvalues(obs, null, log=False)
    t = pd.DataFrame({"набл./медиана нуля": obs / null.median(), "выше всех": obs > null.max(),
                      "p_emp": M.null_pvalues(obs, null), "p_tail_log": M.null_tail_pvalues(obs, null, log=not meth.startswith("pi")),
                      "p_tail_raw": p_raw, "q_tail_raw": M.bh_fdr(p_raw)}).sort_values("p_emp").head(6)
    print(meth)
    print(t.round(4).to_string())
null = fw.xs("shap_test", level="method")
for log in (True, False):
    ps = np.concatenate([M.null_tail_pvalues(null.iloc[i], null.drop(index=null.index[i]), log=log).to_numpy()
                         for i in range(len(null))])
    print(f"Калибровка, SHAP Ridge с фикс. α, {'log' if log else 'исходная шкала'}:",
          {t: round(float(np.mean(ps < t)), 4) for t in (0.05, 0.01, 0.001)})

# %% [markdown]
# ## 6. Устойчивость: 5×5 CV и seed моделей

# %%
def pairwise_rho(t):
    """Средний ρ Спирмена важностей между всеми парами повторов."""
    vals = [M.rank_agreement(t.iloc[i], t.iloc[j])["spearman"] for i, j in combinations(range(len(t)), 2)]
    return np.nanmean(vals)


rows = []
for t in TRAITS:
    for m in MODELS:
        for meth in METHODS:
            cv, null = get(t, m, "cv", meth), get(t, m, "null", meth)
            rows.append({"trait": t, "model": m, "method": meth, "CV": pairwise_rho(cv),
                         "нуль": pairwise_rho(null.iloc[:CV_SPLITS * CV_REPEATS])})
stab = pd.DataFrame(rows)
print("Средний ρ важностей между повторами (CV — 25 фолдов; нуль — 25 перестановок на одном разбиении):")
print(stab.pivot_table(index=["trait", "model"], columns="method", values=["CV", "нуль"])
      .reindex(columns=METHODS, level=1).round(2).to_string())

# %%
def topk_share(t):
    ranks = t.rank(axis=1, ascending=False)
    return (ranks <= K).mean()


rows = []
for t in TRAITS:
    for m in MODELS:
        for meth in TEST_METHODS:
            cv, null = get(t, m, "cv", meth), get(t, m, "null", meth)
            r_cv = cv.rank(axis=1, ascending=False)
            for f in cv.columns:
                rows.append({"trait": t, "model": m, "method": meth, "feature": f,
                             "top6_cv": (r_cv[f] <= K).mean(), "top6_null": topk_share(null)[f],
                             "rank_median": r_cv[f].median(), "rank_iqr": r_cv[f].quantile(0.75) - r_cv[f].quantile(0.25),
                             "imp_cv_mean": cv[f].mean(), "imp_cv_sd": cv[f].std()})
topk = pd.DataFrame(rows)
for t in TRAITS:
    sub = topk[topk["trait"] == t]
    order = sub.groupby("feature")["top6_cv"].mean().sort_values(ascending=False).index[:8]
    print(f"\n{t}: доля фолдов CV, где признак в top-6 (в скобках — доля перестановок), 8 самых частых:")
    s = sub[sub["feature"].isin(order)]
    tab = (s["top6_cv"].round(2).astype(str) + " (" + s["top6_null"].round(2).astype(str) + ")")
    print(s.assign(v=tab).pivot_table(index=["feature"], columns=["model", "method"], values="v", aggfunc="first")
          .reindex(order).to_string())

# %%
for t in TRAITS:
    sub = topk[(topk["trait"] == t) & topk["method"].isin(["pi_test", "shap_test"])]
    order = sub.groupby("feature")["top6_cv"].mean().sort_values(ascending=False).index[:5]
    print(f"\n{t}: медианный ранг и IQR ранга по 25 фолдам, CV важности (SD/среднее):")
    s = sub[sub["feature"].isin(order)].assign(cv_coef=lambda d: d["imp_cv_sd"] / d["imp_cv_mean"].abs())
    print(s.pivot_table(index=["feature"], columns=["model", "method"], values=["rank_median", "rank_iqr", "cv_coef"])
          .reindex(order).round(1).to_string())

# %% [markdown]
# ## 7. Знаки выделившихся признаков

# %%
for t in TRAITS:
    if not top_feats[t]:
        continue
    print(f"\n{t}: доля фолдов CV, где знак эффекта +1 (SHAP, LIME, PDP, ALE):")
    rows = []
    for m in MODELS:
        for meth in GROUP:
            s = signs.loc[(t, m, "cv")].xs(meth, level="method")
            for f in top_feats[t]:
                if f in s and s[f].notna().any():
                    rows.append({"feature": f, "model": m, "method": meth, "plus": (s[f] > 0).mean()})
    print(pd.DataFrame(rows).pivot_table(index="feature", columns=["model", "method"], values="plus").round(2).to_string())

# %% [markdown]
# ## 8. Согласие методов между собой
#
# Как в E2: пары методов на 20 непрерывных признаках. Группа SHAP, LIME, PDP, ALE и PI против
# остальных. Отдельно на фолдах CV (настоящая цель) и на перестановках (нуль).

# %%
cont = [c for c in CONTINUOUS if c in FEATURES]
rows = []
for (t, m, kind, idx), g in wide.groupby(level=["trait", "model", "kind", "idx"]):
    if kind == "obs":
        continue
    g = g.droplevel(["trait", "model", "kind", "idx"])
    for a, b in combinations(TEST_METHODS, 2):
        r = M.rank_agreement(g.loc[a], g.loc[b], cont)
        rows.append({"trait": t, "model": m, "kind": kind, "idx": idx, "a": a, "b": b, "spearman": r["spearman"]})
agree = pd.DataFrame(rows)
agree["group"] = np.where(agree["a"].eq("pi_test") | agree["b"].eq("pi_test"), "PI — остальные", "внутри SHAP, LIME, PDP, ALE")
grp = agree.groupby(["trait", "model", "kind", "group"])["spearman"].mean().unstack(["kind", "group"])
print(grp.round(2).to_string())
print("\nПо парам (CV):")
print(agree[agree["kind"] == "cv"].assign(pair=lambda d: d["a"] + "–" + d["b"])
      .pivot_table(index=["trait", "model"], columns="pair", values="spearman").round(2).to_string())

# %%
fig, axes = plt.subplots(1, len(TRAITS), figsize=(10, 3.6), sharey=True)
for ax, t in zip(axes, TRAITS):
    xs = np.arange(len(MODELS))
    for k, (g, color) in enumerate([("внутри SHAP, LIME, PDP, ALE", "#eb6834"), ("PI — остальные", "#2a78d6")]):
        for j, (kind, alpha) in enumerate([("cv", 1.0), ("null", 0.45)]):
            sub = agree[(agree["trait"] == t) & (agree["group"] == g) & (agree["kind"] == kind)]
            per = sub.groupby(["model", "idx"])["spearman"].mean()
            mean = per.groupby("model").mean().reindex(MODELS)
            q = per.groupby("model").quantile([0.1, 0.9]).unstack().reindex(MODELS)
            pos = xs + (k * 2 + j - 1.5) * 0.17
            ax.errorbar(pos, mean, yerr=[mean - q[0.1], q[0.9] - mean], fmt="o", color=color, alpha=alpha,
                        ms=5, capsize=2, label=f"{g}, {'CV' if kind == 'cv' else 'нуль'}")
    ax.axhline(0, color="#888888", lw=1, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS)
    ax.set_title(t)
    ax.grid(alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ρ Спирмена важностей (20 непрерывных)")
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("E3: согласие методов на настоящей цели (CV) и на перестановках (нуль); 10–90% повторов")
fig.tight_layout()
fig.savefig(FIGURES / "e3_agreement_groups.png", dpi=110)
plt.show()

# %% [markdown]
# ## 9. Обучение против теста

# %%
rows = []
for t in TRAITS:
    for m in MODELS:
        for kind in ("cv", "null"):
            for meth in ("shap", "lime"):
                a, b = get(t, m, kind, f"{meth}_train"), get(t, m, kind, f"{meth}_test")
                rows.append({"trait": t, "model": m, "kind": kind, "method": meth,
                             "rho": np.nanmean([M.rank_agreement(a.iloc[i], b.iloc[i])["spearman"] for i in range(len(a))])})
print("ρ рангов обучение–тест (среднее по повторам):")
print(pd.DataFrame(rows).pivot_table(index=["trait", "model"], columns=["kind", "method"], values="rho").round(2).to_string())

# %% [markdown]
# ## Выводы
#
# Подробно с числами и сравнением с E1 и E2: `docs/e3_results.md`.
#
# 1. **`extra`:** у деревьев от нуля отличаются друзья и лайки (у RF ровно они у всех пяти
#    методов). HGB по PI выделяет ещё три флага. Ridge с RidgeCV нельзя сравнивать с нулём:
#    нулевые модели выбирают α = 10⁴. С фиксированным α — друзья, лайки, has_university.
# 2. **`emo`:** ни один признак не отличается от нуля ни у одной модели и метода.
# 3. **Раскол методов** тот же, что в E1 и E2: SHAP, LIME, PDP и ALE согласны (0.81–0.97)
#    у обеих черт и в нуле, PI с ними согласен только у `extra`.
# 4. **Устойчивость рангов SHAP и LIME** между фолдами высокая и у `emo` (0.74–0.85 у деревьев),
#    поэтому она не доказывает сигнал.

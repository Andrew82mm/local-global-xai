# %% [markdown]
# # 09. E4: разбор причин расхождений
#
# Дизайн и гипотезы записаны до прогонов в `docs/e4_results.md`. Пункты по приоритету:
# 1. флаги: PI против SHAP и LIME, LIME с `categorical_features`;
# 2. LIME: ширина ядра, seed, local R² (синтетика E1 и полусинтетика E2);
# 3. взаимодействия в реальных данных (`extra`) против нуля из перестановок;
# 4. PI группой против PI по одному признаку (E2 и E3);
# 5. порог около 2% дисперсии y на признак.
#
# **Персональные данные:** сохраняются и выводятся только агрегаты (важности по признакам,
# медианы по объектам). Кеш: `results/e4_*.csv`, пересчёт: `E4_RECOMPUTE=1`, проба: `E4_SMOKE=1`.
# Запуск: `uv run python notebooks/09_e4_causes.py`.

# %%
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from xai_consistency import explain as E  # noqa: E402
from xai_consistency import metrics as M  # noqa: E402
from xai_consistency.data import CONTINUOUS, FEATURES_FLAG, FEATURES_LOG, FEATURES_OTHER, load_real  # noqa: E402
from xai_consistency.models import fit_transform, make_pipeline  # noqa: E402
from xai_consistency.semisynthetic import FoldTruth, Truth, make_target, to_explain_space  # noqa: E402
from xai_consistency.synthetic import make_synthetic  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 300)

SMOKE = os.environ.get("E4_SMOKE") == "1"
RECOMPUTE = os.environ.get("E4_RECOMPUTE") == "1"
N_SEEDS = 2 if SMOKE else 30
N_PERM_INT = 3 if SMOKE else 100     # пункт 3
N_PERM_GROUP = 5 if SMOKE else 1000  # пункт 4, E3
TEST_SIZE = 0.2
N_JOBS = min(6, os.cpu_count() or 1)
RESULTS = ROOT / "results"
TAG = "e4_smoke" if SMOKE else "e4"

X_REAL, _ = load_real("extra")  # для E2 и пункта 5 нужны только признаки
FEATURES = list(X_REAL.columns)
T_FULL = to_explain_space(X_REAL)
TRUTH = Truth(T_FULL)
FLAGS = list(FEATURES_FLAG)
HU, FR, LK = "has_university", "vk_counter_friends", "vk_post_avg_likes"
INTER = ["vk_counter_photos", "vk_wall_posts_total"]


def cached(name, compute):
    """Таблица из кеша results/e4_<name>.csv или расчёт и запись."""
    path = RESULTS / f"{TAG}_{name}.csv"
    if path.exists() and not RECOMPUTE:
        return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    t0 = time.perf_counter()
    df = compute()
    print(f"[{name}] расчёт {(time.perf_counter() - t0) / 60:.1f} мин")
    df.to_csv(path, index=False)
    return df


def long_rows(key, imp_by_variant):
    return [{**key, "variant": v, "feature": f, "importance": float(x)}
            for v, s in imp_by_variant.items() for f, x in s.items()]


def per_object_rho(a, b):
    """Медиана по объектам ρ Спирмена между |a| и |b| (две локальные таблицы)."""
    r = [M.rank_agreement(a.iloc[i].abs(), b.iloc[i].abs())["spearman"] for i in range(len(a))]
    return float(np.nanmedian(r))


# %% [markdown]
# ## 1–2б. Полусинтетика E2: флаги и LIME
#
# Те же seed, что в E2 (`make_target`, разбиение и seed модели = seed повтора), поэтому PI, SHAP
# и LIME по умолчанию воспроизводят E2. Варианты LIME на модели: по умолчанию (0.75·√d),
# `categorical_features` для 14 флагов, ширина ядра 0.3 и 3.0, другой seed. На истинной функции:
# точные SHAP, PI и оба вида LIME на 50 объектах.

# %%
E2_R2 = (0.0, 0.08, 0.2)


def run_e2(r2, model_name, seed):
    y, _ = make_target(TRUTH, T_FULL, r2, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X_REAL, y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                            curve_features=CONTINUOUS, seed=seed, methods=("pi", "shap", "lime"), splits=("test",))
    model = res.pipe.named_steps["model"]
    Z_tr = pd.DataFrame(res.pipe.named_steps["prep"].transform(X_tr), columns=res.features)
    S = res.samples["test"]
    lime = {"lime": res[("lime", "test")]}
    lime["lime_cat"] = E.lime_values(model, Z_tr, S, "test", seed, categorical=FLAGS)
    lime["lime_k03"] = E.lime_values(model, Z_tr, S, "test", seed, kernel_width_factor=0.3)
    lime["lime_k3"] = E.lime_values(model, Z_tr, S, "test", seed, kernel_width_factor=3.0)
    lime["lime_seed"] = E.lime_values(model, Z_tr, S, "test", seed + 1000)
    imp = {"pi": res[("pi", "test")].importance, "shap": res[("shap", "test")].importance}
    imp.update({k: ex.importance for k, ex in lime.items()})
    imp["lime_contrib"] = lime["lime"].info["contrib"].abs().mean()
    imp["lime_cat_flip"] = lime["lime_cat"].importance.copy()
    imp["lime_cat_flip"][FLAGS] = lime["lime_cat"].info["flip"][FLAGS].abs().mean()
    key = {"r2": r2, "model": model_name, "seed": seed}
    run = {**key, "r2_test": res.r2_test,
           **{f"local_r2_{k}": float(ex.info["local_r2"].median()) for k, ex in lime.items()},
           "lime_seed_rho_global": M.rank_agreement(imp["lime"], imp["lime_seed"])["spearman"],
           "lime_seed_rho_object": per_object_rho(lime["lime"].local, lime["lime_seed"].local)}
    if r2 > 0:  # истинная функция теми же методами
        ft = FoldTruth(TRUTH, res.pipe, res.features, r2)
        imp["truth_shap"] = ft.true_shap(S, res.background).abs().mean()
        imp["truth_pi"] = E.permutation_importance(ft, res.Z_test, y_te, 10, seed).importance
        imp["truth_lime"] = E.lime_values(ft, Z_tr, S.iloc[:50], "test", seed).importance
        imp["truth_lime_cat"] = E.lime_values(ft, Z_tr, S.iloc[:50], "test", seed, categorical=FLAGS).importance
    return long_rows(key, imp), run


def compute_e2():
    jobs = [(r2, m, s) for r2 in E2_R2 for m in ("rf", "hgb") for s in range(N_SEEDS)]
    out = Parallel(n_jobs=N_JOBS)(delayed(run_e2)(*j) for j in jobs)
    runs = pd.DataFrame([o[1] for o in out])
    runs.to_csv(RESULTS / f"{TAG}_e2_runs.csv", index=False)
    return pd.DataFrame([r for o in out for r in o[0]])


e2 = cached("e2_importance", compute_e2)
e2_runs = pd.read_csv(RESULTS / f"{TAG}_e2_runs.csv")
W2 = e2.pivot(index=["r2", "model", "seed", "variant"], columns="feature", values="importance")[FEATURES]
R2 = W2.rank(axis=1, ascending=False)

# %% [markdown]
# ### Проверка: воспроизводится ли E2

# %%
ref_path = RESULTS / "e2_features.csv"
if ref_path.exists() and not SMOKE:
    ref = pd.read_csv(ref_path)
    ref = ref[(ref["feature"] == HU) & ref["method"].isin(["pi_test", "shap_test", "lime_test"]) & ref["r2"].isin(E2_R2)]
    ref = ref[ref["model"].isin(["rf", "hgb"])].assign(variant=lambda d: d["method"].str.replace("_test", ""))
    mine = W2[HU].rename("mine").reset_index()
    chk = ref.merge(mine, on=["r2", "model", "seed", "variant"])
    print(f"E2: has_university, {len(chk)} пар; макс. |разница важности| = {(chk['importance'] - chk['mine']).abs().max():.2e}")

# %% [markdown]
# ### 1. Флаги

# %%
VARIANTS_1 = ["pi", "shap", "lime", "lime_cat", "lime_cat_flip"]
print("Доля повторов, где has_university в top-6 (из 34):")
top6 = (R2[HU] <= 6).groupby(level=["r2", "model", "variant"]).mean().unstack("variant")
print(top6[VARIANTS_1 + ["lime_k03", "lime_k3"]].round(2).to_string())

# %%
print("Отношение важности has_university к друзьям, медиана по повторам (модель и истинная функция):")
ratio = (W2[HU] / W2[FR]).groupby(level=["r2", "model", "variant"]).median().unstack("variant")
cols = ["pi", "truth_pi", "shap", "truth_shap", "lime", "truth_lime", "lime_cat", "truth_lime_cat"]
print(ratio.loc[[0.08, 0.2], cols].round(3).to_string())
print("\nLIME на истинной функции: по умолчанию / categorical =",
      ratio.loc[[0.08, 0.2], "truth_lime"].div(ratio.loc[[0.08, 0.2], "truth_lime_cat"]).round(2).to_dict())

# %% [markdown]
# ### 1б. LIME на истине со ступенькой, а не лестницей
#
# Добавлено после первого прогона. Флаг в истинной функции был задан как round(T): на данных
# (0 и 1) это ступенька, но вне {0, 1} — лестница с шагом каждые 2.66 SD. LIME сэмплирует соседей
# вне {0, 1} и пересекал её в обе стороны, поэтому LIME по умолчанию на истине почти не занижал
# флаг (отношение к categorical 0.86). Теперь флаг — clip(round(T), 0, 1), как ступенька дерева
# (вне данных дерево постоянно). На самих данных f не изменилась: истинные SHAP и PI те же.

# %%
def run_truth_lime(r2, model_name, seed):
    y, _ = make_target(TRUTH, T_FULL, r2, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X_REAL, y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                            curve_features=CONTINUOUS, seed=seed, methods=(), splits=())
    Z_tr = pd.DataFrame(res.pipe.named_steps["prep"].transform(X_tr), columns=res.features)
    S = res.samples["test"].iloc[:50]
    ft = FoldTruth(TRUTH, res.pipe, res.features, r2)
    imp = {"truth_lime_step": E.lime_values(ft, Z_tr, S, "test", seed).importance,
           "truth_lime_cat_step": E.lime_values(ft, Z_tr, S, "test", seed, categorical=FLAGS).importance}
    return long_rows({"r2": r2, "model": model_name, "seed": seed}, imp)


tl = cached("e2_truth_lime_step", lambda: pd.DataFrame([r for o in Parallel(n_jobs=N_JOBS)(
    delayed(run_truth_lime)(r2, m, s) for r2 in E2_R2 if r2 > 0 for m in ("rf", "hgb") for s in range(N_SEEDS)) for r in o]))
TL = tl.pivot(index=["r2", "model", "seed", "variant"], columns="feature", values="importance")[FEATURES]
rt = (TL[HU] / TL[FR]).groupby(level=["r2", "model", "variant"]).median().unstack("variant")
print("LIME на истине-ступеньке: has_university / friends (медиана) и отношение по умолчанию / categorical:")
print(rt.assign(ratio=rt["truth_lime_step"] / rt["truth_lime_cat_step"]).round(3).to_string())

# %%
print("z has_university относительно собственного нуля (тот же метод и модель, R² = 0, 30 повторов):")
null = W2.xs(0.0, level="r2")[HU].groupby(level=["model", "variant"]).agg(["mean", "std"])
rows = []
for (r2, m, s, v), x in W2[HU].items():
    if r2 > 0 and (m, v) in null.index:
        rows.append({"r2": r2, "model": m, "variant": v, "z": (x - null.loc[(m, v), "mean"]) / null.loc[(m, v), "std"]})
zt = pd.DataFrame(rows).groupby(["r2", "model", "variant"])["z"].median().unstack("variant")
print(zt[VARIANTS_1].round(2).to_string())
print("\nДля сравнения z друзей:")
rows = []
nf = W2.xs(0.0, level="r2")[FR].groupby(level=["model", "variant"]).agg(["mean", "std"])
for (r2, m, s, v), x in W2[FR].items():
    if r2 > 0 and (m, v) in nf.index:
        rows.append({"r2": r2, "model": m, "variant": v, "z": (x - nf.loc[(m, v), "mean"]) / nf.loc[(m, v), "std"]})
print(pd.DataFrame(rows).groupby(["r2", "model", "variant"])["z"].median().unstack("variant")[VARIANTS_1].round(1).to_string())

# %%
print("R² = 0: среднее число флагов в top-6 (из 14 флагов и 20 непрерывных; при случайном порядке 6·14/34 = 2.5):")
flags_top = (R2.xs(0.0, level="r2")[FLAGS] <= 6).sum(axis=1).groupby(level=["model", "variant"]).mean().unstack("variant")
print(flags_top[VARIANTS_1 + ["lime_k03", "lime_k3"]].round(2).to_string())
print("R² = 0.2: то же")
print((R2.xs(0.2, level="r2")[FLAGS] <= 6).sum(axis=1).groupby(level=["model", "variant"]).mean()
      .unstack("variant")[VARIANTS_1].round(2).to_string())

# %% [markdown]
# ### 2б. LIME на E2: ширина ядра, вклады, seed, local R²

# %%
V2 = ["shap", "lime", "lime_contrib", "lime_k03", "lime_k3", "lime_cat"]
print("Доля повторов в top-6: photos и wall_posts (признаки взаимодействия), likes:")
for f in INTER + [LK]:
    print(f"\n{f}")
    print((R2[f] <= 6).groupby(level=["r2", "model", "variant"]).mean().unstack("variant")[V2].round(2).to_string())

# %%
print("Recall@6 по 6 истинным признакам (среднее по повторам):")
rel = TRUTH.relevant
rec = (R2[rel] <= 6).mean(axis=1).groupby(level=["r2", "model", "variant"]).mean().unstack("variant")
print(rec[["pi"] + V2].round(2).to_string())

# %%
print("Медиана local R² по объектам (среднее по повторам):")
lr = e2_runs.groupby(["r2", "model"])[[c for c in e2_runs if c.startswith("local_r2_")]].mean()
print(lr.round(3).to_string())
print("\nLIME, seed против seed + 1000: ρ глобальных важностей и медиана ρ по объектам:")
print(e2_runs.groupby(["r2", "model"])[["lime_seed_rho_global", "lime_seed_rho_object"]].mean().round(3).to_string())

# %% [markdown]
# ## 2а. Синтетика E1, сценарий interaction
#
# Генерация, разбиение и seed как в E1 (ноутбук 06). Истинные признаки x1..x4, взаимодействие x1·x2.

# %%
E1_R2 = (0.0, 0.08, 0.2, 0.5)
E1_REL = ["x1", "x2", "x3", "x4"]


def run_e1(r2, model_name, seed):
    data = make_synthetic("interaction", n=1000, r2=r2, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(data.X, data.y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, seed=seed, methods=("shap", "lime"), splits=("test",))
    model = res.pipe.named_steps["model"]
    Z_tr = pd.DataFrame(res.pipe.named_steps["prep"].transform(X_tr), columns=res.features)
    S = res.samples["test"]
    lime = {"lime": res[("lime", "test")],
            "lime_k03": E.lime_values(model, Z_tr, S, "test", seed, kernel_width_factor=0.3),
            "lime_k3": E.lime_values(model, Z_tr, S, "test", seed, kernel_width_factor=3.0),
            "lime_seed": E.lime_values(model, Z_tr, S, "test", seed + 1000)}
    imp = {"shap": res[("shap", "test")].importance, **{k: ex.importance for k, ex in lime.items()}}
    for k in ("lime", "lime_k03", "lime_k3"):
        imp[f"{k}_contrib"] = lime[k].info["contrib"].abs().mean()
    key = {"r2": r2, "model": model_name, "seed": seed}
    run = {**key, "r2_test": res.r2_test,
           **{f"local_r2_{k}": float(ex.info["local_r2"].median()) for k, ex in lime.items()},
           "lime_seed_rho_global": M.rank_agreement(imp["lime"], imp["lime_seed"])["spearman"],
           "lime_seed_rho_object": per_object_rho(lime["lime"].local, lime["lime_seed"].local)}
    return long_rows(key, imp), run


def compute_e1():
    jobs = [(r2, m, s) for r2 in E1_R2 for m in ("rf", "hgb") for s in range(N_SEEDS)]
    out = Parallel(n_jobs=N_JOBS)(delayed(run_e1)(*j) for j in jobs)
    pd.DataFrame([o[1] for o in out]).to_csv(RESULTS / f"{TAG}_e1_runs.csv", index=False)
    return pd.DataFrame([r for o in out for r in o[0]])


e1 = cached("e1_importance", compute_e1)
e1_runs = pd.read_csv(RESULTS / f"{TAG}_e1_runs.csv")
W1 = e1.pivot(index=["r2", "model", "seed", "variant"], columns="feature", values="importance")
R1 = W1.rank(axis=1, ascending=False)
V1 = ["shap", "lime", "lime_contrib", "lime_k03", "lime_k03_contrib", "lime_k3", "lime_k3_contrib", "lime_seed"]

# %%
print("(а), (б) Доля повторов, где все 4 истинных признака в top-4:")
allf = (R1[E1_REL] <= 4).all(axis=1).groupby(level=["r2", "model", "variant"]).mean().unstack("variant")
print(allf.loc[[0.08, 0.2, 0.5], V1].round(2).to_string())
print("\nДоля повторов, где x1 и x2 оба в top-4:")
both = (R1[["x1", "x2"]] <= 4).all(axis=1).groupby(level=["r2", "model", "variant"]).mean().unstack("variant")
print(both.loc[[0.08, 0.2, 0.5], V1].round(2).to_string())
ref1 = RESULTS / "e1_methods.csv"
if ref1.exists() and not SMOKE:
    em = pd.read_csv(ref1)
    em = em[(em["scenario"] == "interaction") & em["method"].isin(["shap_test", "lime_test"]) & em["model"].isin(["rf", "hgb"])]
    print("\nE1 (ноутбук 06), all_found для сравнения:")
    print(em.groupby(["r2", "model", "method"])["all_found"].mean().unstack("method").round(2).to_string())

# %%
print("(в) Отношение LIME/SHAP (медиана по повторам): x1, x2 (взаимодействие) и x3, x4 (линейные)")
rows = []
for k in ("lime", "lime_k03", "lime_k3"):
    rat = W1.xs(k, level="variant") / W1.xs("shap", level="variant")
    for grp, fs in [("x1,x2", ["x1", "x2"]), ("x3,x4", ["x3", "x4"]), ("шум", [f"x{j}" for j in range(5, 11)])]:
        s = rat[fs].mean(axis=1).groupby(level=["r2", "model"]).median()
        rows += [{"r2": r2, "model": m, "lime": k, "group": grp, "ratio": v} for (r2, m), v in s.items()]
print(pd.DataFrame(rows).pivot_table(index=["r2", "model"], columns=["lime", "group"], values="ratio").round(2).to_string())

# %%
print("(г) Шум в единицах x3: среднее важности x5..x10 / важность x3 (медиана по повторам)")
noise = W1[[f"x{j}" for j in range(5, 11)]].mean(axis=1) / W1["x3"]
print(noise.groupby(level=["r2", "model", "variant"]).median().unstack("variant").loc[[0.08, 0.2, 0.5], V1].round(3).to_string())

# %%
print("Медиана local R² по объектам (среднее по повторам) и устойчивость к seed:")
print(e1_runs.groupby(["r2", "model"])[[c for c in e1_runs if c.startswith("local_r2_") or c.startswith("lime_seed")]]
      .mean().round(3).to_string())

# %% [markdown]
# ## 3. Взаимодействия в реальных данных
#
# `extra`, RF и HGB: наблюдаемая модель на разбиении 0 и 100 перестановок с протоколом и seed E3
# (`split_for` повторяет ноутбук 08). SHAP interaction values (`tree_path_dependent`) на тех же
# 200 тестовых объектах. H² Фридмана для friends × likes. Для сравнения наблюдаемая модель `emo`.

# %%
DATA = {}
for t in ("extra", "emo"):
    Xt, yt = load_real(t)
    DATA[t] = (Xt, (yt - yt.mean()) / yt.std())
SPLIT_SEED = 0


def split_for(trait, kind, idx):
    """Как в ноутбуке 08: obs, null (перестановка цели) или cv (фолд 5×5)."""
    X, y = DATA[trait]
    if kind in ("obs", "null"):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        if kind == "null":
            r = np.random.default_rng(SPLIT_SEED + 10_000 + idx)
            return X_tr, X_te, r.permutation(y_tr), r.permutation(y_te), SPLIT_SEED + idx
        return X_tr, X_te, y_tr, y_te, SPLIT_SEED
    tr, te = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=0).split(X))[idx]
    return X.iloc[tr], X.iloc[te], y[tr], y[te], 100 + idx


def h_statistic(model, Z, a, b):
    """H² Фридмана для пары (a, b), PD в точках самих объектов (как в ноутбуке 07)."""
    Zv = np.asarray(Z, float)
    n = len(Zv)
    ia, ib = Z.columns.get_loc(a), Z.columns.get_loc(b)

    def pd_at(cols):
        big = np.tile(Zv, (n, 1))
        for c in cols:
            big[:, c] = np.repeat(Zv[:, c], n)
        v = model.predict(big).reshape(n, n).mean(axis=1)
        return v - v.mean()

    pab, pa, pb = pd_at([ia, ib]), pd_at([ia]), pd_at([ib])
    return float(np.sum((pab - pa - pb) ** 2) / np.sum(pab**2))


def run_inter(trait, model_name, kind, idx):
    X_tr, X_te, y_tr, y_te, seed = split_for(trait, kind, idx)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                            curve_features=CONTINUOUS, seed=seed, methods=(), splits=())
    model, S = res.pipe.named_steps["model"], res.samples["test"]
    iv = np.abs(shap.TreeExplainer(model).shap_interaction_values(S.to_numpy())).mean(axis=0)  # 34 × 34
    total = iv.sum()
    off = iv.copy()
    np.fill_diagonal(off, 0)
    names = res.features
    i_f, i_l = names.index(FR), names.index(LK)
    iu = np.triu_indices(len(names), 1)
    pair = 2 * off[iu] / total  # доля пары (Φ_ij и Φ_ji)
    row = {"trait": trait, "model": model_name, "kind": kind, "idx": idx, "r2_test": res.r2_test,
           "inter_share": off.sum() / total, "max_pair_share": pair.max(),
           "fl_pair_share": 2 * off[i_f, i_l] / total,
           "friends_inter_share": off[i_f].sum() / iv[i_f].sum(),
           "h2_fl": h_statistic(model, S, FR, LK)}
    pairs = pd.Series(pair, index=[f"{names[a]}|{names[b]}" for a, b in zip(*iu)])
    return row, pairs


def compute_inter():
    jobs = [("extra", m, "obs", 0) for m in ("rf", "hgb")] + [("emo", m, "obs", 0) for m in ("rf", "hgb")]
    jobs += [("extra", m, "null", i) for m in ("rf", "hgb") for i in range(N_PERM_INT)]
    out = Parallel(n_jobs=N_JOBS)(delayed(run_inter)(*j) for j in jobs)
    pairs = pd.DataFrame([o[1] for o in out])
    pairs.insert(0, "key", [f"{o[0]['trait']}|{o[0]['model']}|{o[0]['kind']}|{o[0]['idx']}" for o in out])
    pairs.to_csv(RESULTS / f"{TAG}_inter_pairs.csv", index=False)
    return pd.DataFrame([o[0] for o in out])


inter = cached("inter", compute_inter)
inter_pairs = pd.read_csv(RESULTS / f"{TAG}_inter_pairs.csv").set_index("key")

# %%
STATS = ["inter_share", "max_pair_share", "fl_pair_share", "friends_inter_share", "h2_fl"]
rows = []
for m in ("rf", "hgb"):
    null = inter[(inter["model"] == m) & (inter["kind"] == "null")]
    for t in ("extra", "emo"):
        obs = inter[(inter["model"] == m) & (inter["kind"] == "obs") & (inter["trait"] == t)].iloc[0]
        for s in STATS:
            rows.append({"model": m, "trait": t, "stat": s, "набл.": obs[s], "нуль: среднее": null[s].mean(),
                         "нуль 95%": null[s].quantile(0.95), "процентиль": (null[s] < obs[s]).mean() * 100})
print("Взаимодействия: наблюдаемая модель против 100 перестановок extra (для emo — тот же нуль, ориентир):")
print(pd.DataFrame(rows).set_index(["model", "stat", "trait"]).round(4).to_string())

# %%
print("extra: 5 пар с наибольшей долей у наблюдаемой модели, их процентиль в нуле и эмпирическое p:")
for m in ("rf", "hgb"):
    obs = inter_pairs.loc[f"extra|{m}|obs|0"]
    null = inter_pairs[inter_pairs.index.str.startswith(f"extra|{m}|null|")]
    top = obs.sort_values(ascending=False).head(5)
    p = M.null_pvalues(obs, null)
    q = M.bh_fdr(p)
    print(f"\n{m}: пар с q_emp < 0.05 (BH по {len(obs)} парам): {int((q < 0.05).sum())}")
    print(pd.DataFrame({"доля": top, "нуль 95%": null[top.index].quantile(0.95), "p_emp": p[top.index]}).round(4).to_string())

# %% [markdown]
# ### Аддитивная HGB против обычной на 25 фолдах

# %%
def run_additive(trait, idx):
    X_tr, X_te, y_tr, y_te, seed = split_for(trait, "cv", idx)
    out = {"trait": trait, "idx": idx}
    for name in ("full", "additive"):
        pipe = make_pipeline("hgb", FEATURES_LOG, FEATURES_OTHER, random_state=seed)
        if name == "additive":
            m = pipe.named_steps["model"]
            pipe.set_params(model=HistGradientBoostingRegressor(**{**m.get_params(), "interaction_cst": [[i] for i in range(len(FEATURES))]}))
        pipe.fit(X_tr, y_tr)
        out[name] = r2_score(y_te, pipe.predict(X_te))
    return out


add = cached("additive_hgb", lambda: pd.DataFrame(Parallel(n_jobs=N_JOBS)(
    delayed(run_additive)(t, i) for t in ("extra", "emo") for i in range(2 if SMOKE else 25))))
add["diff"] = add["additive"] - add["full"]
print("R² на 25 фолдах: обычная HGB, аддитивная HGB, разница (аддитивная − обычная):")
print(add.groupby("trait")[["full", "additive", "diff"]].agg(["mean", "std"]).round(4).to_string())
print("Доля фолдов, где аддитивная не хуже:", add.assign(ok=add["diff"] >= 0).groupby("trait")["ok"].mean().round(2).to_dict())

# %% [markdown]
# Добавлено после первого прогона. У аддитивной HGB при тех же 150 итерациях меньше ёмкость
# (каждое дерево по одному признаку), поэтому разница смешивает взаимодействия и недообучение.
# Проверка: аддитивная с 450 итерациями и аддитивная, где разрешено одно взаимодействие friends × likes.

# %%
def run_additive2(trait, idx):
    X_tr, X_te, y_tr, y_te, seed = split_for(trait, "cv", idx)
    i_f, i_l = FEATURES.index(FR), FEATURES.index(LK)
    singles = [[i] for i in range(len(FEATURES))]
    pair = [[i] for i in range(len(FEATURES)) if i not in (i_f, i_l)] + [[i_f, i_l]]
    out = {"trait": trait, "idx": idx}
    for name, cst, it in [("additive_450", singles, 450), ("fl_pair_150", pair, 150), ("fl_pair_450", pair, 450),
                          ("full_450", None, 450)]:
        pipe = make_pipeline("hgb", FEATURES_LOG, FEATURES_OTHER, random_state=seed)
        m = pipe.named_steps["model"]
        pipe.set_params(model=HistGradientBoostingRegressor(**{**m.get_params(), "interaction_cst": cst, "max_iter": it}))
        pipe.fit(X_tr, y_tr)
        out[name] = r2_score(y_te, pipe.predict(X_te))
    return out


add2 = cached("additive_hgb2", lambda: pd.DataFrame(Parallel(n_jobs=N_JOBS)(
    delayed(run_additive2)(t, i) for t in ("extra", "emo") for i in range(2 if SMOKE else 25))))
add2 = add2.merge(add, on=["trait", "idx"])
cols = ["full", "additive", "additive_450", "fl_pair_150", "fl_pair_450", "full_450"]
print("R² на 25 фолдах (среднее), 150 и 450 итераций:")
print(add2.groupby("trait")[cols].mean().round(4).to_string())
for a, b in [("additive_450", "full"), ("fl_pair_150", "additive"), ("fl_pair_450", "additive_450")]:
    d = add2[a] - add2[b]
    print(f"{a} − {b}:", {t: f"{v.mean():+.4f} (доля фолдов > 0: {(v > 0).mean():.2f})" for t, v in d.groupby(add2['trait'])})

# %% [markdown]
# ## 4. PI группой против PI по одному признаку
#
# Группы заданы до прогонов: связные компоненты графа |ρ Спирмена| ≥ 0.5 (после log1p, все строки).

# %%
GROUPS = {
    "posts": ["vk_post_avg_likes", "vk_post_avg_views", "vk_post_avg_reposts", "photo_per_post", "repost_share", "vk_counter_gifts"],
    "profile": ["profile_block", "personal_block", "has_relation"],
    "audio": ["vk_counter_audios", "vk_can_see_audio"],
    "wall": ["vk_counter_photos", "vk_wall_posts_total"],
}
GROUPED = [f for g in GROUPS.values() for f in g]
UNITS = {**GROUPS, **{f: [f] for f in FEATURES if f not in GROUPED}}
T_rho = T_FULL.corr(method="spearman")
print("Мин. |ρ| внутри групп (по рёбрам компоненты не меньше 0.5):",
      {g: round(float(T_rho.loc[fs, fs].abs().min().min()), 2) for g, fs in GROUPS.items()})


def pi_single_and_groups(model, Z_te, y_te, seed):
    single = E.permutation_importance(model, Z_te, y_te, 10, seed).importance
    grp = E.permutation_importance(model, Z_te, y_te, 10, seed, groups=UNITS).importance
    return single, grp


def run_group_e2(r2, model_name, seed):
    y, _ = make_target(TRUTH, T_FULL, r2, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X_REAL, y, test_size=TEST_SIZE, random_state=seed)
    pipe = make_pipeline(model_name, FEATURES_LOG, FEATURES_OTHER, random_state=seed)
    Z_tr, Z_te = fit_transform(pipe, X_tr, y_tr, X_te)
    key = {"r2": r2, "model": model_name, "seed": seed}
    s, g = pi_single_and_groups(pipe.named_steps["model"], Z_te, y_te, seed)
    rows = long_rows({**key, "target": "model"}, {"single": s, "group": g})
    if r2 > 0:
        ft = FoldTruth(TRUTH, pipe, list(Z_te.columns), r2)
        s, g = pi_single_and_groups(ft, Z_te, y_te, seed)
        rows += long_rows({**key, "target": "truth"}, {"single": s, "group": g})
    return rows


grp_e2 = cached("group_e2", lambda: pd.DataFrame([r for o in Parallel(n_jobs=N_JOBS)(
    delayed(run_group_e2)(r2, m, s) for r2 in E2_R2 for m in ("ridge", "rf", "hgb") for s in range(N_SEEDS)) for r in o]))

# %%
G2 = grp_e2.pivot(index=["r2", "model", "seed", "target", "variant"], columns="feature", values="importance")
print("E2: PI группы / сумма PI её признаков (медиана по повторам):")
rows = []
for (r2, m, s, tgt), g in G2.groupby(level=["r2", "model", "seed", "target"]):
    sg, gg = g.xs("single", level="variant").iloc[0], g.xs("group", level="variant").iloc[0]
    for name, fs in GROUPS.items():
        rows.append({"r2": r2, "model": m, "target": tgt, "group": name, "ratio": gg[name] / sg[fs].sum(),
                     "group_pi": gg[name], "sum_single": sg[fs].sum()})
gr = pd.DataFrame(rows)
print(gr[gr["r2"] > 0].pivot_table(index=["r2", "model", "target"], columns="group", values="ratio", aggfunc="median").round(2).to_string())
print("\nE2, audio: PI группы и PI одного audios у модели против PI истины (медианы):")
aud = G2.xs("model", level="target")
tru = G2.xs("truth", level="target")
rows = []
for r2 in (0.08, 0.2):
    for m in ("ridge", "rf", "hgb"):
        a = aud.loc[(r2, m)]
        t = tru.loc[(r2, m)]
        rows.append({"r2": r2, "model": m,
                     "истина: audios": t.xs("single", level="variant")["vk_counter_audios"].median(),
                     "модель: audios": a.xs("single", level="variant")["vk_counter_audios"].median(),
                     "модель: двойник": a.xs("single", level="variant")["vk_can_see_audio"].median(),
                     "модель: группа audio": a.xs("group", level="variant")["audio"].median(),
                     "истина: группа audio": t.xs("group", level="variant")["audio"].median()})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# %% [markdown]
# ### PI группой на `extra`: 1000 перестановок
#
# Протокол и seed E3. У Ridge α зафиксирован на наблюдаемом (как в E3, раздел 5.3).

# %%
X_tr0, _, y_tr0, _, _ = split_for("extra", "obs", 0)
ALPHA_OBS = make_pipeline("ridge", FEATURES_LOG, FEATURES_OTHER).fit(X_tr0, y_tr0).named_steps["model"].alpha_
print(f"Ridge, наблюдаемый α у extra: {ALPHA_OBS:.1f}")


def run_group_e3(model_name, kind, idx):
    X_tr, X_te, y_tr, y_te, seed = split_for("extra", kind, idx)
    pipe = make_pipeline(model_name, FEATURES_LOG, FEATURES_OTHER, random_state=seed)
    if model_name == "ridge":
        pipe.set_params(model=RidgeCV(alphas=[ALPHA_OBS]))
    Z_tr, Z_te = fit_transform(pipe, X_tr, y_tr, X_te)
    s, g = pi_single_and_groups(pipe.named_steps["model"], Z_te, y_te, seed)
    return long_rows({"model": model_name, "kind": kind, "idx": idx}, {"single": s, "group": g})


grp_e3 = cached("group_e3", lambda: pd.DataFrame([r for o in Parallel(n_jobs=N_JOBS)(
    delayed(run_group_e3)(m, k, i) for m in ("ridge", "rf", "hgb")
    for k, i in [("obs", 0)] + [("null", j) for j in range(N_PERM_GROUP)]) for r in o]))

# %%
G3 = grp_e3.pivot(index=["model", "kind", "idx", "variant"], columns="feature", values="importance")
e3_imp = RESULTS / "e3_importance.csv"
if e3_imp.exists() and not SMOKE:  # PI наблюдаемых RF и HGB должны совпасть с E3
    ref = pd.read_csv(e3_imp, keep_default_na=False, na_values=[""])
    ref = ref[(ref["trait"] == "extra") & (ref["kind"] == "obs") & (ref["method"] == "pi_test")]
    for m in ("rf", "hgb"):
        a = G3.loc[(m, "obs", 0, "single")][FEATURES]
        b = ref[ref["model"] == m].set_index("feature")["importance"][FEATURES]
        print(f"{m}: макс. |PI − PI из E3| = {(a - b).abs().max():.2e}")
rows = {}
for m in ("ridge", "rf", "hgb"):
    for v, cols in [("single", FEATURES), ("group", list(UNITS))]:
        obs = G3.loc[(m, "obs", 0, v)][cols]
        null = G3.loc[(m, "null")].xs(v, level="variant")[cols]
        p = M.null_pvalues(obs, null)
        rows[(m, v)] = pd.DataFrame({"pi": obs, "n_ge": (null >= obs).sum(), "p_emp": p, "q_emp": M.bh_fdr(p)})
g3 = pd.concat(rows, names=["model", "variant", "unit"])
print(f"extra, {N_PERM_GROUP} перестановок: единицы с q_emp < 0.05 (BH по 34 признакам или по 25 единицам)")
print({k: g3.loc[k].index[g3.loc[k, "q_emp"] < 0.05].tolist() for k in rows})
show = list(GROUPS) + [FR] + [f for g in GROUPS.values() for f in g]
print(g3.reset_index().query("unit in @show").pivot_table(index="unit", columns=["model", "variant"], values="q_emp")
      .reindex(show).round(3).to_string())
print("\nPI группы / сумма PI её признаков у наблюдаемой модели:")
for m in ("ridge", "rf", "hgb"):
    s, g = G3.loc[(m, "obs", 0, "single")], G3.loc[(m, "obs", 0, "group")]
    print(m, {name: round(float(g[name] / s[fs].sum()), 2) for name, fs in GROUPS.items()})

# %% [markdown]
# ## 5. Порог доли дисперсии на признак
#
# y = √s·g + √(1 − s)·ε, g — одна проба, стандартизованная по всем строкам (в пространстве после log1p).

# %%
PROBES = [FR, "vk_counter_audios", HU]
SHARES = (0.005, 0.01, 0.02, 0.04)


def probe_signal(f):
    g = T_FULL[f].to_numpy(float)
    return (g - g.mean()) / g.std()


def run_probe(probe, s, model_name, seed):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(len(X_REAL))
    y = eps if probe == "none" else np.sqrt(s) * probe_signal(probe) + np.sqrt(1 - s) * eps
    X_tr, X_te, y_tr, y_te = train_test_split(X_REAL, y, test_size=TEST_SIZE, random_state=seed)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                            curve_features=CONTINUOUS, seed=seed, methods=("pi", "shap"), splits=("test",))
    return long_rows({"probe": probe, "s": s, "model": model_name, "seed": seed},
                     {"pi": res[("pi", "test")].importance, "shap": res[("shap", "test")].importance})


probe_jobs = [(p, s, m, i) for p in PROBES for s in SHARES for m in ("ridge", "rf", "hgb") for i in range(N_SEEDS)]
probe_jobs += [("none", 0.0, m, i) for m in ("ridge", "rf", "hgb") for i in range(N_SEEDS)]
probe = cached("probe", lambda: pd.DataFrame([r for o in Parallel(n_jobs=N_JOBS)(delayed(run_probe)(*j) for j in probe_jobs) for r in o]))

# %%
P = probe.pivot(index=["probe", "s", "model", "seed", "variant"], columns="feature", values="importance")[FEATURES]
PR = P.rank(axis=1, ascending=False)
null_p = P.xs("none", level="probe").droplevel("s")
rows = []
for (pr, s, m, seed, v), r in PR.iterrows():
    if pr == "none":
        continue
    nm = null_p.xs((m, v), level=["model", "variant"])[pr]
    rows.append({"probe": pr, "s": s, "model": m, "variant": v, "top6": r[pr] <= 6, "top1": r[pr] == 1,
                 "z": (P.loc[(pr, s, m, seed, v), pr] - nm.mean()) / nm.std()})
pt = pd.DataFrame(rows)
null_rows = [{"probe": pr, "s": 0.0, "model": m, "variant": v, "top6": r[pr] <= 6, "top1": r[pr] == 1, "z": np.nan}
             for (_, _, m, seed, v), r in PR.xs("none", level="probe", drop_level=False).iterrows() for pr in PROBES]
pt = pd.concat([pd.DataFrame(null_rows), pt], ignore_index=True)
for stat in ("top6", "top1", "z"):
    print(f"\n{stat}: доля повторов (z — медиана) по доле дисперсии пробы s")
    print(pt.pivot_table(index=["probe", "model", "variant"], columns="s", values=stat,
                         aggfunc="median" if stat == "z" else "mean").round(2).to_string())

# %% [markdown]
# ## 6. Local R² LIME на реальных данных (после основного E4)
#
# Открытый вопрос пункта 2: отличает ли local R² сигнал от нуля на настоящих чертах.
# `extra` и `emo`, RF и HGB: наблюдаемая модель (разбиение 0), 100 перестановок и 25 фолдов CV
# с протоколом и seed E3 (`split_for`). LIME по умолчанию на тех же 200 тестовых объектах.
# Сохраняются только агрегаты по объектам: медиана, квартили и среднее local R².
# Ridge не считается: для линейной модели локальная линейная модель точна, local R² = 1 по построению.

# %%
N_PERM_LOCAL = 3 if SMOKE else 100


def run_local_r2(trait, model_name, kind, idx):
    X_tr, X_te, y_tr, y_te, seed = split_for(trait, kind, idx)
    res = E.fit_and_explain(model_name, X_tr, y_tr, X_te, y_te, log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                            curve_features=CONTINUOUS, seed=seed, methods=("lime",), splits=("test",))
    r = res[("lime", "test")].info["local_r2"].to_numpy()
    return {"trait": trait, "model": model_name, "kind": kind, "idx": idx, "r2_test": res.r2_test,
            "local_r2_median": float(np.median(r)), "local_r2_q25": float(np.percentile(r, 25)),
            "local_r2_q75": float(np.percentile(r, 75)), "local_r2_mean": float(r.mean())}


local_jobs = [(t, m, k, i) for t in ("extra", "emo") for m in ("rf", "hgb")
              for k, n in (("obs", 1), ("null", N_PERM_LOCAL), ("cv", 2 if SMOKE else 25)) for i in range(n)]
local_real = cached("local_r2_real", lambda: pd.DataFrame(Parallel(n_jobs=N_JOBS)(
    delayed(run_local_r2)(*j) for j in local_jobs)))

# %%
rows = []
for (t, m), g in local_real.groupby(["trait", "model"]):
    for stat in ("local_r2_median", "local_r2_q25"):
        obs = g.loc[g["kind"] == "obs", stat].iloc[0]
        null, cv = g.loc[g["kind"] == "null", stat], g.loc[g["kind"] == "cv", stat]
        rows.append({"trait": t, "model": m, "stat": stat, "набл.": obs, "CV": cv.mean(), "CV SD": cv.std(),
                     "нуль": null.mean(), "нуль 5%": null.quantile(0.05), "нуль 95%": null.quantile(0.95),
                     "нуль max": null.max(), "процентиль набл.": (null < obs).mean() * 100,
                     "процентиль CV": (null < cv.mean()).mean() * 100,
                     "доля фолдов > 95% нуля": (cv > null.quantile(0.95)).mean()})
print(pd.DataFrame(rows).set_index(["stat", "trait", "model"]).sort_index().round(3).to_string())
print("\nСвязь local R² с R² на тесте по фолдам CV (ρ Спирмена):")
for (t, m), g in local_real[local_real["kind"] == "cv"].groupby(["trait", "model"]):
    print(t, m, round(g[["r2_test", "local_r2_median"]].corr("spearman").iloc[0, 1], 2))

# %% [markdown]
# ## Выводы
#
# Подробно с числами: `docs/e4_results.md`.
#
# 1. **Флаги.** LIME недооценивает флаг из-за гауссова возмущения: на ступеньке наклон ≈ φ(1.33)·Δ,
#    0.43 от честного (по формуле 0.44). С `categorical_features` и пересчётом на SD LIME находит
#    has_university не реже SHAP. PI ставит флаги выше SHAP в основном из-за шумового «верха»
#    непрерывных признаков у SHAP.
# 2. **LIME.** Ширина ядра 0.3–3·√d почти ничего не меняет (окрестность задаёт сэмплирование).
#    Преимущество во взаимодействии наполовину объясняется тем, что LIME мерит наклон, а не вклад.
#    Глобальная важность от seed не зависит (ρ ≥ 0.985). Local R² растёт с сигналом.
# 3. **Взаимодействия у extra.** H² friends × likes в пределах нуля; interaction values пары выше
#    нуля только из-за главных эффектов. Аддитивная HGB хуже на 0.012 R², причина не ясна.
# 4. **PI группой** помогает Ridge (значимы posts и profile) и RF, вредит HGB.
# 5. **Порог на признак** ниже 2%: одиночный признак находится при 0.5–1% дисперсии y.
# 6. **Local R² на реальных данных:** у `extra` выше всех 100 перестановок на разбиении 0 и в 25 из 25
#    фолдов (0.56–0.64 против нуля 0.37–0.41). У `emo` на разбиении 0 в пределах нуля, среднее CV
#    на 88–95 процентиле нуля одного разбиения.

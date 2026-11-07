# %% [markdown]
# # 04. Проверка общего кода на синтетике с известной истиной
#
# Это не эксперимент E1, а проверка инструмента перед ним. Вопросы:
#
# 1. Дают ли генераторы заданный R² и заданные корреляции?
# 2. Совпадают ли наши реализации PI, PDP и ALE с библиотечными? Выполняется ли
#    аддитивность SHAP?
# 3. При сильном сигнале (R² = 0.5) находят ли все пять методов истинные признаки,
#    знаки и форму эффекта? Если нет, это ошибка или свойство метода (модели)?
# 4. Работает ли нулевой эталон (модели на перемешанной цели)?
# 5. Сколько времени занимают методы и хватит ли вычислений на E1–E4?
#
# Решения о настройках описаны в `docs/methodology.md`.
#
# Реальные данные используются только в разделе 6 и только для замера времени:
# на экран выводятся секунды, а не объяснения.
#
# Запуск целиком: `uv run python notebooks/04_framework_check.py` (несколько минут).

# %%
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # запуск скриптом: графики только в файл
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging

from PyALE import ale
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from xai_consistency import explain as E  # noqa: E402
from xai_consistency import metrics as M  # noqa: E402
from xai_consistency.data import CONTINUOUS, FEATURES_LOG, FEATURES_OTHER, load_real  # noqa: E402
from xai_consistency.models import MODELS, fit_transform, make_pipeline  # noqa: E402
from xai_consistency.synthetic import R2_LEVELS, SCENARIOS, make_synthetic  # noqa: E402

logging.getLogger("PyALE").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)

SEED = 42
N = 1000  # как в реальных данных
TEST_SIZE = 0.2
N_JOBS = min(6, os.cpu_count() or 1)
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"


def split(data, seed=SEED):
    return train_test_split(data.X, data.y, test_size=TEST_SIZE, random_state=seed)


# %% [markdown]
# ## 1. Генераторы: заданный R² и корреляции
#
# Доля дисперсии сигнала в y должна быть близка к заданному R² (разброс из-за
# конечной выборки). У сценария correlated каждый истинный признак x1..x4 имеет
# «двойника» x5..x8 без собственного эффекта с корреляцией ρ.

# %%
rows = []
for sc in SCENARIOS:
    for r2 in R2_LEVELS:
        big = make_synthetic(sc, n=100_000, r2=r2, rho=0.9, seed=1)
        rows.append({"scenario": sc, "r2_target": r2,
                     "r2_signal": big.signal.var() / big.y.var(),
                     "var_y": big.y.var(),
                     "sum_true_shap_eq_signal": np.allclose(big.true_shap.sum(axis=1), big.signal)})
gen = pd.DataFrame(rows)
print(gen.round(3).to_string(index=False))

big = make_synthetic("correlated", n=100_000, r2=0.5, rho=0.9, seed=1)
print("\ncorrelated, ρ=0.9: corr(x1,x5) =", round(big.X["x1"].corr(big.X["x5"]), 3),
      "| corr(x1,x2) =", round(big.X["x1"].corr(big.X["x2"]), 3))
print("Истинная важность (mean |SHAP|), interaction R²=0.5:")
print(make_synthetic("interaction", n=100_000, r2=0.5, seed=1).true_importance.round(3).to_dict())

# %% [markdown]
# ## 2. Наши реализации против библиотечных
#
# - PI: наш (с перестановкой групп) против `sklearn.inspection.permutation_importance`.
#   Случайные перестановки разные, поэтому сравниваем значения с точностью до шума.
# - PDP: наш на нашей квантильной сетке против `sklearn.inspection.partial_dependence`
#   с той же сеткой (`custom_values`). Должны совпасть точно.
# - ALE: наш против PyALE на сетке PyALE. Должны совпасть с точностью до округления.
# - SHAP: сумма вкладов + базовое значение = прогноз (аддитивность).

# %%
data = make_synthetic("nonlinear", n=N, r2=0.5, seed=SEED)
X_train, X_test, y_train, y_test = split(data)
checks = []
for model_name in MODELS:
    pipe = make_pipeline(model_name, other_cols=list(data.X.columns), random_state=SEED)
    Z_train, Z_test = fit_transform(pipe, X_train, y_train, X_test)
    model = pipe.named_steps["model"]
    grids = E.make_grids(Z_test, Z_test.columns)

    ours_pi = E.permutation_importance(model, Z_test, y_test, n_repeats=30, seed=SEED)
    sk_pi = sk_permutation_importance(model, Z_test.to_numpy(), y_test, n_repeats=30,
                                      random_state=SEED, scoring="r2")
    pi_diff = np.abs(ours_pi.importance.to_numpy() - sk_pi.importances_mean)
    pi_se = np.sqrt(ours_pi.info["sd"].to_numpy() ** 2 / 30 + sk_pi.importances_std**2 / 30)

    ours_pdp = E.partial_dependence(model, Z_test, grids)
    pdp_err = 0.0
    for f, c in ours_pdp.curves.items():
        sk = partial_dependence(model, Z_test.to_numpy(), [Z_test.columns.get_loc(f)],
                                custom_values={Z_test.columns.get_loc(f): c["x"].to_numpy()},
                                kind="average", method="brute")
        pdp_err = max(pdp_err, np.abs(sk["average"][0] - c["value"].to_numpy()).max())

    class _Wrap:  # PyALE передаёт DataFrame, модель обучена на массиве
        def predict(self, df):
            return model.predict(df.to_numpy())

    ale_err, ale_scale = 0.0, 0.0
    for f in ["x1", "x2", "x4"]:
        ref = ale(X=Z_test, model=_Wrap(), feature=[f], grid_size=20, include_CI=False, plot=False)
        edges = ref.index.to_numpy(float)
        ours = E.accumulated_local_effects(model, Z_test, {f: edges}).curves[f]
        ref_centers = (ref["eff"].to_numpy()[:-1] + ref["eff"].to_numpy()[1:]) / 2
        ale_err = max(ale_err, np.abs(ours["value"].to_numpy() - ref_centers).max())
        ale_scale = max(ale_scale, np.abs(ref_centers).max())

    sh = E.shap_values(model, model_name, Z_train.iloc[:100], Z_test, "test")
    additivity = np.abs(sh.info["expected_value"] + sh.local.sum(axis=1).to_numpy() - model.predict(Z_test.to_numpy())).max()
    checks.append({"model": model_name,
                   "PI: max |наш − sklearn|": pi_diff.max(), "PI: max |разница| / SE": (pi_diff / pi_se).max(),
                   "PDP: max |наш − sklearn|": pdp_err,
                   "ALE: max |наш − PyALE|": ale_err, "ALE: размах кривой": ale_scale,
                   "SHAP: max ошибка аддитивности": additivity})
checks = pd.DataFrame(checks).set_index("model")
print(checks.T.to_string(float_format=lambda v: f"{v:.2e}"))

# %% [markdown]
# ## 3. Сильный сигнал: находят ли методы истину?
#
# R² = 0.5, n = 1000 (800 обучение, 200 тест), три модели, пять сценариев
# (correlated при ρ = 0.5 и 0.9). Для каждого метода:
#
# - recall@k — доля истинных признаков в top-k (k = число истинных);
# - margin — мин. важность истинного / макс. важность шумового (> 1: все истинные выше шума);
# - ρ_rel — порядок среди истинных признаков против истинной важности;
# - знаки — совпадение направления эффекта с истиной (только монотонные эффекты);
# - CCC кривых — согласие формы PDP/ALE/SHAP с истинным главным эффектом
#   (только признаки с ненулевым главным эффектом).
#
# Важности PDP и ALE — это размах кривой (взвешенное SD), см. methodology.md.

# %%
CONFIGS = [("independent", None), ("correlated", 0.5), ("correlated", 0.9),
           ("interaction", None), ("nonlinear", None)]


def check_config(scenario, rho, model_name, r2=0.5, seed=SEED):
    data = make_synthetic(scenario, n=N, r2=r2, rho=rho or 0.5, seed=seed)
    X_train, X_test, y_train, y_test = split(data, seed)
    res = E.fit_and_explain(model_name, X_train, y_train, X_test, y_test, seed=seed)
    rows = []
    with_curve = [f for f in data.relevant
                  if np.std(data.true_curve(f, np.linspace(-2, 2, 50))) > 1e-9]
    for (method, part), ex in res.explanations.items():
        fid = M.fidelity(ex.importance, data.true_importance, data.relevant)
        row = {"scenario": scenario if rho is None else f"{scenario} ρ={rho}", "model": model_name,
               "method": f"{method}_{part}", "r2_test": res.r2_test, **fid,
               "sign": M.sign_agreement(ex.sign, data.true_sign) if method != "pi" else np.nan,
               "seconds": ex.seconds}
        if ex.curves:
            truth = M.true_curves(ex.curves, data.true_curve)
            cmp = M.compare_curves(ex.curves, truth).reindex(with_curve)
            row["curve_ccc"] = cmp["ccc"].mean()
            row["curve_ccc_min"] = cmp["ccc"].min()
        rows.append(row)
    return rows, res, data


t0 = time.perf_counter()
out = Parallel(n_jobs=N_JOBS)(delayed(check_config)(sc, rho, m) for sc, rho in CONFIGS for m in MODELS)
print(f"Время: {time.perf_counter() - t0:.0f} с на {len(out)} конфигураций (параллельно, {N_JOBS} процессов)")
strong = pd.DataFrame([r for rows, _, _ in out for r in rows])
runs = {(rows[0]["scenario"], rows[0]["model"]): (res, data) for rows, res, data in out}
strong.to_csv(RESULTS / "framework_check_strong.csv", index=False)

print("\nR² на тесте (у лучшей возможной модели около 0.5):")
print(strong.drop_duplicates(["scenario", "model"]).pivot(index="scenario", columns="model", values="r2_test").round(3))

# %%
print("recall@k (1 = все истинные признаки в top-k):")
print(strong.pivot_table(index=["scenario", "model"], columns="method", values="recall_at_k").round(2))

# %%
print("margin (> 1: все истинные признаки важнее любого шумового):")
print(strong.pivot_table(index=["scenario", "model"], columns="method", values="margin").round(2))

# %%
print("ρ Спирмена среди истинных признаков (правильный ли порядок):")
print(strong.pivot_table(index=["scenario", "model"], columns="method", values="spearman_relevant").round(2))

# %%
print("Совпадение знаков с истиной (монотонные эффекты):")
print(strong.pivot_table(index=["scenario", "model"], columns="method", values="sign").round(2))

# %%
print("CCC кривой с истинным главным эффектом, среднее и минимум по признакам с эффектом:")
curve_cols = ["ale_test", "pdp_test", "shap_test", "shap_train"]
cc = strong[strong["method"].isin(curve_cols)]
print(cc.pivot_table(index=["scenario", "model"], columns="method", values=["curve_ccc", "curve_ccc_min"]).round(2))

# %% [markdown]
# ### Разбор: где методы «не нашли» истину
#
# Ниже — важности по признакам в двух сценариях, где расхождение ожидается
# теоретически: взаимодействие (главный эффект x1 и x2 равен нулю) и
# коррелированные двойники при ρ = 0.9.

# %%
for key in [("interaction", "hgb"), ("interaction", "ridge"), ("correlated ρ=0.9", "rf"), ("nonlinear", "ridge")]:
    res, data = runs[key]
    imp = res.importances().assign(true=data.true_importance)
    print(f"\n{key[0]}, {key[1]}: R² теста {res.r2_test:.3f}")
    print(imp.round(3).to_string())

# %% [markdown]
# ### Кривые для нелинейного сценария
#
# PDP, ALE и зависимость SHAP (тест) против истинного главного эффекта, бустинг.
# Все кривые центрированы.

# %%
res, data = runs[("nonlinear", "hgb")]
colors = {"pdp": "#2a78d6", "ale": "#eb6834", "shap": "#1baf7a"}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
for ax, f in zip(axes, ["x1", "x2", "x3", "x4"]):
    for method in ("pdp", "ale", "shap"):
        c = res[(method, "test")].curves[f]
        w = c["count"].to_numpy(float)
        v = c["value"].to_numpy(float)
        v = v - np.nansum(v * w) / w[~np.isnan(v)].sum()
        ax.plot(c["x"], v, color=colors[method], lw=2, label=method.upper())
    t = M.true_curves({f: c}, data.true_curve)[f]
    tv = t["value"] - np.average(t["value"], weights=t["count"])
    ax.plot(t["x"], tv, color="#555555", lw=2, ls="--", label="истина")
    ax.set_title(f)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("эффект на прогноз (центр.)")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Нелинейный сценарий, R² = 0.5, HGB: кривые эффекта на тесте")
fig.tight_layout()
fig.savefig(FIGURES / "04_nonlinear_curves.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Нулевой эталон на перемешанной цели
#
# Независимые признаки, R² = 0.5, RF. 20 перестановок, без LIME (его время ниже
# оценивается отдельно). У истинных признаков p должно быть минимально
# возможным (1/21 ≈ 0.048), у шумовых — распределено примерно равномерно.

# %%
data = make_synthetic("independent", n=N, r2=0.5, seed=SEED)
X_train, X_test, y_train, y_test = split(data)
obs = E.fit_and_explain("rf", X_train, y_train, X_test, y_test, methods=("pi", "shap"), seed=SEED)
t0 = time.perf_counter()
null, null_r2 = E.null_importances("rf", X_train, y_train, X_test, y_test, n_perm=20, seed=SEED,
                                   n_jobs=N_JOBS, methods=("pi", "shap"))
print(f"20 перестановок RF (PI + SHAP): {time.perf_counter() - t0:.1f} с параллельно")
print("R² на тесте при перемешанной цели: среднее", round(null_r2.mean(), 3), "| макс", round(null_r2.max(), 3))
pv = pd.DataFrame({k: M.null_pvalues(obs.importances()[k], null[k]) for k in null})
pv["relevant"] = pv.index.isin(data.relevant)
print(pv.round(3).to_string())
print("\nНулевая важность SHAP на обучении (среднее по перестановкам) против теста:")
print(pd.DataFrame({"null_shap_train": null["shap_train"].mean(), "null_shap_test": null["shap_test"].mean(),
                    "null_pi_test": null["pi_test"].mean()}).round(4).T.to_string())

# %% [markdown]
# ## 5. LIME: сколько соседей нужно
#
# Один и тот же объект с двумя разными seed. Если коэффициенты сильно меняются,
# объяснение — шум сэмплирования. Мера: ρ Спирмена между |коэффициентами| двух
# seed (медиана по 50 объектам) и средняя разница коэффициентов, делённая на их
# средний модуль. Модель RF, независимые признаки, R² = 0.5.

# %%
pipe = make_pipeline("rf", other_cols=list(data.X.columns), random_state=SEED)
Z_train, Z_test = fit_transform(pipe, X_train, y_train, X_test)
model = pipe.named_steps["model"]
Z50 = Z_test.iloc[:50].reset_index(drop=True)
rows = []
for ns in (1000, 2000, 5000, 10000):
    a = E.lime_values(model, Z_train, Z50, "test", seed=1, num_samples=ns)
    b = E.lime_values(model, Z_train, Z50, "test", seed=2, num_samples=ns)
    rho = [M.rank_agreement(a.local.iloc[i].abs(), b.local.iloc[i].abs())["spearman"] for i in range(len(Z50))]
    rel = (a.local - b.local).abs().to_numpy().mean() / a.local.abs().to_numpy().mean()
    top4 = np.mean([M.top_k_overlap(a.local.iloc[i].abs(), b.local.iloc[i].abs(), 4) for i in range(len(Z50))])
    glob = M.rank_agreement(a.importance, b.importance)["spearman"]
    rows.append({"num_samples": ns, "сек на объект": a.seconds / len(Z50), "median ρ между seed": np.median(rho),
                 "top-4 совпадение": top4, "отн. разница коэф.": rel, "ρ глобальных": glob,
                 "local R² (медиана)": a.info["local_r2"].median()})
lime_ns = pd.DataFrame(rows)
print(lime_ns.round(3).to_string(index=False))

# %% [markdown]
# ## 6. Время работы и бюджет на E1–E4
#
# Последовательный замер (один процесс) для трёх размеров задачи:
# - синтетика, 10 признаков (E1);
# - синтетика, 34 независимых признака (размер реальной задачи);
# - реальные 34 признака `extra` (E2, E3). Выводится только время.
#
# SHAP и LIME объясняют по 200 объектов в каждой части (обучение и тест).

# %%
def time_one(X_train, X_test, y_train, y_test, log_cols=(), other_cols=None, label=""):
    rows = []
    for model_name in MODELS:
        res = E.fit_and_explain(model_name, X_train, y_train, X_test, y_test, log_cols=log_cols,
                                other_cols=other_cols, curve_features=curve_feats, seed=SEED)
        t = res.timings()
        rows.append({"задача": label, "model": model_name, "fit": res.seconds_fit, **t.to_dict(),
                     "всё без LIME": res.seconds_fit + t.drop(["lime_train", "lime_test"]).sum(),
                     "всего": res.seconds_fit + t.sum()})
    return rows


timing_rows = []
d10 = make_synthetic("independent", n=N, r2=0.5, seed=SEED)
curve_feats = None
timing_rows += time_one(*split(d10), label="синт., 10 призн.")
d34 = make_synthetic("independent", n=N, r2=0.5, n_features=34, seed=SEED)
timing_rows += time_one(*split(d34), label="синт., 34 призн.")
X_real, y_real = load_real("extra")
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_real, y_real, test_size=TEST_SIZE, random_state=SEED)
curve_feats = CONTINUOUS
timing_rows += time_one(Xr_tr, Xr_te, yr_tr, yr_te, log_cols=FEATURES_LOG, other_cols=FEATURES_OTHER,
                        label="реальные, 34 призн.")
del X_real, y_real, Xr_tr, Xr_te, yr_tr, yr_te
timing = pd.DataFrame(timing_rows)
timing.round(3).to_csv(RESULTS / "framework_check_timing.csv", index=False)
print(timing.round(2).to_string(index=False))

# %% [markdown]
# ### Оценка бюджета
#
# Число прогонов «обучить модель + все объяснения» по плану:
# - E1: 4 сценария (+ρ 0.5 и 0.9) = 5 конфигураций × 5 уровней R² × 3 модели × 10 повторов seed;
# - E2: 3 уровня R² × 3 модели × 10 повторов;
# - E3: 2 черты × 3 модели × (25 фолдов 5×5 CV + 100 перестановок нулевого эталона);
# - E4: порядка 100 дополнительных прогонов (ширина ядра LIME, seed, группы).
#
# Время одного прогона берём из таблицы выше (среднее по моделям), параллельно
# на 6 процессах (берём ускорение 5×).

# %%
per_run = timing.groupby("задача")[["всего", "всё без LIME"]].mean()
plan = {
    "E1": (5 * 5 * 3 * 10, "синт., 10 призн."),
    "E2": (3 * 3 * 10, "реальные, 34 призн."),
    "E3 CV": (2 * 3 * 25, "реальные, 34 призн."),
    "E3 нулевой эталон": (2 * 3 * 100, "реальные, 34 призн."),
    "E4": (100, "реальные, 34 призн."),
}
budget = pd.DataFrame([{"часть": k, "прогонов": n, "задача": task,
                        "часы, всё": n * per_run.loc[task, "всего"] / 3600 / 5,
                        "часы, без LIME": n * per_run.loc[task, "всё без LIME"] / 3600 / 5}
                       for k, (n, task) in plan.items()])
print(budget.round(2).to_string(index=False))
print("Итого часов (всё / без LIME):", budget["часы, всё"].sum().round(1), "/", budget["часы, без LIME"].sum().round(1))

# %% [markdown]
# ## Выводы
#
# По прогону 2026-11-07 (подробнее в `docs/methodology.md`, разделы 10–11).
#
# 1. **Генераторы** дают заданный R² (отклонение до 0.006) и корреляции.
# 2. **Реализации верны:** PDP совпадает со sklearn точно, ALE с PyALE до 1e-16,
#    PI со sklearn в пределах 2 SE. Аддитивность SHAP до 1e-8 после исправления
#    float32 для HGB (без него ошибка была до 0.2, см. `models.to_float32_precision`).
# 3. **При сильном сигнале** в сценариях independent и correlated ρ = 0.5 все пять
#    методов у всех трёх моделей нашли все истинные признаки, знаки совпали везде,
#    CCC кривых с истиной 0.88–0.99.
# 4. **Провалы объясняются, ошибок кода не найдено:**
#    - PDP и ALE не видят x1, x2 во взаимодействии: главный эффект там действительно
#      равен нулю (свойство метода);
#    - Ridge не выражает x² и x1·x2 (свойство модели);
#    - HGB с заданными настройками плохо выучил x1·x2 (R² 0.23 из 0.5); PI это
#      взаимодействие видит, SHAP и LIME почти нет. Разобрать в E1;
#    - при ρ = 0.9 RF отдаёт вклад двойнику x5 больше, чем x4, и все методы с этим
#      согласны (модель не совпадает с истиной, а методы описывают модель верно).
# 5. **Нулевой эталон работает:** у истинных признаков p минимально, у шумовых по PI
#    p 0.2–0.6. По SHAP эталон для шумовых признаков консервативен (p ≈ 1): в модели
#    с сигналом шум получает меньше разбиений, чем в модели на чистом шуме.
# 6. **LIME:** 5000 соседей дают ρ 0.98 между двумя seed, этого достаточно.
# 7. **Время:** один прогон всех методов 1–3 с (Ridge), 13–15 с (RF), 4–6 с (HGB),
#    из них 72–96% приходится на LIME. Весь план E1–E4 около 0.7 часа на 6 процессах,
#    так что вычисления не ограничение.

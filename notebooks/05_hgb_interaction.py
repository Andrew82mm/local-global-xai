# %% [markdown]
# # 05. Разбор: HGB и взаимодействие x1·x2
#
# В ноутбуке 04 (сценарий interaction, R² = 0.5, n = 1000) у бустинга PI нашёл
# x1 и x2 (margin 5.4), а SHAP и LIME почти нет: |SHAP| у x1 и x2 около 0.04
# при истинных 0.16 и при шумовых признаках до 0.05.
#
# Это баг или свойство метода? Проверяем по шагам:
# 1. аддитивность SHAP на этих данных;
# 2. TreeExplainer против точного перебора всех коалиций (свой код и `shap.explainers.Exact`);
# 3. что модель реально выучила: PDP по x1 и поверхность прогноза на сетке x1 × x2;
# 4. влияет ли округление до float32;
# 5. механизм: почему PI и SHAP дают разную картину при одной и той же модели.
#
# Запуск: `uv run python notebooks/05_hgb_interaction.py` (около минуты).

# %%
import sys
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from xai_consistency import explain as E  # noqa: E402
from xai_consistency import metrics as M  # noqa: E402
from xai_consistency.models import fit_transform, make_pipeline  # noqa: E402
from xai_consistency.synthetic import make_synthetic  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 30)
SEED = 42
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Те же данные и модель, что в ноутбуке 04
data = make_synthetic("interaction", n=1000, r2=0.5, seed=SEED)
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=SEED)
pipe = make_pipeline("hgb", other_cols=list(data.X.columns), random_state=SEED)
Z_train, Z_test = fit_transform(pipe, X_train, y_train, X_test)
model = pipe.named_steps["model"]
rng = np.random.default_rng(SEED)
background = Z_train.iloc[rng.choice(len(Z_train), 100, replace=False)]
print(f"R² теста: {model.score(Z_test.to_numpy(), y_test):.3f}")

# Коэффициент истинного взаимодействия: signal содержит c·x1·x2
c_true = data._scales[0][1]
print(f"Истинный коэффициент при x1·x2: {c_true:.3f}")

# %% [markdown]
# ## 1. Аддитивность SHAP

# %%
sh_test = E.shap_values(model, "hgb", background, Z_test, "test")
sh_train = E.shap_values(model, "hgb", background, Z_train.iloc[:200], "train")
for name, sh, Z in [("test", sh_test, Z_test), ("train", sh_train, Z_train.iloc[:200])]:
    err = sh.info["expected_value"] + sh.local.sum(axis=1) - model.predict(Z.to_numpy())
    print(f"{name}: max |сумма SHAP + E[f] − прогноз| = {np.abs(err).max():.2e}, "
          f"E[f] = {sh.info['expected_value']:.4f}, среднее прогноза по фону = {model.predict(background.to_numpy()).mean():.4f}")

# %% [markdown]
# ## 2. TreeExplainer против точного перебора
#
# Интервенционный SHAP с фоном B — это значения Шепли игры
# v(S) = среднее по z из B от f(x_S, z_{−S}). При 10 признаках можно перебрать все
# 1024 коалиции: 1024 × 100 прогнозов на объект.

# %%
objs = Z_test.iloc[:25]
bg = background.to_numpy()
exact = np.array([E.exact_interventional_shap(model, x, bg) for x in objs.to_numpy()])
tree = sh_test.local.iloc[:25].to_numpy()
ex_shap = shap.explainers.Exact(model.predict, shap.maskers.Independent(bg, max_samples=len(bg)))
lib_exact = ex_shap(objs.to_numpy()).values
print("max |TreeExplainer − свой перебор|:     ", f"{np.abs(tree - exact).max():.2e}")
print("max |shap.Exact − свой перебор|:        ", f"{np.abs(lib_exact - exact).max():.2e}")
print("Размах значений SHAP на этих объектах:   ", f"{np.abs(exact).max():.3f}")
print("\nmean |SHAP| на 25 объектах (перебор / TreeExplainer):")
print(pd.DataFrame({"exact": np.abs(exact).mean(0), "tree": np.abs(tree).mean(0)}, index=Z_test.columns).round(4).T)

# %% [markdown]
# ## 3. Что выучила модель
#
# Двумерный PDP по сетке x1 × x2 (остальные признаки из теста) раскладывается на
# главные эффекты и чистое взаимодействие:
# h(a, b) = PD₁₂(a, b) − PD₁(a) − PD₂(b) + среднее.
# Подгоняем h ≈ ĉ·a·b и сравниваем ĉ с истинным коэффициентом.

# %%
def pd2(model, Z, f1, f2, grid):
    Zv = Z.to_numpy()
    j1, j2 = Z.columns.get_loc(f1), Z.columns.get_loc(f2)
    out = np.zeros((len(grid), len(grid)))
    for a_i, a in enumerate(grid):
        big = np.tile(Zv, (len(grid), 1))
        big[:, j1] = a
        big[:, j2] = np.repeat(grid, len(Zv))
        out[a_i] = model.predict(big).reshape(len(grid), len(Zv)).mean(axis=1)
    return out


def learned_interaction(model, Z, grid):
    pd12 = pd2(model, Z, "x1", "x2", grid)
    h = pd12 - pd12.mean(axis=1, keepdims=True) - pd12.mean(axis=0, keepdims=True) + pd12.mean()
    A, B = np.meshgrid(grid, grid, indexing="ij")
    c_hat = np.sum(h * A * B) / np.sum((A * B) ** 2)
    h_stat = h.var() / (pd12 - pd12.mean()).var()  # доля взаимодействия в поверхности (H² Фридмана на сетке)
    return pd12, h, c_hat, h_stat


grid = np.linspace(-2, 2, 17)
pd12, h, c_hat, h_stat = learned_interaction(model, Z_test, grid)
print(f"HGB, n=1000: ĉ = {c_hat:.3f} (истина {c_true:.3f}, доля {c_hat / c_true:.0%}); "
      f"доля взаимодействия в поверхности x1×x2: {h_stat:.2f}")

pdp = E.partial_dependence(model, Z_test, E.make_grids(Z_test, ["x1", "x2", "x3"]))
print("Размах 1D PDP (SD кривой): x1 = {:.3f}, x2 = {:.3f}, x3 = {:.3f}".format(
    *(pdp.importance[f] for f in ["x1", "x2", "x3"])))

# Сравнение с RF и с бустингом на большей выборке
compare = {}
for label, model_name, n in [("HGB n=1000", "hgb", 1000), ("RF n=1000", "rf", 1000), ("HGB n=5000", "hgb", 5000)]:
    d = make_synthetic("interaction", n=n, r2=0.5, seed=SEED)
    Xa, Xb, ya, yb = train_test_split(d.X, d.y, test_size=0.2, random_state=SEED)
    res = E.fit_and_explain(model_name, Xa, ya, Xb, yb, methods=("pi", "shap"), seed=SEED)
    m = make_pipeline(model_name, other_cols=list(d.X.columns), random_state=SEED)
    Za, Zb = fit_transform(m, Xa, ya, Xb)
    _, _, ch, hs = learned_interaction(m.named_steps["model"], Zb, grid)
    imp = res.importances()
    noise = [c for c in d.X.columns if c not in d.relevant]
    compare[label] = {"R² теста": res.r2_test, "ĉ / c": ch / c_true, "H²": hs,
                      "|SHAP| x1 тест": imp.loc["x1", "shap_test"], "|SHAP| макс шум тест": imp.loc[noise, "shap_test"].max(),
                      "PI x1": imp.loc["x1", "pi_test"], "PI макс шум": imp.loc[noise, "pi_test"].max(),
                      "margin SHAP": M.fidelity(imp["shap_test"], d.true_importance, d.relevant)["margin"],
                      "margin PI": M.fidelity(imp["pi_test"], d.true_importance, d.relevant)["margin"]}
compare = pd.DataFrame(compare).T
print(compare.round(3).to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
A, B = np.meshgrid(grid, grid, indexing="ij")
for ax, surf, title in [(axes[0], c_true * A * B, "истина: c·x1·x2"), (axes[1], h, "HGB n=1000: взаимодействие в прогнозе")]:
    im = ax.contourf(grid, grid, surf.T, levels=np.linspace(-1.3, 1.3, 14), cmap="RdBu_r")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title, fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.85, label="вклад в прогноз")
fig.savefig(FIGURES / "05_hgb_interaction_surface.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Влияет ли округление до float32
#
# Обучаем ту же модель без шага округления и сравниваем прогнозы и SHAP на тесте.

# %%
no_round = make_pipeline("hgb", other_cols=list(data.X.columns), random_state=SEED)
no_round.steps[0][1].steps.pop()  # убираем шаг float32
no_round.fit(X_train, y_train)
Zt_raw = pd.DataFrame(no_round.named_steps["prep"].transform(X_test), columns=Z_test.columns)
m_raw = no_round.named_steps["model"]
bg_raw = pd.DataFrame(no_round.named_steps["prep"].transform(X_train), columns=Z_test.columns).iloc[background.index]
sh_raw = E.shap_values(m_raw, "hgb", bg_raw, Zt_raw, "test")
print("max |прогноз с округлением − без|:", f"{np.abs(model.predict(Z_test.to_numpy()) - m_raw.predict(Zt_raw.to_numpy())).max():.2e}")
print(pd.DataFrame({"с округлением": sh_test.importance, "без округления": sh_raw.importance}).round(4).T)

# %% [markdown]
# ## 5. Механизм: почему PI видит взаимодействие, а SHAP почти нет
#
# Оценим, каким должен быть |SHAP| x1 при выученном ĉ, и сравним шум у двух методов.

# %%
x1x2 = (Z_test["x1"] * Z_test["x2"]).abs().mean()
print(f"Ожидаемый |SHAP| x1 от выученного взаимодействия: ½·|ĉ|·E|x1·x2| = {0.5 * abs(c_hat) * x1x2:.3f}"
      f" (наблюдаемый {sh_test.importance['x1']:.3f}, при истинном c: {0.5 * c_true * x1x2:.3f})")
noise = [c for c in data.X.columns if c not in data.relevant]
pi = E.permutation_importance(model, Z_test, y_test, n_repeats=30, seed=SEED)
var_pred = model.predict(Z_test.to_numpy()).var()
print(f"\nPI измеряет падение R² (квадратичная шкала), SHAP — средний |вклад| (линейная шкала).")
print(f"PI:   x1 = {pi.importance['x1']:.3f}, x3 = {pi.importance['x3']:.3f}, шум макс = {pi.importance[noise].max():.3f}, "
      f"шум среднее = {pi.importance[noise].mean():+.3f}")
print(f"SHAP: x1 = {sh_test.importance['x1']:.3f}, x3 = {sh_test.importance['x3']:.3f}, шум макс = {sh_test.importance[noise].max():.3f}, "
      f"шум среднее = {sh_test.importance[noise].mean():.3f}")
# Какую долю дисперсии прогноза дают шумовые признаки (используются моделью, но не обобщаются)
print(f"SD прогноза на тесте: {np.sqrt(var_pred):.3f}")
print("Сумма |SHAP| шумовых признаков / сумма |SHAP| всех:", round(sh_test.importance[noise].sum() / sh_test.importance.sum(), 3))

# %% [markdown]
# ### Проверка формулы для PI
#
# Пусть истинный вклад t = c·x1·x2, а часть модели, зависящая от x1, g(x1, x2)
# (у чистого взаимодействия среднее по x1 равно нулю). Перестановка x1 даёт g' и
#
# PI = E[(g − g')²] + 2·E[(y − f)(g − g')] = 2·Var g + 2·cov(t, g) − 2·Var g = **2·cov(t, g)**.
#
# То есть PI на тесте — это ковариация выученной части с **истиной**. Слагаемое
# «насколько модель использует признак» (Var g) сокращается. Отсюда:
# - шумовой признак: cov = 0, PI ≈ 0, сколько бы модель его ни использовала;
# - слабо выученное сильное взаимодействие: PI ≈ 2·c·ĉ, то есть мало выученное ĉ
#   умножено на большое истинное c.
#
# SHAP измеряет сам вклад в прогноз: |SHAP x1| ≈ ½·|ĉ|·E|x1·x2|, без множителя c.
# Шумовые признаки модель тоже использует (подгонка под шум), и их |SHAP| не
# сокращается. Поэтому margin у PI большой, а у SHAP меньше 1.

# %%
Zp = Z_test.to_numpy().copy()
j1 = Z_test.columns.get_loc("x1")
g_minus = []
for r in range(30):
    Zq = Zp.copy()
    Zq[:, j1] = rng.permutation(Zq[:, j1])
    g_minus.append(model.predict(Zp) - model.predict(Zq))  # g − g'
g_minus = np.array(g_minus)
t_test = c_true * X_test["x1"].to_numpy() * X_test["x2"].to_numpy()
print(f"2·c·ĉ·E[(x1·x2)²] = {2 * c_true * c_hat * np.mean(t_test**2) / c_true**2:.3f}")
# E[t·(g − g')] = cov(t, g), потому что t и переставленное g' независимы
print(f"2·cov(t, g) по перестановкам, без шума y: {2 * np.mean(t_test * g_minus):.3f}")
print(f"PI x1 (30 повторов): {pi.importance['x1']:.3f}")

# %% [markdown]
# ### Почему деревья не выучили x1·x2
#
# У x1 и x2 нет главного эффекта: E[y | x1] не зависит от x1. Жадное дерево
# выбирает разбиение по выигрышу на одном шаге, и разбиение по x1 в корне почти
# ничего не даёт. Взаимодействие можно поймать, только если сначала случайно
# разбить по x1, а потом по x2 (классическая задача XOR). Проверим, где
# в деревьях бустинга встречаются x1 и x2.

# %%
roots, all_splits = [], []
for (pred,) in model._predictors:
    nodes = pred.nodes
    inner = nodes[~nodes["is_leaf"].astype(bool)]
    roots.append(Z_test.columns[nodes[0]["feature_idx"]])
    all_splits += [Z_test.columns[i] for i in inner["feature_idx"]]
print("Признак в корне деревьев (доля):", pd.Series(roots).value_counts(normalize=True).round(2).to_dict())
print("Доля всех разбиений:", pd.Series(all_splits).value_counts(normalize=True).round(3).to_dict())

# Контроль: тот же бустинг, если добавить x1·x2 как признак, выучивает его сразу
d_ctrl = data.X.assign(x1x2=data.X["x1"] * data.X["x2"])
Xa, Xb, ya, yb = train_test_split(d_ctrl, data.y, test_size=0.2, random_state=SEED)
ctrl = make_pipeline("hgb", other_cols=list(d_ctrl.columns), random_state=SEED).fit(Xa, ya)
print(f"\nКонтроль: HGB с готовым признаком x1·x2, R² теста = {ctrl.score(Xb, yb):.3f} (было {model.score(Z_test.to_numpy(), y_test):.3f})")

# %% [markdown]
# ## Выводы
#
# **Бага нет.** SHAP посчитан верно: аддитивность до 1e-8, TreeExplainer совпадает с
# точным перебором всех 1024 коалиций до 3e-9, а `shap.Exact` с перебором до 1e-15.
# Округление до float32 не меняет ни прогнозы, ни SHAP. Тест `test_tree_shap_matches_exact_enumeration`
# закрепляет это для RF и HGB.
#
# **Модель почти не выучила взаимодействие.** Коэффициент выученного x1·x2 равен
# 8% истинного (ĉ = 0.04 при c = 0.50). R² 0.23 почти целиком даёт аддитивная
# часть (x3, x4): её потолок R² = 0.25. Ожидание «при R² 0.23 вклад x1 и x2
# должен быть заметным» неверно: высокий R² получен не за счёт взаимодействия.
# Причина в жадном построении деревьев. У x1 и x2 нет главного эффекта, поэтому
# разбиение по ним в корне ничего не даёт: x1 и x2 почти не встречаются в корне,
# там x3 в 65% деревьев. Это классическая задача XOR. RF и HGB на n = 5000 тоже
# выучивают не больше 6–11%. Контроль: с готовым признаком x1·x2 тот же бустинг
# получает R² 0.45.
#
# **Почему PI всё равно видит x1.** PI на тесте ≈ 2·cov(истинный эффект, выученная
# часть модели). Слагаемое «насколько модель использует признак» сокращается.
# Для слабо выученного сильного взаимодействия PI ≈ 2·c·ĉ: маленькое ĉ умножается на
# большое c. Оценки 0.033–0.036 против наблюдаемых 0.044, остаток сопоставим с шумом PI.
# У шумовых признаков ковариация с истиной 0, поэтому PI ≈ 0.
#
# **Почему SHAP почти не видит x1.** SHAP измеряет сам вклад x1 в прогноз модели.
# От выученного взаимодействия это ½·|ĉ|·E|x1·x2| ≈ 0.012, остальное (до 0.033) дают
# разбиения по x1, которые не повторяют истину. Шумовые признаки модель тоже
# использует, подгоняясь под шум: средний |SHAP| шума 0.028, и этот уровень не
# сокращается. Отсюда margin < 1.
#
# **Что из этого следует для работы.** PI на тесте и SHAP отвечают на разные вопросы:
# PI — «сколько верного выучено про признак», SHAP — «насколько прогноз модели зависит
# от признака». Их расхождение здесь — свойство методов, его стоит показать в E1.
# Отдельно надо разделять провал метода и провал модели: в сценарии interaction
# для этого полезно смотреть ĉ / c.

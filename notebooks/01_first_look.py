# %% [markdown]
# # 01. Знакомство с методами: все пять объяснений на одной модели
#
# Цель: увидеть, что выдаёт каждый метод, на понятной задаче регрессии.
# Датасет California Housing: предсказываем медианную цену дома в районе
# (в сотнях тысяч $) по 8 признакам.
#
# Файл разбит на ячейки `# %%`: в VS Code их можно запускать по одной
# (Run Cell), как в Jupyter.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from PyALE import ale
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = RandomForestRegressor(
    n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)
print(f"R^2 на тесте: {model.score(X_test, y_test):.3f}")

# Для объяснений берём подвыборку теста: SHAP и LIME считаются по объекту,
# на всех 4000 объектах это долго.
X_explain = X_test.sample(300, random_state=RANDOM_STATE)

# %% [markdown]
# ## 1. Permutation Importance (глобальный)
# Перемешиваем один признак и смотрим, насколько упало качество модели (R^2).
# Чем больше падение, тем важнее признак *для качества*.

# %%
pi = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)
pi_importance = pd.Series(pi.importances_mean, index=X.columns).sort_values()
pi_importance.plot.barh(title="Permutation Importance (падение R^2)")
plt.show()

# %% [markdown]
# ## 2. SHAP (локальный → глобальный)
# Для каждого объекта: на сколько каждый признак сдвинул прогноз
# относительно среднего прогноза. Сумма вкладов = прогноз − среднее.
# Глобальная важность = среднее |SHAP| по объектам.

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_explain)

# Локальное объяснение одного объекта
shap.plots.waterfall(shap_values[0])

# %%
# Глобальная картина: каждая точка — один объект
shap.plots.beeswarm(shap_values)

# %%
shap_importance = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=X.columns
).sort_values()
shap_importance.plot.barh(title="Глобальная важность: mean |SHAP|")
plt.show()

# %% [markdown]
# ## 3. LIME (локальный)
# Вокруг объекта генерируем «соседей», смотрим прогнозы модели на них и
# обучаем на этих соседях линейную модель. Её коэффициенты = объяснение.

# %%
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=list(X.columns),
    mode="regression",
    random_state=RANDOM_STATE,
)
exp = lime_explainer.explain_instance(
    X_explain.iloc[0].values, model.predict, num_features=len(X.columns)
)
exp.as_pyplot_figure()
plt.show()

# %%
# Агрегируем LIME в глобальную важность так же, как SHAP: mean |вклад|.
# На 300 объектах займёт около минуты.
lime_weights = []
for _, row in X_explain.iterrows():
    e = lime_explainer.explain_instance(
        row.values, model.predict, num_features=len(X.columns)
    )
    # as_map() возвращает пары (индекс признака, вес)
    w = dict(e.as_map()[1])
    lime_weights.append([w[i] for i in range(len(X.columns))])
lime_weights = np.array(lime_weights)

lime_importance = pd.Series(
    np.abs(lime_weights).mean(axis=0), index=X.columns
).sort_values()
lime_importance.plot.barh(title="Глобальная важность: mean |LIME|")
plt.show()

# %% [markdown]
# ## 4. PDP (глобальный)
# Для значения признака v: подставляем v всем объектам, усредняем прогноз.
# Тонкие линии (ICE) — то же самое для каждого объекта отдельно.

# %%
PartialDependenceDisplay.from_estimator(
    model,
    X_explain,
    features=["MedInc", "AveOccup", "Latitude"],
    kind="both",  # PDP + ICE
    subsample=50,
    random_state=RANDOM_STATE,
)
plt.show()

# %% [markdown]
# ## 5. ALE (глобальный)
# Как PDP, но считает изменение прогноза только внутри узких интервалов
# признака, на реальных объектах из этого интервала. Поэтому не создаёт
# нереалистичных комбинаций признаков (важно при корреляциях).

# %%
for feature in ["MedInc", "AveOccup", "Latitude"]:
    ale(X=X_explain, model=model, feature=[feature], grid_size=20, include_CI=False)
    plt.show()

# %% [markdown]
# ## Первое сравнение: согласуются ли ранги важности?
# Это и есть основная метрика темы: ранговая корреляция Спирмена.

# %%
importances = pd.DataFrame(
    {"PI": pi_importance, "SHAP": shap_importance, "LIME": lime_importance}
)
print(importances.sort_values("SHAP", ascending=False))
print("\nКорреляция Спирмена между рангами:")
print(importances.corr(method="spearman").round(2))

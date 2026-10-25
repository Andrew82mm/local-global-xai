# %% [markdown]
# # 02. Игрушечный пример расхождения: взаимодействие признаков
#
# Истинная зависимость: y = x1 * x2 + x3, все признаки ~ U(-1, 1), независимы.
#
# - Локально x1 важен: при x2 = 1 он сильно увеличивает y, при x2 = -1 уменьшает.
# - В среднем по x2 эффект x1 равен нулю, поэтому PDP/ALE по x1 будут почти
#   плоскими, хотя признак «важный».
#
# Это главный механизм расхождения локальных и глобальных объяснений.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

rng = np.random.default_rng(0)
n = 3000
X = pd.DataFrame(rng.uniform(-1, 1, size=(n, 3)), columns=["x1", "x2", "x3"])
y = X["x1"] * X["x2"] + X["x3"] + rng.normal(0, 0.05, n)

model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=0)
model.fit(X, y)
print(f"R^2 на обучении: {model.score(X, y):.3f}")

# %% [markdown]
# ## PDP + ICE по x1
# PDP (жирная линия) почти горизонтальна, а ICE-линии отдельных объектов
# идут и вверх, и вниз: локальные эффекты разных знаков гасят друг друга.

# %%
PartialDependenceDisplay.from_estimator(
    model, X, features=["x1", "x3"], kind="both", subsample=100, random_state=0
)
plt.show()

# %% [markdown]
# ## SHAP: вклад x1 зависит от x2
# Наклон облака точек по x1 меняет знак в зависимости от цвета (значения x2).

# %%
shap_values = shap.TreeExplainer(model)(X.iloc[:500])
shap.plots.scatter(shap_values[:, "x1"], color=shap_values[:, "x2"])

# %% [markdown]
# ## А вот важность при этом согласуется
# PI и mean |SHAP| видят, что x1 важен, в отличие от формы эффекта по PDP.
# Вывод: согласованность нужно проверять отдельно для важности и для формы эффекта.

# %%
pi = permutation_importance(model, X, y, n_repeats=10, random_state=0)
print(
    pd.DataFrame(
        {
            "PI": pi.importances_mean,
            "mean|SHAP|": np.abs(shap_values.values).mean(axis=0),
        },
        index=X.columns,
    ).round(3)
)

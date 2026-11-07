"""Модели и предобработка.

Схема: Pipeline(prep, model). prep обучается только на обучающей части фолда.
Объясняем последний шаг (model) на преобразованной матрице Z = prep.transform(X),
см. docs/methodology.md, раздел 1.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MODELS = ("ridge", "rf", "hgb")
MISSING_SUFFIX = "__missing"


def to_float32_precision(Z):
    """Округление до точности float32 (тип остаётся float64).

    Зачем: TreeExplainer считает деревья в float32. Пороги HGB совпадают с
    обучающими значениями, и после округления точки из обучения (объясняемые и
    фоновые) уходили на другую сторону порога: сумма интервенционных SHAP
    расходилась с прогнозом до 0.2 при SD прогноза около 0.3. Если модель сразу
    учится на значениях, точно представимых в float32, округление внутри shap
    ничего не меняет и аддитивность выполняется до 1e-7.
    """
    return np.asarray(Z, dtype=np.float32).astype(np.float64)


def make_model(name, random_state=42):
    """Гиперпараметры заданы заранее (как в ноутбуке 03) и не подбираются."""
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-1, 4, 20))
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33,
                                     n_jobs=1, random_state=random_state)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3,
                                             min_samples_leaf=20, early_stopping=False,
                                             random_state=random_state)
    raise ValueError(f"Неизвестная модель {name!r}")


def make_pipeline(model_name, log_cols=(), other_cols=(), indicator_cols=(), random_state=42):
    """log1p для log_cols, медианная импутация и масштабирование для всех.

    indicator_cols: для каких колонок добавить индикатор пропуска. Список задаётся
    явно, а не через add_indicator=True: так набор колонок Z не зависит от того,
    попал ли пропуск в обучающую часть фолда. Индикатор получает имя
    '<признак>__missing' и не масштабируется.
    """
    tail = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    parts = []
    if log_cols:
        log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        parts.append(("log", Pipeline([("log1p", log1p), *tail]), list(log_cols)))
    if other_cols:
        parts.append(("other", Pipeline(tail), list(other_cols)))
    if indicator_cols:
        parts.append(("miss", MissingIndicator(features="all"), list(indicator_cols)))
    prep = Pipeline([
        ("columns", ColumnTransformer(parts, verbose_feature_names_out=False)),
        ("float32", FunctionTransformer(to_float32_precision, feature_names_out="one-to-one")),
    ])
    return Pipeline([("prep", prep), ("model", make_model(model_name, random_state))])


def transformed_names(pipe, indicator_cols=()):
    """Имена колонок Z. Индикаторы переименованы в '<признак>__missing'."""
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    miss = [f"missingindicator_{c}" for c in indicator_cols]
    return [n.replace("missingindicator_", "") + MISSING_SUFFIX if n in miss else n for n in names]


def base_features(names):
    """Исходные признаки без индикаторов пропуска: по ним считаем согласованность."""
    return [n for n in names if not n.endswith(MISSING_SUFFIX)]


def fit_transform(pipe, X_train, y_train, X_test, indicator_cols=()):
    """Обучает Pipeline и возвращает (Z_train, Z_test) как DataFrame с именами колонок."""
    pipe.fit(X_train, y_train)
    names = transformed_names(pipe, indicator_cols)
    prep = pipe.named_steps["prep"]
    Z_train = pd.DataFrame(prep.transform(X_train), columns=names).astype(float)
    Z_test = pd.DataFrame(prep.transform(X_test), columns=names).astype(float)
    return Z_train, Z_test

# Согласованность локальных и глобальных объяснений моделей МО

Курсовая работа. Сравнение методов объяснения LIME, SHAP, Permutation Importance,
PDP и ALE: насколько глобальные закономерности модели соответствуют локальным
объяснениям и почему возникают расхождения.

## Установка

Нужен [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Запуск

Ноутбуки хранятся как `.py`-файлы с ячейками `# %%`. В VS Code их можно
запускать по ячейкам (Run Cell), как в Jupyter. Также можно запустить файл целиком:

```bash
uv run python notebooks/01_first_look.py
```

## Структура

```
notebooks/                 эксперименты, по порядку
  01_first_look.py           все пять методов на California Housing
  02_interaction_toy.py      пример расхождения из-за взаимодействия признаков
  03_data_exploration.py     исследование реальных данных
  04_framework_check.py      проверка общего кода на синтетике, время методов
  05_hgb_interaction.py      разбор: почему PI видит x1·x2 у бустинга, а SHAP нет
  06_e1_synthetic.py         E1: синтетика при разном уровне сигнала
  07_e2_semisynthetic.py     E2: полусинтетика на реальных признаках
  08_e3_real.py              E3: extra и emo, нулевой эталон и устойчивость
src/xai_consistency/       общий код
  data.py                    кеш реальных данных и 34 признака
  synthetic.py               синтетика с известной истиной
  semisynthetic.py           истинная функция E2 на реальных признаках
  models.py                  Pipeline и модели Ridge, RF, HGB
  explain.py                 PI, PDP, ALE, SHAP, LIME в едином формате, нулевой эталон
  metrics.py                 метрики согласованности
tests/                     тесты генераторов, метрик и методов
data/                      локальные данные (не в git)
results/                   таблицы и графики (не в git)
docs/                      литература, исследование данных, методика
```

Решения о настройках методов: [docs/methodology.md](docs/methodology.md).
Результаты E1: [docs/e1_results.md](docs/e1_results.md), E2: [docs/e2_results.md](docs/e2_results.md), E3: [docs/e3_results.md](docs/e3_results.md).

## Тесты

```bash
uv run pytest
```

## План работы

1. Теория: методы и причины расхождений ([docs/reading.md](docs/reading.md)).
2. Данные: синтетические сценарии (независимые признаки, корреляции,
   взаимодействия) + реальные датасеты (California Housing, Bike Sharing).
3. Модели: линейная регрессия, Random Forest, градиентный бустинг, MLP.
4. Метрики согласованности: ранговые корреляции важностей (Спирмен, Кендалл,
   top-k), сравнение формы эффектов (PDP/ALE против SHAP/LIME).
5. Анализ расхождений и выводы.

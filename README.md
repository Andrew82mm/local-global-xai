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
  09_e4_causes.py            E4: флаги, LIME, взаимодействия, PI группой, порог
  10_figures.py              главные графики для текста, только из кеша
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
Сводка: [docs/summary.md](docs/summary.md). Результаты E1: [docs/e1_results.md](docs/e1_results.md), E2: [docs/e2_results.md](docs/e2_results.md), E3: [docs/e3_results.md](docs/e3_results.md), E4: [docs/e4_results.md](docs/e4_results.md).

## Тесты

```bash
uv run pytest
```

## План работы

Задача — регрессия черт личности (`extra`, `emo`) по признакам профиля ВКонтакте,
1000 человек, 34 признака. Модели Ridge, Random Forest, HistGradientBoosting.
Методы PI, SHAP, LIME, PDP, ALE. Метрики: ранговое согласие (Спирмен, Кендалл, top-k),
recall истинных признаков, знаки, CCC кривых, нулевой эталон из перестановок цели.

1. Теория и литература ([docs/reading.md](docs/reading.md)), методика ([docs/methodology.md](docs/methodology.md)). Сделано.
2. Исследование реальных данных ([docs/data_exploration.md](docs/data_exploration.md)). Сделано.
3. E1: синтетика с известной истиной, 5 сценариев × 5 уровней R² ([docs/e1_results.md](docs/e1_results.md)). Сделано.
4. E2: полусинтетика, реальные признаки и заданная истинная функция ([docs/e2_results.md](docs/e2_results.md)). Сделано.
5. E3: реальные черты против нулевого эталона, устойчивость, локальное против глобального ([docs/e3_results.md](docs/e3_results.md)). Сделано.
6. E4: проверка причин расхождений: флаги, LIME, взаимодействия, PI группой, порог ([docs/e4_results.md](docs/e4_results.md)). Сделано.
7. Сводка для текста и графики ([docs/summary.md](docs/summary.md), `notebooks/10_figures.py`). Сделано.
8. Текст курсовой.

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

Запустите JupyterLab из корня проекта и откройте нужный файл `.ipynb` в папке `notebooks/`.

```bash
uv run jupyter lab
```

## Структура

```
notebooks/                 эксперименты, по порядку
  01_first_look.ipynb          все пять методов на California Housing
  02_interaction_toy.ipynb     пример расхождения из-за взаимодействия признаков
  03_data_exploration.ipynb    исследование реальных данных
  04_framework_check.ipynb     проверка общего кода на синтетике, время методов
  05_hgb_interaction.ipynb     разбор: почему PI видит x1·x2 у бустинга, а SHAP нет
  06_e1_synthetic.ipynb        E1: синтетика при разном уровне сигнала
  07_e2_semisynthetic.ipynb    E2: полусинтетика на реальных признаках
  08_e2_report.ipynb           рисунки отчёта E2 и аддитивный контроль
  09_e3_real.ipynb             E3: реальные черты, перестановки и устойчивость
src/xai_consistency/       общий код
  data.py                    кеш реальных данных и 34 признака
  synthetic.py               синтетика с известной истиной
  semisynthetic.py           истинная функция E2 на реальных признаках
  models.py                  Pipeline и модели Ridge, RF, HGB
  explain.py                 PI, PDP, ALE, SHAP, LIME в едином формате, нулевой эталон
  metrics.py                 метрики согласованности
tests/                     тесты генераторов, метрик и методов
data/                      локальные данные (не в git)
results/                   локальные таблицы и графики, выбранные PNG в git
docs/                      литература, исследование данных, методика
```

Решения о настройках методов: [docs/methodology.md](docs/methodology.md).
Результаты E1: [docs/e1_results.md](docs/e1_results.md), E2: [docs/e2_results.md](docs/e2_results.md).
Таблицы и графики E3 строит [ноутбук 09](notebooks/09_e3_real.ipynb).

## Воспроизведение E3

Подготовьте локальный кеш признаков `data/exploration_table.csv.gz` ноутбуком 03.
Ноутбук 09 использует сохранённые агрегаты `results/e3_*.csv`, а недостающие
прогоны рассчитывает. Графики сохраняются в `results/figures/`.

```bash
uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.timeout=-1 notebooks/09_e3_real.ipynb > /dev/null
```

Для проверки с сокращённым числом прогонов добавьте `E3_SMOKE=1` перед командой.
Для полного пересчёта задайте `E3_RECOMPUTE=1`.

## Промежуточные отчёты

- [Отчёт 1 — данные и E1](reports/report1_data_e1.pdf), представлен 13 ноября 2026.
- [Отчёт 2 — полусинтетический эксперимент E2](reports/report2_e2.pdf), представлен 20 ноября 2026.
  Обновлён 23 ноября: уточнены сравнения методов и добавлен аддитивный контроль.

Исходники Typst и рисунки для сборки включены в репозиторий.

```bash
uv run --with typst python reports/build.py
```

Чтобы пересчитать рисунки E2, подготовьте локальный кеш признаков и агрегаты ноутбуков 06 и 07.
Затем запустите ноутбук 08 и сборку отчёта E2.

```bash
uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.timeout=-1 notebooks/08_e2_report.ipynb > /dev/null
uv run --with typst python reports/build.py report2_e2
```

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

"""Сборка промежуточных отчётов в PDF.

Запуск из корня проекта:
    uv run --with typst python reports/build.py

Для пересчёта рисунков E2 нужны локальный кеш признаков и агрегаты ноутбуков 06 и 07:
    uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.timeout=-1 notebooks/08_e2_report.ipynb > /dev/null
    uv run --with typst python reports/build.py report2_e2
"""

import sys
from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NAMES = sys.argv[1:] or ["report1_data_e1", "report2_e2"]

for name in NAMES:
    src = REPORTS / f"{name}.typ"
    if not src.exists():
        print(f"нет {src.name}, пропуск")
        continue
    out = REPORTS / f"{name}.pdf"
    typst.compile(str(src), output=str(out), root=str(ROOT))
    print(f"{out.relative_to(ROOT)}")

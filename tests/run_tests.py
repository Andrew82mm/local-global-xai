"""Запуск тестов без pytest: uv run python tests/run_tests.py

Тесты написаны в стиле pytest (функции test_*, assert), поэтому при установке
pytest их можно запускать и через него: uv run pytest tests
"""

import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

failed = 0
total = 0
for path in sorted(HERE.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        total += 1
        t0 = time.perf_counter()
        try:
            getattr(module, name)()
            print(f"ok    {path.stem}.{name} ({time.perf_counter() - t0:.1f} с)")
        except Exception:
            failed += 1
            print(f"FAIL  {path.stem}.{name}")
            traceback.print_exc()
print(f"\n{total - failed} из {total} тестов прошли")
sys.exit(1 if failed else 0)

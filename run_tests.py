import inspect
import sys
import time
from pathlib import Path

# Add project root to path for local module discovery
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Import test suite module
from tests import test_loan_api


class SimpleMonkeyPatch:
    """Lightweight monkeypatch helper for standalone execution without pytest.

    Allows unit tests to mock functions and restore originals cleanly.
    """

    def __init__(self):
        self._undo_list = []

    def setattr(self, target, name, value):
        old_val = getattr(target, name)
        setattr(target, name, value)
        self._undo_list.append((target, name, old_val))

    def undo(self):
        for target, name, old_val in reversed(self._undo_list):
            setattr(target, name, old_val)
        self._undo_list.clear()


def main():
    """Discover and execute all test functions in tests/test_loan_api.py."""
    print("=" * 70)
    print(" >>> RUNNING LOAN APPROVAL AI AUTOMATED TEST SUITE <<<")
    print("=" * 70)

    # Discover all test functions defined in test_loan_api
    test_functions = [
        (name, func)
        for name, func in inspect.getmembers(test_loan_api, inspect.isfunction)
        if name.startswith("test_")
    ]

    passed = 0
    failed = 0
    start_time = time.time()

    for name, func in test_functions:
        sig = inspect.signature(func)
        mp = SimpleMonkeyPatch()
        try:
            # Provide monkeypatch fixture if requested by test signature
            if "monkeypatch" in sig.parameters:
                func(monkeypatch=mp)
            else:
                func()
            print(f"  [PASSED] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAILED] {name}: Assertion failed -> {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR]  {name}: Unexpected error -> {e}")
            failed += 1
        finally:
            mp.undo()

    duration = round(time.time() - start_time, 3)
    print("=" * 70)
    print(f" RESULTS: {passed} passed, {failed} failed in {duration}s")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Run every module's self-test. Usage: python run_selftests.py   (needs only NumPy; the C++ order book is compiled and checked too when clang++ or g++ is installed)."""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = ["factor_test", "strategy", "risk_model", "optimizer", "portfolio_demo", "ml_data", "gbdt", "validation", "ml_demo", "execution", "options", "probability", "variance", "orderbook", "microstructure", "meta_labeling", "svg"]


def main():
    failed = []
    for name in MODULES:
        print(f"\n== quantlab.{name}")
        t0 = time.time()
        ok = importlib.import_module(f"quantlab.{name}").selftest()
        print(f"   ({time.time() - t0:.1f} s)")
        if not ok:
            failed.append(name)
    print("\nAll self-tests passed." if not failed else f"\nFAILED: {', '.join(failed)}")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

"""Small shared helpers."""
import numpy as np


def mm(a, b):
    """Matrix product with floating-point warnings silenced.

    Some Apple-Accelerate builds of NumPy emit spurious divide/overflow/invalid warnings from ``@`` on ordinary data
    (the results are correct). We silence the warnings and instead verify that the result is finite.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        c = a @ b
    if not np.all(np.isfinite(c)):
        raise FloatingPointError("matrix product produced non-finite values")
    return c


class Checker:
    """Minimal self-test helper: ``check(name, condition)`` prints a tick or a cross and counts failures."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def __call__(self, name, cond):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
        print(("  ok    " if cond else "  FAIL  ") + name)

    def summary(self):
        print(f"\n  {self.passed} passed / {self.failed} failed")
        return self.failed == 0

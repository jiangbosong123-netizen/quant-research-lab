"""A price-time-priority limit order book in plain Python: the reference implementation the C++ book in ``cpp/`` is checked against (Part 9).

Rules (the same as ``cpp/orderbook.hpp``):
  - a limit order first trades against the opposite side at prices it accepts, best price first and, at one price, first come first served;
    trades print at the resting (maker) order's price; any remainder rests at the order's own price;
  - a market order trades until the opposite side is empty; any remainder is dropped;
  - cancel by order id; reduce by order id keeps the order's place in the queue (reducing to zero cancels it);
  - a duplicate live id, a non-positive quantity or a non-positive price is rejected.
Prices are integer ticks: no floating point inside the book.

The text protocol below is the one ``cpp/engine_cli.cpp`` speaks, so the two implementations can be diffed line by line on random order flow
(``differential_check``; it compiles the C++ when a compiler is available and is skipped otherwise).
"""
import bisect
import collections
import os
import random
import shutil
import subprocess
import tempfile

from ._util import Checker

CPP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
CXX_FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]


class OrderBook:
    def __init__(self):
        self.levels = {"B": {}, "S": {}}      # price -> deque([order_id, qty])
        self.prices = {"B": [], "S": []}      # sorted ascending; best bid is the last, best ask the first
        self.index = {}                       # order_id -> (side, price)

    def _best(self, side):
        ps = self.prices[side]
        return (ps[-1] if side == "B" else ps[0]) if ps else None

    def _drop_level(self, side, price):
        del self.levels[side][price]
        self.prices[side].pop(bisect.bisect_left(self.prices[side], price))

    def _take(self, opposite, taker, qty, accepts):
        trades = []
        while qty > 0 and self.prices[opposite]:
            price = self._best(opposite)
            if not accepts(price):
                break
            queue = self.levels[opposite][price]
            while qty > 0 and queue:
                maker = queue[0]
                q = min(maker[1], qty)
                trades.append((taker, maker[0], price, q))
                maker[1] -= q
                qty -= q
                if maker[1] == 0:
                    del self.index[maker[0]]
                    queue.popleft()
            if not queue:
                self._drop_level(opposite, price)
        return trades, qty

    def limit(self, oid, side, price, qty):
        if qty <= 0 or price <= 0 or oid in self.index:
            return []
        opposite = "S" if side == "B" else "B"
        trades, rest = self._take(opposite, oid, qty, (lambda p: p <= price) if side == "B" else (lambda p: p >= price))
        if rest > 0:
            if price not in self.levels[side]:
                self.levels[side][price] = collections.deque()
                bisect.insort(self.prices[side], price)
            self.levels[side][price].append([oid, rest])
            self.index[oid] = (side, price)
        return trades

    def market(self, oid, side, qty):
        if qty <= 0:
            return []
        return self._take("S" if side == "B" else "B", oid, qty, lambda p: True)[0]

    def reduce(self, oid, qty):
        if oid not in self.index or qty <= 0:
            return False
        side, price = self.index[oid]
        queue = self.levels[side][price]
        for i, order in enumerate(queue):
            if order[0] == oid:
                if qty >= order[1]:
                    del queue[i]
                    del self.index[oid]
                else:
                    order[1] -= qty
                break
        if not queue:
            self._drop_level(side, price)
        return True

    def cancel(self, oid):
        return self.reduce(oid, float("inf"))

    def depth(self, side, n):
        ps = self.prices[side][::-1] if side == "B" else self.prices[side]
        return [(p, sum(o[1] for o in self.levels[side][p])) for p in ps[:n]]


def run_commands(lines):
    """Text protocol (one command per line) -> output lines, exactly as cpp/engine_cli.cpp prints them.
    L id side px qty | M id side qty | C id | R id qty  ->  'T taker maker px qty' per trade, then 'S bids=px:qty,... asks=...' (top 5) and ok/no for C and R."""
    book = OrderBook()
    out = []
    for line in lines:
        x = line.split()
        flag = None
        if x[0] == "L":
            t = book.limit(int(x[1]), x[2], int(x[3]), int(x[4]))
        elif x[0] == "M":
            t = book.market(int(x[1]), x[2], int(x[3]))
        elif x[0] == "C":
            t, flag = [], ("ok" if book.cancel(int(x[1])) else "no")
        else:
            t, flag = [], ("ok" if book.reduce(int(x[1]), int(x[2])) else "no")
        out += [f"T {a} {m} {p} {q}" for a, m, p, q in t]
        snap = "S bids=" + ",".join(f"{p}:{q}" for p, q in book.depth("B", 5)) + " asks=" + ",".join(f"{p}:{q}" for p, q in book.depth("S", 5))
        out.append(snap + (f" {flag}" if flag else ""))
    return out


def random_commands(n, seed):
    """Random order flow: 55 % limit orders within 20 ticks of a slowly drifting mid, 10 % market orders,
    30 % cancels and 5 % reductions of one of the last 200 ids (which may already be gone)."""
    g = random.Random(seed)
    mid, oid, out = 10000, 1, []
    for _ in range(n):
        if g.random() < 0.01:
            mid += g.choice((-1, 1))
        x, side, qty = g.random(), g.choice("BS"), 1 + int(g.random() * 500)
        if x < 0.55:
            off = int(g.random() * 20)
            out.append(f"L {oid} {side} {mid - off if side == 'B' else mid + off} {qty}")
            oid += 1
        elif x < 0.65:
            out.append(f"M {oid} {side} {qty}")
            oid += 1
        elif x < 0.95:
            out.append(f"C {max(1, oid - 1 - int(g.random() * 200))}")
        else:
            out.append(f"R {max(1, oid - 1 - int(g.random() * 200))} {qty // 4 + 1}")
    return out


def find_compiler():
    return shutil.which("clang++") or shutil.which("g++")


def build(target, out_dir, cxx=None):
    """Compile cpp/<target>.cpp with -Wall -Wextra -Wpedantic -Werror; returns the binary's path."""
    cxx = cxx or find_compiler()
    exe = os.path.join(out_dir, target)
    r = subprocess.run([cxx, *CXX_FLAGS, "-o", exe, os.path.join(CPP_DIR, f"{target}.cpp")], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"compiling {target}.cpp failed:\n{r.stderr[-2000:]}")
    return exe


def differential_check(n=20000, seeds=(1, 2, 3), out_dir=None):
    """Feed the same random commands to the C++ engine and to this reference; every trade and every top-5 snapshot must match exactly.
    Returns (number of commands compared, all equal) or None when no C++ compiler is installed."""
    if not find_compiler():
        return None
    with tempfile.TemporaryDirectory() as td:
        exe = build("engine_cli", out_dir or td)
        total, same = 0, True
        for s in seeds:
            cmds = random_commands(n, s)
            r = subprocess.run([exe], input="\n".join(cmds) + "\n", capture_output=True, text=True)
            same &= r.returncode == 0 and r.stdout.splitlines() == run_commands(cmds)
            total += len(cmds)
        return total, same


def selftest():
    check = Checker()
    b = OrderBook()
    b.limit(1, "S", 101, 50)
    b.limit(2, "S", 101, 30)
    b.limit(3, "S", 102, 40)
    t = b.limit(4, "B", 102, 100)
    check("time priority at one price, best price first, trades at the maker's price, the remainder rests at its own price",
          t == [(4, 1, 101, 50), (4, 2, 101, 30), (4, 3, 102, 20)] and b.depth("S", 5) == [(102, 20)] and b.depth("B", 5) == [])
    b.limit(5, "B", 100, 10)
    b.limit(6, "B", 100, 20)
    b.reduce(5, 4)
    t = b.limit(7, "S", 100, 8)
    check("reduce keeps queue priority (order 5 is still first after being reduced to 6)", t == [(7, 5, 100, 6), (7, 6, 100, 2)] and b.depth("B", 5) == [(100, 18)])
    t = b.market(8, "B", 1000)
    check("a market order sweeps the opposite side and the unfilled remainder does not rest", t == [(8, 3, 102, 20)] and b.depth("S", 5) == [] and 8 not in b.index)
    check("cancel removes the order and its empty level; cancelling twice reports failure", b.cancel(6) and not b.cancel(6) and b.depth("B", 5) == [])
    b.limit(9, "B", 99, 5)
    check("duplicate live ids and non-positive quantities or prices are rejected",
          b.limit(9, "B", 98, 5) == [] and b.depth("B", 5) == [(99, 5)] and b.limit(10, "B", 0, 5) == [] and b.limit(11, "S", 120, 0) == [] and len(b.index) == 1)
    out = run_commands(["L 1 B 100 5", "L 2 S 101 7", "M 3 B 3", "R 2 10", "C 1"])
    check("text protocol: snapshots, trades and ok/no flags in engine_cli's format",
          out == ["S bids=100:5 asks=", "S bids=100:5 asks=101:7", "T 3 2 101 3", "S bids=100:5 asks=101:4", "S bids=100:5 asks= ok", "S bids= asks= ok"])
    book, crossed = OrderBook(), False
    for line in random_commands(5000, 7):
        x = line.split()
        if x[0] == "L":
            book.limit(int(x[1]), x[2], int(x[3]), int(x[4]))
        elif x[0] == "M":
            book.market(int(x[1]), x[2], int(x[3]))
        elif x[0] == "C":
            book.cancel(int(x[1]))
        else:
            book.reduce(int(x[1]), int(x[2]))
        bb, ba = book._best("B"), book._best("S")
        crossed |= bb is not None and ba is not None and bb >= ba
    check("5,000 random commands: the book never crosses (best bid < best ask after every step) and the id index matches the queues",
          not crossed and sorted(book.index) == sorted(o[0] for s in "BS" for q in book.levels[s].values() for o in q))
    res = differential_check(n=5000, seeds=(1,))
    if res is None:
        check("C++ differential check skipped: no clang++ or g++ on PATH", True)
    else:
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([build("tests", td)], capture_output=True, text=True)
        check(f"C++ unit tests (cpp/tests.cpp) pass: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}", r.returncode == 0)
        check(f"C++ engine and this reference agree line by line on {res[0]:,} random commands (trades and top-5 snapshots)", res[1])
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)

// Throughput and per-operation latency of orderbook.hpp on a synthetic order flow.
// Flow (fixed seed): 55% limit orders within +-20 ticks of a drifting mid, 10% market orders, 30% cancels of a random
// live-looking id, 5% partial reduces. Usage: bench [n=1000000] [seed=7]
// Prints one line: ops, seconds, ops_per_sec, and p50 / p99 / p99.9 / max latency in nanoseconds (timed run), plus the
// final number of resting orders so the two runs can be checked to be identical.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "orderbook.hpp"

using namespace lab;

struct Op {
    char kind;
    uint64_t id;
    Side side;
    int64_t px, qty;
};

static std::vector<Op> make_flow(size_t n, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Op> ops;
    ops.reserve(n);
    int64_t mid = 10000;
    uint64_t next = 1;
    for (size_t i = 0; i < n; ++i) {
        if (u(g) < 0.01) mid += (u(g) < 0.5 ? -1 : 1);
        const double x = u(g);
        const Side s = u(g) < 0.5 ? Side::Buy : Side::Sell;
        const int64_t qty = 1 + static_cast<int64_t>(u(g) * 500);
        if (x < 0.55) {
            const int64_t off = static_cast<int64_t>(u(g) * 20);
            const int64_t px = s == Side::Buy ? mid - off : mid + off;
            ops.push_back({'L', next++, s, px, qty});
        } else if (x < 0.65) {
            ops.push_back({'M', next++, s, 0, qty});
        } else if (x < 0.95) {
            const uint64_t back = 1 + static_cast<uint64_t>(u(g) * 200);
            ops.push_back({'C', next > back ? next - back : 1, s, 0, 0});
        } else {
            const uint64_t back = 1 + static_cast<uint64_t>(u(g) * 200);
            ops.push_back({'R', next > back ? next - back : 1, s, 0, qty / 4 + 1});
        }
    }
    return ops;
}

static size_t run(const std::vector<Op>& ops, std::vector<int64_t>* lat) {
    OrderBook b;
    size_t trades = 0;
    for (const Op& o : ops) {
        const auto t0 = lat ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        switch (o.kind) {
            case 'L': trades += b.limit(o.id, o.side, o.px, o.qty).size(); break;
            case 'M': trades += b.market(o.id, o.side, o.qty).size(); break;
            case 'C': b.cancel(o.id); break;
            default: b.reduce(o.id, o.qty); break;
        }
        if (lat) lat->push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    return trades * 1000003 + b.orders();
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
    const uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 7;
    const auto ops = make_flow(n, seed);
    const auto t0 = std::chrono::steady_clock::now();
    const size_t check1 = run(ops, nullptr);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<int64_t> lat;
    lat.reserve(n);
    const size_t check2 = run(ops, &lat);
    std::sort(lat.begin(), lat.end());
    auto q = [&lat](double p) { return lat[std::min(lat.size() - 1, static_cast<size_t>(p * (lat.size() - 1)))]; };
    std::printf("ops=%zu seconds=%.4f ops_per_sec=%.0f p50_ns=%lld p99_ns=%lld p999_ns=%lld max_ns=%lld same_result=%d check=%zu\n", n, secs,
                n / secs, (long long)q(0.5), (long long)q(0.99), (long long)q(0.999), (long long)lat.back(), check1 == check2 ? 1 : 0, check1);
    return check1 == check2 ? 0 : 1;
}

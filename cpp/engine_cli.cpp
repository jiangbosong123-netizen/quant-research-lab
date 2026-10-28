// Text protocol driver for cross-checking the C++ book against the Python reference (quantlab/orderbook.py).
// stdin, one command per line:
//   L id side px qty   limit order   (side: B or S)
//   M id side qty      market order
//   C id               cancel
//   R id qty           reduce
// stdout, per command: one line per trade "T taker maker px qty", then one snapshot line
//   "S bids=px:qty,px:qty,... asks=px:qty,..." (top 5 each side) and the result flag for C / R ("ok" / "no").
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "orderbook.hpp"

using lab::LevelView;
using lab::OrderBook;
using lab::Side;
using lab::Trade;

static void snap(const OrderBook& b, const char* flag) {
    std::string s = "S bids=";
    auto put = [&s](const std::vector<LevelView>& v) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ',';
            s += std::to_string(v[i].price) + ':' + std::to_string(v[i].qty);
        }
    };
    put(b.depth(Side::Buy, 5));
    s += " asks=";
    put(b.depth(Side::Sell, 5));
    if (flag) {
        s += ' ';
        s += flag;
    }
    std::puts(s.c_str());
}

int main() {
    std::ios::sync_with_stdio(false);
    OrderBook b;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream in(line);
        char op = 0;
        in >> op;
        std::vector<Trade> t;
        const char* flag = nullptr;
        if (op == 'L' || op == 'M') {
            unsigned long long id = 0;
            char side = 0;
            long long px = 0, qty = 0;
            if (op == 'L') in >> id >> side >> px >> qty;
            else in >> id >> side >> qty;
            const Side s = side == 'B' ? Side::Buy : Side::Sell;
            t = op == 'L' ? b.limit(id, s, px, qty) : b.market(id, s, qty);
        } else if (op == 'C') {
            unsigned long long id = 0;
            in >> id;
            flag = b.cancel(id) ? "ok" : "no";
        } else if (op == 'R') {
            unsigned long long id = 0;
            long long qty = 0;
            in >> id >> qty;
            flag = b.reduce(id, qty) ? "ok" : "no";
        } else {
            std::fprintf(stderr, "unknown command: %s\n", line.c_str());
            return 2;
        }
        for (const Trade& x : t)
            std::printf("T %llu %llu %lld %lld\n", (unsigned long long)x.taker, (unsigned long long)x.maker, (long long)x.price,
                        (long long)x.qty);
        snap(b, flag);
        if (b.crossed()) {
            std::fprintf(stderr, "book crossed after: %s\n", line.c_str());
            return 3;
        }
    }
    return 0;
}

// Unit tests for orderbook.hpp: hand-worked cases with known answers. Exit code 0 = all pass.
#include <cstdio>
#include <vector>

#include "orderbook.hpp"

using namespace lab;

static int failures = 0, passes = 0;
#define CHECK(cond, name)                                    \
    do {                                                     \
        if (cond) {                                          \
            ++passes;                                        \
        } else {                                             \
            ++failures;                                      \
            std::printf("FAIL %s (line %d)\n", name, __LINE__); \
        }                                                    \
    } while (0)

static bool same(const std::vector<Trade>& t, const std::vector<Trade>& want) {
    if (t.size() != want.size()) return false;
    for (size_t i = 0; i < t.size(); ++i)
        if (t[i].taker != want[i].taker || t[i].maker != want[i].maker || t[i].price != want[i].price || t[i].qty != want[i].qty)
            return false;
    return true;
}

int main() {
    {  // time priority at one price: the older order fills first
        OrderBook b;
        b.limit(1, Side::Sell, 100, 5);
        b.limit(2, Side::Sell, 100, 5);
        auto t = b.limit(3, Side::Buy, 100, 7);
        CHECK(same(t, {{3, 1, 100, 5}, {3, 2, 100, 2}}), "FIFO within a level");
        auto a = b.depth(Side::Sell, 5);
        CHECK(a.size() == 1 && a[0].price == 100 && a[0].qty == 3, "remaining ask size");
        CHECK(b.depth(Side::Buy, 5).empty(), "fully filled buy does not rest");
    }
    {  // price priority and trade at the maker's price
        OrderBook b;
        b.limit(1, Side::Sell, 102, 5);
        b.limit(2, Side::Sell, 101, 5);
        auto t = b.limit(3, Side::Buy, 103, 8);
        CHECK(same(t, {{3, 2, 101, 5}, {3, 1, 102, 3}}), "best price first, prints at maker price");
    }
    {  // limit that does not cross rests; remainder of a crossing limit rests at its own price
        OrderBook b;
        b.limit(1, Side::Sell, 101, 4);
        auto t = b.limit(2, Side::Buy, 101, 10);
        CHECK(same(t, {{2, 1, 101, 4}}), "partial fill");
        auto bids = b.depth(Side::Buy, 5);
        CHECK(bids.size() == 1 && bids[0].price == 101 && bids[0].qty == 6, "remainder rests at limit price");
        CHECK(!b.crossed(), "book not crossed");
    }
    {  // market order sweeps levels and drops what it cannot fill
        OrderBook b;
        b.limit(1, Side::Buy, 99, 3);
        b.limit(2, Side::Buy, 98, 3);
        auto t = b.market(3, Side::Sell, 10);
        CHECK(same(t, {{3, 1, 99, 3}, {3, 2, 98, 3}}), "market sweeps");
        CHECK(b.depth(Side::Buy, 5).empty() && b.depth(Side::Sell, 5).empty(), "market remainder does not rest");
    }
    {  // cancel and reduce keep the queue consistent; reduce keeps time priority
        OrderBook b;
        b.limit(1, Side::Buy, 100, 5);
        b.limit(2, Side::Buy, 100, 5);
        CHECK(b.reduce(1, 3), "reduce ok");
        auto bids = b.depth(Side::Buy, 5);
        CHECK(bids.size() == 1 && bids[0].qty == 7, "level total after reduce");
        auto t = b.market(9, Side::Sell, 3);
        CHECK(same(t, {{9, 1, 100, 2}, {9, 2, 100, 1}}), "reduced order keeps its place in the queue");
        CHECK(b.cancel(2), "cancel ok");
        CHECK(!b.cancel(2), "cancel twice fails");
        CHECK(b.depth(Side::Buy, 5).empty() && b.orders() == 0, "empty level is removed");
        CHECK(!b.reduce(42, 1), "reduce unknown id fails");
    }
    {  // reduce to zero removes; duplicate id rejected; bad inputs rejected
        OrderBook b;
        b.limit(1, Side::Sell, 100, 5);
        CHECK(b.reduce(1, 5) && b.orders() == 0 && b.depth(Side::Sell, 1).empty(), "reduce to zero removes");
        b.limit(2, Side::Sell, 100, 5);
        auto t = b.limit(2, Side::Sell, 101, 5);
        CHECK(t.empty() && b.orders() == 1 && b.depth(Side::Sell, 5).size() == 1, "duplicate live id rejected");
        CHECK(b.limit(3, Side::Buy, 100, 0).empty() && b.limit(4, Side::Buy, 0, 5).empty() && b.orders() == 1, "zero size / price rejected");
    }
    {  // depth ordering: bids high->low, asks low->high
        OrderBook b;
        b.limit(1, Side::Buy, 97, 1);
        b.limit(2, Side::Buy, 99, 1);
        b.limit(3, Side::Buy, 98, 1);
        b.limit(4, Side::Sell, 103, 1);
        b.limit(5, Side::Sell, 101, 1);
        auto bids = b.depth(Side::Buy, 2);
        auto asks = b.depth(Side::Sell, 5);
        CHECK(bids.size() == 2 && bids[0].price == 99 && bids[1].price == 98, "bid order and depth limit");
        CHECK(asks.size() == 2 && asks[0].price == 101 && asks[1].price == 103, "ask order");
    }
    std::printf("tests: %d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}

// Limit order book with price-time priority (C++17, header-only).
// Same semantics as the Python reference in quantlab/orderbook.py; cross-checked trade by trade (Part 9).
//
// Design (the usual interview answer, and why):
//   - one std::map per side, keyed by price in integer ticks (no floating point in the book);
//     bids sorted high->low, asks low->high, so the best price is always begin()   -> O(log P) per new level
//   - each price level keeps a FIFO queue of resting orders (std::list: stable iterators, O(1) erase from the middle)
//   - an unordered_map from order id to (side, price, list iterator) gives O(1) cancel / reduce by id
//   - trades always print at the resting (maker) order's price
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lab {

enum class Side : int8_t { Buy = 1, Sell = -1 };

struct Trade {
    uint64_t taker;
    uint64_t maker;
    int64_t price;
    int64_t qty;
};

struct LevelView {
    int64_t price;
    int64_t qty;
};

class OrderBook {
    struct Node {
        uint64_t id;
        int64_t qty;
    };
    struct Level {
        std::list<Node> queue;
        int64_t total = 0;
    };
    using Bids = std::map<int64_t, Level, std::greater<int64_t>>;
    using Asks = std::map<int64_t, Level, std::less<int64_t>>;
    struct Ref {
        Side side;
        int64_t price;
        std::list<Node>::iterator it;
    };

    Bids bids_;
    Asks asks_;
    std::unordered_map<uint64_t, Ref> index_;

    // Walk the opposite side from the best price while it crosses; fills oldest first within a level.
    template <class Opp, class Crosses>
    void sweep(Opp& opp, uint64_t taker, int64_t& qty, Crosses crosses, std::vector<Trade>& out) {
        while (qty > 0 && !opp.empty() && crosses(opp.begin()->first)) {
            auto lvl = opp.begin();
            auto& q = lvl->second.queue;
            while (qty > 0 && !q.empty()) {
                Node& maker = q.front();
                const int64_t fill = maker.qty < qty ? maker.qty : qty;
                out.push_back(Trade{taker, maker.id, lvl->first, fill});
                maker.qty -= fill;
                lvl->second.total -= fill;
                qty -= fill;
                if (maker.qty == 0) {
                    index_.erase(maker.id);
                    q.pop_front();
                }
            }
            if (q.empty()) opp.erase(lvl);
        }
    }

    template <class Book>
    void rest(Book& book, Side s, uint64_t id, int64_t px, int64_t qty) {
        Level& lvl = book[px];
        lvl.queue.push_back(Node{id, qty});
        lvl.total += qty;
        index_.emplace(id, Ref{s, px, std::prev(lvl.queue.end())});
    }

    template <class Book>
    bool shrink(Book& book, const Ref& r, int64_t qty, uint64_t id) {
        auto lvl = book.find(r.price);
        if (lvl == book.end()) throw std::logic_error("index points at a missing level");
        auto it = r.it;
        if (qty >= it->qty) {
            lvl->second.total -= it->qty;
            lvl->second.queue.erase(it);
            index_.erase(id);
        } else {
            it->qty -= qty;
            lvl->second.total -= qty;
        }
        if (lvl->second.queue.empty()) book.erase(lvl);
        return true;
    }

public:
    // Limit order: match against the other side while it crosses, rest the remainder at px.
    // Returns the trades. A duplicate live id or a non-positive size/price is rejected (no trades, nothing rests).
    std::vector<Trade> limit(uint64_t id, Side s, int64_t px, int64_t qty) {
        std::vector<Trade> out;
        if (qty <= 0 || px <= 0 || index_.count(id)) return out;
        if (s == Side::Buy) {
            sweep(asks_, id, qty, [px](int64_t ask) { return ask <= px; }, out);
            if (qty > 0) rest(bids_, s, id, px, qty);
        } else {
            sweep(bids_, id, qty, [px](int64_t bid) { return bid >= px; }, out);
            if (qty > 0) rest(asks_, s, id, px, qty);
        }
        return out;
    }

    // Market order: sweep until filled or the other side is empty; any remainder is dropped (immediate-or-cancel).
    std::vector<Trade> market(uint64_t id, Side s, int64_t qty) {
        std::vector<Trade> out;
        if (qty <= 0) return out;
        auto any = [](int64_t) { return true; };
        if (s == Side::Buy) sweep(asks_, id, qty, any, out);
        else sweep(bids_, id, qty, any, out);
        return out;
    }

    // Full cancel by id. False when the id is not resting.
    bool cancel(uint64_t id) {
        auto f = index_.find(id);
        if (f == index_.end()) return false;
        const Ref r = f->second;
        return r.side == Side::Buy ? shrink(bids_, r, INT64_MAX, id) : shrink(asks_, r, INT64_MAX, id);
    }

    // Partial cancel or execution of a resting order by id (keeps its time priority); removes it when the size hits 0.
    bool reduce(uint64_t id, int64_t qty) {
        auto f = index_.find(id);
        if (f == index_.end() || qty <= 0) return false;
        const Ref r = f->second;
        return r.side == Side::Buy ? shrink(bids_, r, qty, id) : shrink(asks_, r, qty, id);
    }

    std::vector<LevelView> depth(Side s, size_t n) const {
        std::vector<LevelView> v;
        if (s == Side::Buy) {
            for (auto it = bids_.begin(); it != bids_.end() && v.size() < n; ++it) v.push_back({it->first, it->second.total});
        } else {
            for (auto it = asks_.begin(); it != asks_.end() && v.size() < n; ++it) v.push_back({it->first, it->second.total});
        }
        return v;
    }

    size_t orders() const { return index_.size(); }
    bool crossed() const { return !bids_.empty() && !asks_.empty() && bids_.begin()->first >= asks_.begin()->first; }
};

}  // namespace lab

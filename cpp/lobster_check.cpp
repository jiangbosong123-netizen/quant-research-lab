// Check the order-book update rules on real NASDAQ data (LOBSTER sample, 2012-06-21, level 10).
// LOBSTER gives a message file (one event per line) and an order-book file (the top-10 book after each event).
// Its message file only contains events inside the requested 10-level window, so the book cannot be rebuilt from
// messages alone; instead every event is checked on its own:
//   take the official book after event k-1, apply event k with our rules, compare with the official book after event k.
// Rules: 1 new limit order -> add size at price; 2 partial cancel, 3 delete, 4 visible execution -> remove size at price;
//        5 hidden execution, 6 cross trade, 7 halt -> the visible book does not change.
// When our update empties a level, the official book pulls a new level into the window that we cannot know about,
// so only the levels we can know are compared (all of them otherwise).
// Usage: lobster_check <message.csv> <orderbook.csv> [levels=10] [max_rows=0 (all)]
// Prints one line: key=value pairs (rows, checked, exact, by-type counts, first mismatches, seconds).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

const long long kDummy = 9999999999LL;

bool read_line(FILE* f, std::string& s) {
    s.clear();
    int c;
    while ((c = std::fgetc(f)) != EOF) {
        if (c == '\n') return true;
        if (c != '\r') s.push_back(static_cast<char>(c));
    }
    return !s.empty();
}

// Split on commas into integers (the message time column is parsed separately and ignored here).
void parse_ints(const std::string& s, std::vector<long long>& out) {
    out.clear();
    const char* p = s.c_str();
    while (*p) {
        char* end = nullptr;
        out.push_back(std::strtoll(p, &end, 10));
        p = end;
        while (*p && *p != ',') ++p;  // skip a fractional part if any
        if (*p == ',') ++p;
    }
}

struct Book {
    std::vector<std::pair<long long, long long>> bid, ask;  // (price, size), best first
};

Book from_row(const std::vector<long long>& r, int levels) {
    Book b;
    for (int i = 0; i < levels; ++i) {
        const long long ap = r[4 * i], as = r[4 * i + 1], bp = r[4 * i + 2], bs = r[4 * i + 3];
        if (as > 0 && ap != kDummy) b.ask.emplace_back(ap, as);
        if (bs > 0 && bp != -kDummy) b.bid.emplace_back(bp, bs);
    }
    return b;
}

// Returns false when a remove hits a price that is not in the window.
bool apply(Book& b, int type, long long size, long long price, int dir) {
    if (type == 5 || type == 6 || type == 7) return true;
    auto& side = dir == 1 ? b.bid : b.ask;
    auto it = std::find_if(side.begin(), side.end(), [price](const std::pair<long long, long long>& x) { return x.first == price; });
    if (type == 1) {
        if (it != side.end()) it->second += size;
        else side.emplace_back(price, size);
    } else if (type == 2 || type == 3 || type == 4) {
        if (it == side.end()) return false;
        it->second -= size;
        if (it->second <= 0) side.erase(it);
    } else {
        return false;
    }
    if (dir == 1) std::sort(side.begin(), side.end(), [](auto& x, auto& y) { return x.first > y.first; });
    else std::sort(side.begin(), side.end(), [](auto& x, auto& y) { return x.first < y.first; });
    return true;
}

// Compare the levels we can know: all `levels` if we still have that many, otherwise the ones we have.
bool same_side(const std::vector<std::pair<long long, long long>>& ours, const std::vector<std::pair<long long, long long>>& off,
               int levels, bool shrank) {
    const size_t n = shrank ? std::min(ours.size(), off.size()) : static_cast<size_t>(levels);
    const size_t m = std::min(n, static_cast<size_t>(levels));
    for (size_t i = 0; i < m; ++i) {
        const bool a = i < ours.size(), o = i < off.size();
        if (a != o) return false;
        if (a && ours[i] != off[i]) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: lobster_check <message.csv> <orderbook.csv> [levels]\n");
        return 2;
    }
    const int levels = argc > 3 ? std::atoi(argv[3]) : 10;
    const long long max_rows = argc > 4 ? std::atoll(argv[4]) : 0;
    FILE* fm = std::fopen(argv[1], "rb");
    FILE* fo = std::fopen(argv[2], "rb");
    if (!fm || !fo) {
        std::fprintf(stderr, "cannot open input\n");
        return 2;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::string ml, ol;
    std::vector<long long> m, r, prev;
    long long rows = 0, checked = 0, exact = 0, missing = 0;
    long long by_type[8] = {0}, bad_type[8] = {0};
    std::vector<long long> first_bad;
    while ((max_rows == 0 || rows < max_rows) && read_line(fm, ml)) {
        if (!read_line(fo, ol)) {
            std::fprintf(stderr, "order-book file is shorter than the message file\n");
            return 2;
        }
        ++rows;
        // message: time(seconds, decimal), type, id, size, price, direction
        const char* comma = std::strchr(ml.c_str(), ',');
        parse_ints(comma ? std::string(comma + 1) : std::string(), m);
        parse_ints(ol, r);
        if (static_cast<int>(r.size()) != 4 * levels || m.size() != 5) {
            std::fprintf(stderr, "bad line %lld\n", rows);
            return 2;
        }
        if (!prev.empty()) {
            const int type = static_cast<int>(m[0]);
            Book ours = from_row(prev, levels);
            const size_t nb = ours.bid.size(), na = ours.ask.size();
            const bool ok_apply = apply(ours, type, m[2], m[3], static_cast<int>(m[4]));
            const Book off = from_row(r, levels);
            ++checked;
            if (type >= 1 && type <= 7) ++by_type[type];
            bool ok = ok_apply && same_side(ours.bid, off.bid, levels, ours.bid.size() < nb) &&
                      same_side(ours.ask, off.ask, levels, ours.ask.size() < na);
            if (!ok_apply) ++missing;
            if (ok) ++exact;
            else {
                if (type >= 1 && type <= 7) ++bad_type[type];
                if (first_bad.size() < 5) first_bad.push_back(rows);
            }
        }
        prev.swap(r);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("rows=%lld checked=%lld exact=%lld missing_level=%lld", rows, checked, exact, missing);
    for (int k = 1; k <= 7; ++k) std::printf(" type%d=%lld/%lld", k, by_type[k] - bad_type[k], by_type[k]);
    std::printf(" first_bad=");
    for (size_t i = 0; i < first_bad.size(); ++i) std::printf(i ? ",%lld" : "%lld", first_bad[i]);
    std::printf(" seconds=%.3f\n", secs);
    std::fclose(fm);
    std::fclose(fo);
    return 0;
}

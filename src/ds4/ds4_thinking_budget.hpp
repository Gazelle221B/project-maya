// Request-local reasoning limit, shared with CPU-only protocol/decode regression tests.
#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>

namespace strata::ds4 {

struct ThinkBudget {
    long long budget = 0;
    int end = -1;
    bool active = false;

    bool parse(const std::string& key, const std::string& value) {
        if (key != "think_budget" && key != "think_end") return false;
        char* tail = nullptr;
        errno = 0;
        const long long n = std::strtoll(value.c_str(), &tail, 10);
        const bool valid = tail != value.c_str() && *tail == '\0' && errno != ERANGE;
        if (key == "think_budget") budget = valid ? n : 0;
        else end = valid && n >= 0 && n <= INT_MAX ? (int) n : -1;
        return true;
    }

    void start(int vocab) { active = budget > 0 && end >= 0 && end < vocab; }
    void on_token(int token) { if (token == end) active = false; }
    bool force(long long produced) const { return active && produced >= budget; }
    int room(int available, long long produced) const {
        return active ? (int) std::min<long long>(available, std::max(0LL, budget - produced)) : available;
    }
};

} // namespace strata::ds4

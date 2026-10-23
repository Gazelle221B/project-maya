// CPU stand-in for DS4's actual parser, speculative round and serve emission loop.
#include "ds4_thinking_budget.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(__func__) + ": " + #x); } while (false)

namespace strata::ds4 {
struct Model {
    std::vector<int> fed{1, 2};
    int accept_limit = 4, natural_end = -1;
    bool fail = false;
    std::vector<int> verified_sizes;
    int next(const std::vector<int>& history) const {
        const int i = (int) history.size() - 2;
        if (i == natural_end) return 99;
        return std::find(history.begin(), history.end(), 99) != history.end() ? 1000 + i : 10 + i;
    }
    bool step(int tok, bool, std::string&) {
        if (fail) return false;
        fed.push_back(tok);
        return true;
    }
    int sample(float, float, int, float, unsigned long long, unsigned long long) const { return next(fed); }
    int draft_block() const { return 4; }
    long long pos() const { return (long long) fed.size(); }
    long long max_context() const { return 4096; }
    bool draft(int tok, std::vector<int>& drafts, std::vector<float>& conf, std::string&,
               float, float, int, float, unsigned long long, unsigned long long) {
        auto history = fed;
        history.push_back(tok);
        for (int i = 0; i < 4; ++i) {
            const int t = next(history);
            drafts.push_back(t);
            conf.push_back(1.0f);
            history.push_back(t);
        }
        return !fail;
    }
    bool verify(int tok, const std::vector<int>& drafts, float, float, int, float,
                unsigned long long, unsigned long long, int& accepted, int& next_tok, std::string& err) {
        verified_sizes.push_back((int) drafts.size());
        if (!step(tok, true, err)) return false;
        accepted = std::min(accept_limit, (int) drafts.size());
        for (int i = 0; i < accepted; ++i) {
            CHECK(drafts[(size_t) i] == next(fed));
            if (!step(drafts[(size_t) i], true, err)) return false;
        }
        next_tok = next(fed);
        return true;
    }
};
} // namespace strata::ds4

#include "ds4_budget_protocol.inc"

struct Result {
    std::vector<int> emitted, fed;
    std::string finish;
    long long produced;
    unsigned long long counter;
};

Result run(Req r, bool spec, int accepted = 4, int natural_end = -1, int eos_token = -1,
           bool cancelled = false, bool fail = false) {
    strata::ds4::Model m;
    m.accept_limit = accepted;
    m.natural_end = natural_end;
    m.fail = fail;
    r.thinking.start(2048);
    struct { std::atomic<bool> stop{false}; } in;
    in.stop.store(cancelled);
    std::vector<int> emitted, got, eos{eos_token};
    unsigned long long counter = 1;
    int tok = m.next(m.fed);
    long long produced = 0, d_off = 0, d_acc = 0;
    const long long limit = r.max_new;
    const float conf_min = 0.0f;
    bool cool = false;
    const char* finish = "length";
    std::string err;
    const auto stat_line = [] {};
    const auto dump_rows = [] {};
#include "ds4_budget_loop.inc"
    CHECK(produced == (long long) emitted.size());
    if (!cancelled && !fail && !emitted.empty()) {
        // A round may already have fed accepted tokens past an EOS. Other cases
        // must leave exactly its last emitted token pending, including think_end.
        if (finish != std::string("stop")) {
            std::vector<int> wanted{1, 2};
            wanted.insert(wanted.end(), emitted.begin(), emitted.end() - 1);
            CHECK(m.fed == wanted);
        }
    }
    return {emitted, m.fed, finish, produced, counter};
}

Req request(long long budget, long long limit = 16, int end = 99) {
    Req r;
    r.max_new = limit;
    r.thinking.budget = budget;
    r.thinking.end = end;
    return r;
}

void parse_tests() {
    Req r;
    CHECK(parse_gen("GEN 20 temperature=0 think_budget=512 spec=1 think_end=99 1,2,3", r, false));
    CHECK(r.max_new == 20 && r.thinking.budget == 512 && r.thinking.end == 99);
    CHECK(r.spec == 1 && r.ids == std::vector<int>({1, 2, 3}));
    r.thinking.start(100);
    CHECK(r.thinking.active);
    Req next;
    CHECK(parse_gen("GEN 4 7,8", next, false));
    next.thinking.start(100);
    CHECK(next.thinking.budget == 0 && next.thinking.end == -1 && !next.thinking.active);
    Req image;
    CHECK(parse_gen("GENI 9 think_end=99 dspark_conf=0.8 think_budget=1 picture.bin 5,6", image, true));
    CHECK(image.file == "picture.bin" && image.ids == std::vector<int>({5, 6}));
    CHECK(image.thinking.budget == 1 && image.thinking.end == 99);
    Req duplicate;
    CHECK(parse_gen("GEN 5 think_budget=8 think_budget=2 think_end=99 1,2", duplicate, false));
    CHECK(duplicate.thinking.budget == 2);
}

void edge_tests() {
    using strata::ds4::ThinkBudget;
    for (const std::string value : {"", "junk", "1junk", "9223372036854775808", "-9223372036854775809"}) {
        ThinkBudget b;
        CHECK(b.parse("think_budget", value) && b.budget == 0);
        CHECK(b.parse("think_end", value) && b.end == -1);
    }
    for (const std::string value : {"-1", "2147483648", "999999999999999999999"}) {
        ThinkBudget b;
        CHECK(b.parse("think_end", value) && b.end == -1);
    }
    for (long long budget : {-5LL, 0LL}) {
        auto b = request(budget).thinking;
        b.start(100);
        CHECK(!b.active && !b.force(1000) && b.room(10, 1000) == 10);
    }
    for (int end : {-1, 100, INT_MAX}) {
        auto b = request(1, 3, end).thinking;
        b.start(100);
        CHECK(!b.active && !b.force(1));
    }
    ThinkBudget b;
    CHECK(!b.parse("top_k", "1"));
    CHECK(b.parse("think_budget", "9223372036854775807") && b.budget == LLONG_MAX);
    CHECK(b.parse("think_end", "0"));
    b.start(100);
    CHECK(b.active && b.room(INT_MAX, 1) == INT_MAX && !b.force(1));
    CHECK(b.force(LLONG_MAX) && b.room(10, LLONG_MAX) == 0);
    b.on_token(0);
    CHECK(!b.active && !b.force(LLONG_MAX) && b.room(10, LLONG_MAX) == 10);
    b.start(100);
    CHECK(b.active); // explicit reset/start can reuse a config
}

void force_tests() {
    for (bool spec : {false, true}) for (int accepted : {0, 1, 4}) {
        for (long long budget : {1LL, 2LL, 4LL, 5LL, 8LL, 9LL}) {
            const auto out = run(request(budget), spec, accepted);
            CHECK(out.emitted.size() == 16);
            CHECK(out.emitted[(size_t) budget] == 99);
            CHECK(std::count(out.emitted.begin(), out.emitted.end(), 99) == 1);
            CHECK(out.emitted[(size_t) budget + 1] >= 1000);
            for (int i = 0; i < budget; ++i) CHECK(out.emitted[(size_t) i] == 10 + i);
            CHECK(out.finish == "length");
            CHECK(out.counter == 16);
        }
    }
}

void natural_tests() {
    for (bool spec : {false, true}) for (int accepted : {0, 1, 4}) {
        for (int natural_end : {0, 1, 3, 4, 6}) {
            const auto out = run(request(8), spec, accepted, natural_end);
            CHECK(out.emitted[(size_t) natural_end] == 99);
            CHECK(std::count(out.emitted.begin(), out.emitted.end(), 99) == 1);
            CHECK(out.emitted[8] >= 1000); // no second close at budget boundary
        }
        const auto exact = run(request(4, 4), spec, accepted);
        CHECK(exact.emitted == std::vector<int>({10, 11, 12, 13}));
        const auto long_budget = run(request(512, 4), spec, accepted);
        CHECK(long_budget.emitted == exact.emitted);
    }
}

void disabled_tests() {
    // Without a budget, greedy plain decoding is the fixed output reference.
    const auto reference = run(request(0), false);
    for (bool spec : {false, true}) for (int accepted : {0, 1, 4}) {
        for (long long budget : {-1LL, 0LL, 512LL}) {
            const auto out = run(request(budget), spec, accepted);
            CHECK(out.emitted == reference.emitted && out.fed == reference.fed);
            CHECK(out.counter == reference.counter);
        }
        const auto invalid_end = run(request(1, 16, 2048), spec, accepted);
        CHECK(invalid_end.emitted == reference.emitted);
    }
}

void stop_tests() {
    for (bool spec : {false, true}) {
        const auto stop = run(request(4), spec, 4, -1, 12);
        CHECK(stop.finish == "stop" && stop.emitted == std::vector<int>({10, 11, 12}));
        const auto cancel = run(request(1), spec, 4, -1, -1, true);
        CHECK(cancel.finish == "cancel" && cancel.emitted.empty() && cancel.fed.size() == 2);
        const auto fail = run(request(1), spec, 4, -1, -1, false, true);
        CHECK(fail.finish == "cancel" && fail.emitted == std::vector<int>({10}));
    }
}

int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        const std::string mode = argv[1];
        if (mode == "parse") parse_tests();
        else if (mode == "edges") edge_tests();
        else if (mode == "force") force_tests();
        else if (mode == "natural") natural_tests();
        else if (mode == "disabled") disabled_tests();
        else if (mode == "stop") stop_tests();
        else throw std::runtime_error("unknown case");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

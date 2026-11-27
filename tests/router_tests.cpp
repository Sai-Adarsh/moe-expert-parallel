#include "moe.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class Action>
void rejects(Action action, std::string_view fragment) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        require(std::string_view(error.what()).find(fragment) != std::string_view::npos, error.what());
        return;
    }
    throw std::runtime_error("operation unexpectedly succeeded");
}

Workload work(std::size_t ranks, std::size_t tokens) {
    return {ranks, tokens,
        std::vector<float>(ranks * tokens * width),
        std::vector<float>(ranks * experts_per_rank * width * hidden),
        std::vector<float>(ranks * experts_per_rank * width * hidden),
        std::vector<std::vector<Route>>(ranks * ranks)};
}

void routed(const Workload& workload) {
    std::vector<std::vector<std::size_t>> selected(workload.ranks * workload.tokens_per_rank);
    std::vector<float> weights(selected.size());
    for (std::size_t pair = 0; pair < workload.routes.size(); ++pair) {
        for (const auto& route : workload.routes[pair]) {
            require(route.token / workload.tokens_per_rank == pair / workload.ranks, "wrong source rank");
            require(route.expert / experts_per_rank == pair % workload.ranks, "wrong destination rank");
            require(std::isfinite(route.weight) && route.weight > 0 && route.weight < 1, "invalid gate weight");
            weights[route.token] += route.weight;
            selected[route.token].push_back(route.expert);
        }
    }
    for (std::size_t token = 0; token < selected.size(); ++token) {
        require(selected[token].size() == 2 && selected[token][0] != selected[token][1], "not distinct top-two");
        require(std::abs(weights[token] - 1.0f) < 1e-6f, "gate weights do not sum to one");
    }
}

void routing() {
    auto strict = work(4, 64);
    auto balanced = work(4, 64);
    make_routes(strict, false);
    make_routes(balanced, true);
    routed(strict);
    routed(balanced);
    std::array<std::size_t, 8> strict_load{}, balanced_load{};
    for (const auto& pair : strict.routes) {
        for (const auto& route : pair) ++strict_load[route.expert];
    }
    for (const auto& pair : balanced.routes) {
        for (const auto& route : pair) ++balanced_load[route.expert];
    }
    const auto maximum = [](const auto& counts) {
        std::size_t largest = 0;
        for (auto n : counts) if (n > largest) largest = n;
        return largest;
    };
    require(maximum(balanced_load) < maximum(strict_load), "load penalty did not balance the skewed fixture");
}

void limited_capacity(bool balanced, std::size_t capacity) {
    constexpr std::size_t ranks = 4, per_rank = 8, experts = ranks * experts_per_rank;
    auto workload = work(ranks, per_rank);
    for (std::size_t id = 0; id < experts; ++id) {
        workload.up[id * width * hidden] = 1.0f;
        workload.down[id * width * hidden] = 1.0f;
    }
    std::array<std::size_t, experts> loads{};
    std::array<std::uint32_t, ranks * per_rank> masks{};
    std::array<float, masks.size()> weights{};
    for (std::size_t token = 0; token < masks.size(); ++token) {
        workload.activations[token * width] = static_cast<float>(token + 1);
    }

    make_routes(workload, balanced, capacity);
    for (const auto& peer : workload.routes) {
        for (const auto& route : peer) {
            const auto bit = std::uint32_t{1} << route.expert;
            require((masks[route.token] & bit) == 0, "assigned one expert twice");
            masks[route.token] |= bit;
            weights[route.token] += route.weight;
            require(++loads[route.expert] <= capacity, "expert capacity exceeded");
        }
    }
    const auto rowwise = run_cpu(workload, Wire::rowwise);
    const auto packed = run_cpu(workload, Wire::packed);
    require_exact_match(rowwise.result, packed.result);
    std::array<std::size_t, ranks> admitted{};
    for (std::size_t token = 0; token < masks.size(); ++token) {
        const auto slots = std::popcount(masks[token]);
        const auto output = packed.result[token * width];
        if (slots == 0) {
            require(output == 0, "dropped token ran an expert");
            continue;
        }
        require(slots <= 2 && std::abs(weights[token] - 1.0f) < 1e-6f, "gate was not renormalized");
        require(std::abs(output - workload.activations[token * width]) < 1e-5f,
                "admitted token lost its weighted output");
        ++admitted[token / per_rank];
        if (capacity * experts <= masks.size()) {
            require(slots == 1, "added a second route before single-route admission");
        }
    }
    for (const auto n : admitted) {
        require(n == std::min(per_rank, capacity * experts / ranks), "unfair or incomplete admission");
    }
    if (capacity > per_rank / experts_per_rank) {
        require(std::ranges::any_of(masks, [](auto mask) { return std::popcount(mask) == 2; }),
                "available capacity did not fill second routes");
    }
    make_routes(workload, balanced);
    routed(workload);
}

void transport(bool with_remote) {
    constexpr std::size_t tokens = 3;
    auto workload = work(2, tokens);
    for (std::size_t id = 0; id < 4; ++id) {
        workload.up[id * width * hidden] = 1;
        workload.up[id * width * hidden + width + 1] = 1;
        workload.down[id * width * hidden] = static_cast<float>(id + 1);
        workload.down[id * width * hidden + 1] = static_cast<float>(id + 1);
    }
    for (std::size_t token = 0; token < 2 * tokens; ++token) {
        const auto src = token / tokens;
        workload.activations[token * width] = src == 0 ? 2.0f : -2.0f;
        workload.activations[token * width + 1] = src == 0 ? -4.0f : 4.0f;
        const auto local_expert = src == 0 ? 0U : 3U;
        workload.routes[src * 2 + src].push_back({token, local_expert, with_remote ? 0.25f : 1.0f});
        if (with_remote) {
            workload.routes[src * 2 + (1 - src)].push_back({token, 2U - static_cast<unsigned>(src), 0.75f});
        }
    }
    const auto rowwise = run_cpu(workload, Wire::rowwise);
    const auto packed = run_cpu(workload, Wire::packed);
    std::vector<float> expected(workload.activations.size(), 0.0f);
    for (std::size_t token = 0; token < 2 * tokens; ++token) {
        expected[token * width] = token < tokens ? (with_remote ? 5.0f : 2.0f) : (with_remote ? 10.0f : 16.0f);
    }
    require(difference(expected, rowwise.result).max_abs == 0, "rowwise weighted expert output");
    require(difference(expected, packed.result).max_abs == 0, "packed weighted expert output");
    require(rowwise.send_calls == (with_remote ? 12 : 0), "rowwise socket send count");
    require(packed.send_calls == (with_remote ? 4 : 0), "packed socket send count");
    if (!with_remote) {
        require(rowwise.dispatch_socket_ms == 0 && rowwise.return_socket_ms == 0 &&
                packed.dispatch_socket_ms == 0 && packed.return_socket_ms == 0,
                "local-only routes should skip socket transport");
    }
}

void error_metric() {
    const auto finite = difference({1, 0}, {2, 0});
    require(finite.max_abs == 1 && std::abs(finite.rms_abs - std::sqrt(0.5)) < 1e-12, "wrong error metrics");
    rejects([] { difference({1}, {}); }, "shape mismatch");
    rejects([] { difference({1}, {std::numeric_limits<float>::quiet_NaN()}); }, "nonfinite");
    rejects([] { difference({std::numeric_limits<float>::infinity()}, {1}); }, "nonfinite");
    require_exact_match({1, -2}, {1, -2});
    rejects([] { require_exact_match({1, -2}, {1, -3}); }, "CPU outputs differ");
}

void invalid_workloads() {
    auto workload = work(2, 2);
    workload.routes[0].push_back({0, 2, 1.0f});
    rejects([&] { run_cpu(workload, Wire::packed); }, "wrong destination rank");

    workload.routes[0][0] = {2, 0, 1.0f};
    rejects([&] { run_cpu(workload, Wire::rowwise); }, "wrong source rank");

    workload.routes[0][0] = {0, 0, std::numeric_limits<float>::quiet_NaN()};
    rejects([&] { run_cpu(workload, Wire::packed); }, "nonfinite route weight");
    workload.routes[0][0] = {0, 0, 1.0f};

    workload.up.pop_back();
    rejects([&] { run_cpu(workload, Wire::packed); }, "expert weight shape");
    workload.up.push_back(0.0f);
    workload.activations.pop_back();
    rejects([&] { run_cpu(workload, Wire::packed); }, "activation shape");
    workload.activations.push_back(0.0f);
    workload.routes.pop_back();
    rejects([&] { run_cpu(workload, Wire::packed); }, "rank-pair grid");
}

void cannot_open_socket() {
    struct Limits {
        rlimit saved{};
        Limits() {
            require(getrlimit(RLIMIT_NOFILE, &saved) == 0, "read descriptor limit");
            auto exhausted = saved;
            exhausted.rlim_cur = 0;
            require(setrlimit(RLIMIT_NOFILE, &exhausted) == 0, "limit new file descriptors");
        }
        ~Limits() { setrlimit(RLIMIT_NOFILE, &saved); }
    } limit;
    transport(false);
    auto workload = work(2, 1);
    make_routes(workload, false);
    rejects([&] { run_cpu(workload, Wire::packed); }, "open local transport");
}

}  // namespace

int main() try {
    routing();
    for (const bool balanced : {false, true}) {
        for (const std::size_t capacity : {2, 4, 5}) limited_capacity(balanced, capacity);
    }
    transport(true);
    transport(false);
    error_metric();
    invalid_workloads();
    cannot_open_socket();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

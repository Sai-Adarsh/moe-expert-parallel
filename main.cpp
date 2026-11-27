#include "moe.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

static std::size_t count(std::string_view text, std::size_t max) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0 || value > max) {
        throw std::invalid_argument("invalid numeric option");
    }
    return value;
}

static void make_routes(Workload& work, bool balanced) {
    const std::size_t expert_count = work.ranks * experts_per_rank;
    const std::size_t tokens = work.ranks * work.tokens_per_rank;
    std::vector<std::size_t> load(expert_count);
    std::vector<float> scores(expert_count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    const double fair_share = static_cast<double>(tokens * 2) / expert_count;
    for (std::size_t token = 0; token < tokens; ++token) {
        const auto src = token / work.tokens_per_rank;
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            // Skew scores toward the first shard to exercise expert hot spots.
            scores[expert] = noise(rng) + (expert < experts_per_rank ? 0.8f : 0.0f);
        }
        std::array<std::size_t, 2> chosen{};
        for (std::size_t slot = 0; slot < chosen.size(); ++slot) {
            double best = -1e9;
            for (std::size_t expert = 0; expert < expert_count; ++expert) {
                if (slot != 0 && expert == chosen[0]) continue;
                const double score = scores[expert] - (balanced ? 1.5 * load[expert] / fair_share : 0.0);
                if (score > best) {
                    best = score;
                    chosen[slot] = expert;
                }
            }
            ++load[chosen[slot]];
        }
        const auto max_score = std::max(scores[chosen[0]], scores[chosen[1]]);
        const float first = std::exp(scores[chosen[0]] - max_score);
        const float second = std::exp(scores[chosen[1]] - max_score);
        for (std::size_t i = 0; i < chosen.size(); ++i) {
            const auto expert = chosen[i];
            const auto dst = expert / experts_per_rank;
            work.routes[src * work.ranks + dst].push_back({
                token, expert, (i == 0 ? first : second) / (first + second)
            });
        }
    }
}

static Execution run_cpu(const Workload& work) {
    const auto begin = Clock::now();
    std::vector<std::vector<float>> packed(work.routes.size());
    for (std::size_t pair = 0; pair < work.routes.size(); ++pair) {
        auto& buffer = packed[pair];
        buffer.reserve(work.routes[pair].size() * width);
        for (const auto& route : work.routes[pair]) {
            const auto *input = work.activations.data() + route.token * width;
            buffer.insert(buffer.end(), input, input + width);
        }
    }
    const auto dispatch = elapsed(begin);

    const auto ff_start = Clock::now();
    for (std::size_t dst = 0; dst < work.ranks; ++dst) {
        // One rank processes the experts it owns, across every remote source.
        for (std::size_t src = 0; src < work.ranks; ++src) {
            const auto pair = src * work.ranks + dst;
            for (std::size_t i = 0; i < work.routes[pair].size(); ++i) {
                const auto expert = work.routes[pair][i].expert;
                std::span<float, width> vector(packed[pair].data() + i * width, width);
                const float* up = work.up.data() + expert * hidden * width;
                const float* down = work.down.data() + expert * width * hidden;
                std::array<float, hidden> residual{};
                for (std::size_t row = 0; row < hidden; ++row) {
                    float value = 0.0f;
                    for (std::size_t col = 0; col < width; ++col) {
                        value += up[row * width + col] * vector[col];
                    }
                    residual[row] = std::max(0.0f, value);
                }
                for (std::size_t row = 0; row < width; ++row) {
                    float value = 0.0f;
                    for (std::size_t col = 0; col < hidden; ++col) {
                        value += down[row * hidden + col] * residual[col];
                    }
                    vector[row] = value;
                }
            }
        }
    }
    const auto feed_forward = elapsed(ff_start);

    const auto combine_start = Clock::now();
    std::vector<float> result(work.activations.size(), 0.0f);
    for (std::size_t pair = 0; pair < work.routes.size(); ++pair) {
        for (std::size_t i = 0; i < work.routes[pair].size(); ++i) {
            const auto& route = work.routes[pair][i];
            for (std::size_t d = 0; d < width; ++d) {
                result[route.token * width + d] += route.weight * packed[pair][i * width + d];
            }
        }
    }
    return {std::move(result), dispatch, feed_forward, elapsed(combine_start)};
}

static void print_routing(const Workload& work, std::string_view name) {
    std::size_t peer_messages = 0;
    std::size_t remote_rows = 0;
    std::vector<std::size_t> load(work.ranks * experts_per_rank);
    for (std::size_t src = 0; src < work.ranks; ++src) {
        for (std::size_t dst = 0; dst < work.ranks; ++dst) {
            const auto& route = work.routes[src * work.ranks + dst];
            for (const auto& row : route) ++load[row.expert];
            if (src != dst && !route.empty()) {
                ++peer_messages;
                remote_rows += route.size();
            }
        }
    }
    const double mean = 2.0 * work.ranks * work.tokens_per_rank / load.size();
    std::cout << name << " expert load";
    for (auto value : load) std::cout << ' ' << value;
    std::cout << "\n  max/mean=" << *std::max_element(load.begin(), load.end()) / mean
              << " remote_bytes_per_round_trip=" << 2 * remote_rows * width * sizeof(float)
              << " remote_messages_rowwise=" << 2 * remote_rows
              << " remote_messages_packed=" << 2 * peer_messages << '\n';
}

static void print_timing(std::string_view label, const Execution& exec) {
    std::cout << label << " dispatch_ms=" << exec.dispatch_ms
              << " expert_ms=" << exec.feed_forward_ms
              << " return_combine_ms=" << exec.combine_ms << '\n';
}

int main(int argc, char** argv) try {
    std::size_t ranks = 2;
    std::size_t tokens = 256;
    bool nccl = false;
    bool balanced = true;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string_view option(argv[arg]);
        if (option == "--nccl") nccl = true;
        else if (option == "--strict-top-k") balanced = false;
        else if (option == "--ranks" && arg + 1 < argc) ranks = count(argv[++arg], 16);
        else if (option == "--tokens-per-rank" && arg + 1 < argc) tokens = count(argv[++arg], 16384);
        else throw std::invalid_argument("usage: moe_router [--ranks N] [--tokens-per-rank N] [--strict-top-k] [--nccl]");
    }
    if (ranks < 2) throw std::invalid_argument("at least two ranks are required");
#ifndef MOE_HAS_NCCL
    if (nccl) throw std::invalid_argument("reconfigure with -DMOE_ENABLE_NCCL=ON");
#endif

    Workload work{
        ranks, tokens, std::vector<float>(ranks * tokens * width),
        std::vector<float>(ranks * experts_per_rank * hidden * width),
        std::vector<float>(ranks * experts_per_rank * width * hidden),
        std::vector<std::vector<Route>>(ranks * ranks)
    };
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> value(-0.2f, 0.2f);
    for (auto* matrix : {&work.activations, &work.up, &work.down}) {
        std::generate(matrix->begin(), matrix->end(), [&] { return value(rng); });
    }

    std::cout << std::fixed << std::setprecision(4);
    make_routes(work, false);
    print_routing(work, "strict");
    if (balanced) {
        for (auto& peer : work.routes) peer.clear();
        make_routes(work, true);
        print_routing(work, "balanced");
    }
    auto reference = run_cpu(work);
    print_timing("cpu", reference);
#ifdef MOE_HAS_NCCL
    if (nccl) {
        const auto device = run_nccl(work);
        print_timing("nccl", device);
        float error = 0.0f;
        for (std::size_t i = 0; i < reference.result.size(); ++i) {
            error = std::max(error, std::abs(reference.result[i] - device.result[i]));
        }
        std::cout << "cpu_gpu_max_abs_error=" << error << '\n';
        if (error > 1e-5f) return 2;
    }
#endif
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

#include "moe.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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
              << " modeled_remote_bytes_per_round_trip=" << 2 * remote_rows * width * sizeof(float)
              << " modeled_peer_messages_rowwise=" << 2 * remote_rows
              << " modeled_peer_messages_packed=" << 2 * peer_messages << '\n';
}

struct Difference {
    double max_abs = 0;
    double rms_abs = 0;
};

static Difference difference(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) throw std::runtime_error("expert output shape mismatch");
    Difference errors;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!std::isfinite(left[i]) || !std::isfinite(right[i])) {
            throw std::runtime_error("nonfinite expert output at element " + std::to_string(i));
        }
        const double error = static_cast<double>(left[i]) - right[i];
        errors.max_abs = std::max(errors.max_abs, std::abs(error));
        errors.rms_abs += error * error;
    }
    errors.rms_abs = std::sqrt(errors.rms_abs / left.size());
    return errors;
}

static std::vector<std::uint32_t> route_masks(const Workload& work) {
    std::vector<std::uint32_t> result(work.ranks * work.tokens_per_rank);
    for (const auto& peer : work.routes) {
        for (const auto& route : peer) {
            result[route.token] |= std::uint32_t{1} << route.expert;
        }
    }
    return result;
}

static std::vector<float> compare_wires(const Workload& work, std::string_view policy, std::size_t trials) {
    print_routing(work, policy);
    std::cout << "policy,trial,wire,pack_ms,dispatch_socket_ms,expert_wall_ms,peak_rank_expert_ms,"
                 "return_socket_ms,combine_ms,observed_send_calls\n";
    std::vector<float> result;
    std::array<std::vector<double>, 2> transport_samples;
    for (std::size_t trial = 0; trial <= trials; ++trial) {
        std::array<CpuExecution, 2> samples;
        for (std::size_t offset = 0; offset < samples.size(); ++offset) {
            const auto i = (trial + offset) % samples.size();
            samples[i] = run_cpu(work, i == 0 ? Wire::rowwise : Wire::packed);
        }
        if (difference(samples[0].result, samples[1].result).max_abs != 0) {
            throw std::runtime_error("rowwise and packed CPU outputs differ");
        }
        if (trial == 0) continue;
        const auto print = [&](std::string_view wire, const CpuExecution& result) {
            std::cout << policy << ',' << trial << ',' << wire << ',' << result.pack_ms << ','
                      << result.dispatch_socket_ms << ',' << result.feed_forward_ms << ','
                      << result.peak_rank_ffn_ms << ',' << result.return_socket_ms << ','
                      << result.combine_ms << ',' << result.send_calls << '\n';
        };
        print("rowwise", samples[0]);
        print("packed", samples[1]);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            transport_samples[i].push_back(samples[i].dispatch_socket_ms + samples[i].return_socket_ms);
        }
        result = std::move(samples[1].result);
    }
    const auto median = [](std::vector<double>& values) {
        std::sort(values.begin(), values.end());
        const auto middle = values.size() / 2;
        return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
    };
    std::cout << policy << " transport=unix_socketpair parity=exact"
              << " median_rowwise_round_trip_ms=" << median(transport_samples[0])
              << " median_packed_round_trip_ms=" << median(transport_samples[1]) << '\n';
    return result;
}

#ifdef MOE_HAS_NCCL
static void compare_nccl(const Workload& work, const std::vector<float>& reference, std::string_view policy) {
    const auto device = run_nccl(work);
    const auto errors = difference(reference, device.result);
    std::cout << policy << " transport=nccl packed"
              << " dispatch_ms=" << device.dispatch_ms << " expert_ms=" << device.feed_forward_ms
              << " return_combine_ms=" << device.combine_ms << " cpu_gpu_max_abs=" << errors.max_abs
              << " cpu_gpu_rms_abs=" << errors.rms_abs << '\n';
    if (errors.max_abs > 1e-5) throw std::runtime_error("NCCL output exceeds CPU error tolerance");
}
#endif

int main(int argc, char** argv) try {
    std::signal(SIGPIPE, SIG_IGN);
    std::size_t ranks = 2;
    std::size_t tokens = 256;
    std::size_t trials = 5;
    bool nccl = false;
    bool balanced = true;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string_view option(argv[arg]);
        if (option == "--nccl") nccl = true;
        else if (option == "--strict-top-k") balanced = false;
        else if (option == "--ranks" && arg + 1 < argc) ranks = count(argv[++arg], 16);
        else if (option == "--tokens-per-rank" && arg + 1 < argc) tokens = count(argv[++arg], 16384);
        else if (option == "--trials" && arg + 1 < argc) trials = count(argv[++arg], 32);
        else throw std::invalid_argument("usage: moe_router [--ranks N] [--tokens-per-rank N] [--trials N] [--strict-top-k] [--nccl]");
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
    const auto strict_routes = route_masks(work);
    const auto strict_result = compare_wires(work, "strict", trials);
#ifdef MOE_HAS_NCCL
    if (nccl) compare_nccl(work, strict_result, "strict");
#endif
    if (balanced) {
        for (auto& peer : work.routes) peer.clear();
        make_routes(work, true);
        const auto balanced_result = compare_wires(work, "balanced", trials);
        const auto balanced_routes = route_masks(work);
        std::size_t changed = 0;
        for (std::size_t i = 0; i < strict_routes.size(); ++i) {
            changed += strict_routes[i] != balanced_routes[i];
        }
        const auto drift = difference(strict_result, balanced_result);
        std::cout << "policy_delta changed_top_two_sets=" << changed << " total_tokens=" << strict_routes.size()
                  << " output_max_abs=" << drift.max_abs << " output_rms_abs=" << drift.rms_abs << '\n';
#ifdef MOE_HAS_NCCL
        if (nccl) compare_nccl(work, balanced_result, "balanced");
#endif
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

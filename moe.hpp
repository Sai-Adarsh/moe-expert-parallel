#pragma once

#include <cstddef>
#include <vector>

inline constexpr std::size_t width = 32;
inline constexpr std::size_t hidden = 64;
inline constexpr std::size_t experts_per_rank = 2;

struct Route {
    std::size_t token;
    std::size_t expert;
    float weight;
};

struct Workload {
    std::size_t ranks;
    std::size_t tokens_per_rank;
    std::vector<float> activations;
    std::vector<float> up;
    std::vector<float> down;
    // Source-major owning rank pairs. The two all-to-alls use reversed index order.
    std::vector<std::vector<Route>> routes;
};

struct Execution {
    std::vector<float> result;
    double dispatch_ms;
    double feed_forward_ms;
    double combine_ms;
};

Execution run_nccl(const Workload& work);

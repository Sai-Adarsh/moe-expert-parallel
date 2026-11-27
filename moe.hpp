#pragma once

#include <cstddef>
#include <optional>
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

enum class Wire { rowwise, packed };

struct CpuExecution {
    std::vector<float> result;
    double pack_ms;
    double dispatch_socket_ms;
    double feed_forward_ms;
    double peak_rank_ffn_ms;
    double return_socket_ms;
    double combine_ms;
    std::size_t send_calls;
};

CpuExecution run_cpu(const Workload& work, Wire wire);
Execution run_nccl(const Workload& work);

void make_routes(Workload& work, bool balanced, std::optional<std::size_t> capacity = std::nullopt);

struct Difference {
    double max_abs = 0;
    double rms_abs = 0;
};

Difference difference(const std::vector<float>& left, const std::vector<float>& right);
void require_exact_match(const std::vector<float>& rowwise, const std::vector<float>& packed);

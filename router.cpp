#include "moe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

void make_routes(Workload& work, bool balanced) {
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

Difference difference(const std::vector<float>& left, const std::vector<float>& right) {
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

void require_exact_match(const std::vector<float>& rowwise, const std::vector<float>& packed) {
    if (difference(rowwise, packed).max_abs != 0) {
        throw std::runtime_error("rowwise and packed CPU outputs differ");
    }
}

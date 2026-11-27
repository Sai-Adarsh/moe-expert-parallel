#include "moe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>

void make_routes(Workload& work, bool balanced, std::optional<std::size_t> capacity) {
    const std::size_t expert_count = work.ranks * experts_per_rank;
    const std::size_t tokens = work.ranks * work.tokens_per_rank;
    std::vector<std::size_t> load(expert_count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    const double fair_share = static_cast<double>(tokens * 2) / expert_count;
    for (auto& peer : work.routes) peer.clear();

    const auto fill_scores = [&](std::span<float> scores) {
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            // Skew scores toward the first shard to exercise expert hot spots.
            scores[expert] = noise(rng) + (expert < experts_per_rank ? 0.8f : 0.0f);
        }
    };
    const auto select = [&](std::span<const float> scores, std::size_t exclude) {
        double best = -std::numeric_limits<double>::infinity();
        std::size_t chosen = expert_count;
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            if (expert == exclude || (capacity && load[expert] >= *capacity)) continue;
            const double score = scores[expert] - (balanced ? 1.5 * load[expert] / fair_share : 0.0);
            if (score > best) {
                best = score;
                chosen = expert;
            }
        }
        if (chosen != expert_count) ++load[chosen];
        return chosen;
    };
    const auto append = [&](std::size_t token, std::size_t expert, float weight) {
        const auto source = token / work.tokens_per_rank;
        const auto destination = expert / experts_per_rank;
        work.routes[source * work.ranks + destination].push_back({token, expert, weight});
    };
    const auto emit = [&](std::size_t token, std::span<const float> scores,
                          std::size_t first, std::size_t second) {
        if (second == expert_count) {
            append(token, first, 1.0f);
            return;
        }
        const auto max_score = std::max(scores[first], scores[second]);
        const float first_weight = std::exp(scores[first] - max_score);
        const float second_weight = std::exp(scores[second] - max_score);
        append(token, first, first_weight / (first_weight + second_weight));
        append(token, second, second_weight / (first_weight + second_weight));
    };

    if (!capacity) {
        std::vector<float> scores(expert_count);
        for (std::size_t token = 0; token < tokens; ++token) {
            fill_scores(scores);
            const auto first = select(scores, expert_count);
            const auto second = select(scores, first);
            emit(token, scores, first, second);
        }
        return;
    }

    std::vector<float> logits(tokens * expert_count);
    const auto scores_at = [&](std::size_t token) {
        return std::span(logits).subspan(token * expert_count, expert_count);
    };
    for (std::size_t token = 0; token < tokens; ++token) fill_scores(scores_at(token));
    // Give every token a first slot before filling second slots. Round-robin
    // source ranks so early source ranks do not take the entire budget.
    const auto token_at = [&](std::size_t order) {
        return (order % work.ranks) * work.tokens_per_rank + order / work.ranks;
    };
    std::vector<std::size_t> primary(tokens);
    for (std::size_t order = 0; order < tokens; ++order) {
        const auto token = token_at(order);
        primary[token] = select(scores_at(token), expert_count);
    }
    for (std::size_t order = 0; order < tokens; ++order) {
        const auto token = token_at(order);
        if (primary[token] == expert_count) continue;
        const auto scores = scores_at(token);
        emit(token, scores, primary[token], select(scores, primary[token]));
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

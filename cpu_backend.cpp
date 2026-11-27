#include "moe.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <exception>
#include <latch>
#include <span>
#include <system_error>
#include <thread>
#include <utility>

using Clock = std::chrono::steady_clock;

namespace {

double elapsed(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

using Rows = std::vector<std::vector<float>>;

class Channel {
public:
    Channel() {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd_) == -1) {
            throw std::system_error(errno, std::generic_category(), "open local transport");
        }
    }
    ~Channel() {
        close(fd_[0]);
        close(fd_[1]);
    }
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    int sender() const { return fd_[0]; }
    int receiver() const { return fd_[1]; }
    void shutdown_both() const {
        shutdown(fd_[0], SHUT_RDWR);
        shutdown(fd_[1], SHUT_RDWR);
    }

private:
    int fd_[2];
};

std::size_t send_all(int fd, std::span<const float> block) {
    auto bytes = std::as_bytes(block);
    std::size_t calls = 0;
    while (!bytes.empty()) {
        const auto n = send(fd, bytes.data(), bytes.size(), 0);
        if (n == -1 && errno == EINTR) continue;
        if (n <= 0) throw std::system_error(n == 0 ? EIO : errno, std::generic_category(), "send MoE payload");
        bytes = bytes.subspan(static_cast<std::size_t>(n));
        ++calls;
    }
    return calls;
}

void receive_all(int fd, std::span<float> block) {
    auto bytes = std::as_writable_bytes(block);
    while (!bytes.empty()) {
        const auto n = recv(fd, bytes.data(), bytes.size(), 0);
        if (n == -1 && errno == EINTR) continue;
        if (n <= 0) throw std::system_error(n == 0 ? EIO : errno, std::generic_category(), "receive MoE payload");
        bytes = bytes.subspan(static_cast<std::size_t>(n));
    }
}

struct Exchange {
    Rows rows;
    double ms;
    std::size_t send_calls;
};

Exchange transfer(Rows outgoing, std::size_t ranks, Wire wire) {
    Rows incoming(outgoing.size());
    std::vector<std::size_t> peers;
    for (std::size_t pair = 0; pair < outgoing.size(); ++pair) {
        if (pair / ranks == pair % ranks) {
            incoming[pair] = std::move(outgoing[pair]);
        } else if (!outgoing[pair].empty()) {
            incoming[pair].resize(outgoing[pair].size());
            peers.push_back(pair);
        }
    }

    if (peers.empty()) return {std::move(incoming), 0, 0};

    // A single socket pair meters local CPU serialization costs. Recv runs on a
    // separate thread so large vectors can exceed kernel socket-buffer capacity.
    Channel channel;
    std::exception_ptr error;
    std::latch ready(1);
    const auto segment = [wire](std::size_t row_elements) {
        return wire == Wire::rowwise ? width : row_elements;
    };
    std::jthread receiver([&] {
        ready.count_down();
        try {
            for (auto pair : peers) {
                std::span<float> buffer(incoming[pair]);
                const auto chunk = segment(buffer.size());
                for (std::size_t begin = 0; begin < buffer.size(); begin += chunk) {
                    receive_all(channel.receiver(), buffer.subspan(begin, chunk));
                }
            }
        } catch (...) {
            error = std::current_exception();
            channel.shutdown_both();
        }
    });
    ready.wait();
    auto start = Clock::now();
    std::size_t calls = 0;
    try {
        for (auto pair : peers) {
            std::span<const float> buffer(outgoing[pair]);
            const auto chunk = segment(buffer.size());
            for (std::size_t begin = 0; begin < buffer.size(); begin += chunk) {
                calls += send_all(channel.sender(), buffer.subspan(begin, chunk));
            }
        }
    } catch (...) {
        channel.shutdown_both();
        throw;
    }
    receiver.join();
    if (error) std::rethrow_exception(error);
    return {std::move(incoming), elapsed(start), calls};
}

void expert(const Workload& work, std::size_t id, std::span<float, width> vector) {
    const float* up = work.up.data() + id * hidden * width;
    const float* down = work.down.data() + id * width * hidden;
    std::array<float, hidden> activation{};
    for (std::size_t row = 0; row < hidden; ++row) {
        float value = 0.0f;
        for (std::size_t col = 0; col < width; ++col) {
            value += up[row * width + col] * vector[col];
        }
        activation[row] = std::max(0.0f, value);
    }
    for (std::size_t row = 0; row < width; ++row) {
        float value = 0.0f;
        for (std::size_t col = 0; col < hidden; ++col) {
            value += down[row * hidden + col] * activation[col];
        }
        vector[row] = value;
    }
}

}  // namespace

CpuExecution run_cpu(const Workload& work, Wire wire) {
    validate_workload(work);
    const auto begin = Clock::now();
    Rows packed(work.routes.size());
    for (std::size_t pair = 0; pair < work.routes.size(); ++pair) {
        auto& buffer = packed[pair];
        buffer.reserve(work.routes[pair].size() * width);
        for (const auto& route : work.routes[pair]) {
            const auto* start = work.activations.data() + route.token * width;
            buffer.insert(buffer.end(), start, start + width);
        }
    }
    const double pack_ms = elapsed(begin);
    auto dispatch = transfer(std::move(packed), work.ranks, wire);

    std::vector<double> rank_times(work.ranks);
    const auto ff_start = Clock::now();
    {
        std::vector<std::jthread> ranks;
        ranks.reserve(work.ranks);
        for (std::size_t dst = 0; dst < work.ranks; ++dst) {
            ranks.emplace_back([&, dst] {
                const auto start = Clock::now();
                for (std::size_t src = 0; src < work.ranks; ++src) {
                    const auto pair = src * work.ranks + dst;
                    for (std::size_t row = 0; row < work.routes[pair].size(); ++row) {
                        expert(work, work.routes[pair][row].expert,
                               std::span<float, width>(dispatch.rows[pair].data() + row * width, width));
                    }
                }
                rank_times[dst] = elapsed(start);
            });
        }
    }
    const double feed_forward = elapsed(ff_start);
    auto returned = transfer(std::move(dispatch.rows), work.ranks, wire);

    const auto combine_start = Clock::now();
    std::vector<float> result(work.activations.size(), 0.0f);
    for (std::size_t pair = 0; pair < work.routes.size(); ++pair) {
        for (std::size_t i = 0; i < work.routes[pair].size(); ++i) {
            const auto& route = work.routes[pair][i];
            for (std::size_t j = 0; j < width; ++j) {
                result[route.token * width + j] += route.weight * returned.rows[pair][i * width + j];
            }
        }
    }
    return {std::move(result), pack_ms, dispatch.ms, feed_forward,
            *std::max_element(rank_times.begin(), rank_times.end()), returned.ms,
            elapsed(combine_start), dispatch.send_calls + returned.send_calls};
}

#include "moe.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check(ncclResult_t status) {
    if (status != ncclSuccess) throw std::runtime_error(ncclGetErrorString(status));
}

template <class T> struct Buffer {
    T* data = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (data) cudaFree(data); }

    void alloc(std::size_t count) {
        check(cudaMalloc(&data, std::max(count, std::size_t{1}) * sizeof(T)));
    }
};

struct Rank {
    explicit Rank(std::size_t id) : id(id) {
        check(cudaSetDevice(static_cast<int>(id)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    ~Rank() {
        cudaSetDevice(static_cast<int>(id));
        if (stream) cudaStreamDestroy(stream);
    }

    std::size_t id;
    cudaStream_t stream = nullptr;
    ncclComm_t comm = nullptr;
    Buffer<float> send, receive, expert_output, returned, up, down;
    Buffer<int> expert_ids;
    std::vector<std::size_t> outgoing;
    std::vector<std::size_t> incoming;
    std::vector<float> host_activations;
    std::vector<int> host_experts;
    std::vector<float> host_returns;
};

class CommunicatorScope {
  public:
    explicit CommunicatorScope(const std::vector<std::unique_ptr<Rank>>& ranks) : ranks_(ranks) {}

    ~CommunicatorScope() {
        // Aborting unfinished operations requires every local rank to participate.
        std::vector<std::jthread> aborters;
        for (const auto& rank : ranks_) {
            if (rank->comm) {
                aborters.emplace_back([rank = rank.get()] {
                    cudaSetDevice(static_cast<int>(rank->id));
                    ncclCommAbort(rank->comm);
                });
            }
        }
    }

    void finalize() const {
        check(ncclGroupStart());
        ncclResult_t status = ncclSuccess;
        for (const auto& rank : ranks_) {
            const auto current = ncclCommFinalize(rank->comm);
            if (current != ncclSuccess) status = current;
        }
        const auto end = ncclGroupEnd();
        check(status);
        check(end);
        for (const auto& rank : ranks_) {
            const auto current = ncclCommDestroy(rank->comm);
            rank->comm = nullptr;
            check(current);
        }
    }

  private:
    const std::vector<std::unique_ptr<Rank>>& ranks_;
};

__global__ void expert_ffn(const float* input, const int* expert_ids,
                           const float* up, const float* down, float* output) {
    const int row = threadIdx.x;
    const float* activation = input + blockIdx.x * width;
    const int expert = expert_ids[blockIdx.x];
    __shared__ float intermediate[hidden];
    float value = 0.0f;
    for (std::size_t col = 0; col < width; ++col) {
        value += up[(expert * hidden + row) * width + col] * activation[col];
    }
    intermediate[row] = fmaxf(0.0f, value);
    __syncthreads();
    if (row < width) {
        value = 0.0f;
        for (std::size_t col = 0; col < hidden; ++col) {
            value += down[(expert * width + row) * hidden + col] * intermediate[col];
        }
        output[blockIdx.x * width + row] = value;
    }
}

void synchronize(const std::vector<std::unique_ptr<Rank>>& ranks) {
    for (const auto& rank : ranks) {
        check(cudaSetDevice(static_cast<int>(rank->id)));
        check(cudaStreamSynchronize(rank->stream));
    }
}

void exchange(const Workload& work, const std::vector<std::unique_ptr<Rank>>& ranks,
              bool returning) {
    for (const auto& rank : ranks) {
        const auto id = rank->id;
        const auto local_count = work.routes[id * work.ranks + id].size() * width;
        if (!local_count) continue;
        const auto send_offset = (returning ? rank->incoming : rank->outgoing)[id] * width;
        const auto recv_offset = (returning ? rank->outgoing : rank->incoming)[id] * width;
        const float* source = returning ? rank->expert_output.data : rank->send.data;
        float* destination = returning ? rank->returned.data : rank->receive.data;
        check(cudaSetDevice(static_cast<int>(id)));
        check(cudaMemcpyAsync(destination + recv_offset, source + send_offset,
                              local_count * sizeof(float), cudaMemcpyDeviceToDevice, rank->stream));
    }

    check(ncclGroupStart());
    ncclResult_t communication_error = ncclSuccess;
    cudaError_t device_error = cudaSuccess;
    for (const auto& rank : ranks) {
        const auto id = rank->id;
        const auto status = cudaSetDevice(static_cast<int>(id));
        if (status != cudaSuccess) {
            device_error = status;
            continue;
        }
        const auto& send_offsets = returning ? rank->incoming : rank->outgoing;
        const auto& recv_offsets = returning ? rank->outgoing : rank->incoming;
        const float* sends = returning ? rank->expert_output.data : rank->send.data;
        float* receives = returning ? rank->returned.data : rank->receive.data;
        for (std::size_t peer = 0; peer < work.ranks; ++peer) {
            if (peer == id) continue;
            const auto send_count = (send_offsets[peer + 1] - send_offsets[peer]) * width;
            const auto recv_count = (recv_offsets[peer + 1] - recv_offsets[peer]) * width;
            if (send_count) {
                const auto result = ncclSend(sends + send_offsets[peer] * width, send_count,
                                             ncclFloat, static_cast<int>(peer), rank->comm, rank->stream);
                if (result != ncclSuccess) communication_error = result;
            }
            if (recv_count) {
                const auto result = ncclRecv(receives + recv_offsets[peer] * width, recv_count,
                                             ncclFloat, static_cast<int>(peer), rank->comm, rank->stream);
                if (result != ncclSuccess) communication_error = result;
            }
        }
    }
    const auto end = ncclGroupEnd();
    check(device_error);
    check(communication_error);
    check(end);
}

} // namespace

Execution run_nccl(const Workload& work) {
    validate_workload(work);
    int available = 0;
    check(cudaGetDeviceCount(&available));
    if (available < static_cast<int>(work.ranks)) {
        throw std::runtime_error("NCCL requires one visible CUDA device per rank");
    }

    std::vector<std::unique_ptr<Rank>> ranks;
    for (std::size_t id = 0; id < work.ranks; ++id) {
        auto rank = std::make_unique<Rank>(id);
        for (std::size_t peer = 0; peer < work.ranks; ++peer) {
            rank->outgoing.push_back(rank->host_activations.size() / width);
            for (const auto& route : work.routes[id * work.ranks + peer]) {
                const float* start = work.activations.data() + route.token * width;
                rank->host_activations.insert(rank->host_activations.end(), start, start + width);
            }
            rank->incoming.push_back(rank->host_experts.size());
            for (const auto& route : work.routes[peer * work.ranks + id]) {
                rank->host_experts.push_back(static_cast<int>(route.expert % experts_per_rank));
            }
        }
        rank->outgoing.push_back(rank->host_activations.size() / width);
        rank->incoming.push_back(rank->host_experts.size());
        rank->host_returns.resize(rank->host_activations.size());
        rank->send.alloc(rank->host_activations.size());
        rank->returned.alloc(rank->host_returns.size());
        rank->receive.alloc(rank->host_experts.size() * width);
        rank->expert_output.alloc(rank->host_experts.size() * width);
        rank->expert_ids.alloc(rank->host_experts.size());
        const std::size_t weight_count = experts_per_rank * hidden * width;
        rank->up.alloc(weight_count);
        rank->down.alloc(weight_count);
        check(cudaMemcpyAsync(rank->up.data, work.up.data() + id * weight_count,
                              weight_count * sizeof(float), cudaMemcpyHostToDevice, rank->stream));
        check(cudaMemcpyAsync(rank->down.data, work.down.data() + id * weight_count,
                              weight_count * sizeof(float), cudaMemcpyHostToDevice, rank->stream));
        ranks.push_back(std::move(rank));
    }
    const auto communicator_scope = CommunicatorScope(ranks);
    std::vector<ncclComm_t> communicators(work.ranks, nullptr);
    const auto init = ncclCommInitAll(communicators.data(), static_cast<int>(work.ranks), nullptr);
    for (std::size_t i = 0; i < ranks.size(); ++i) ranks[i]->comm = communicators[i];
    check(init);
    synchronize(ranks);

    const auto dispatch_start = Clock::now();
    for (const auto& rank : ranks) {
        check(cudaSetDevice(static_cast<int>(rank->id)));
        if (!rank->host_activations.empty()) {
            check(cudaMemcpyAsync(rank->send.data, rank->host_activations.data(),
                                  rank->host_activations.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, rank->stream));
        }
        if (!rank->host_experts.empty()) {
            check(cudaMemcpyAsync(rank->expert_ids.data, rank->host_experts.data(),
                                  rank->host_experts.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, rank->stream));
        }
    }
    exchange(work, ranks, false);
    synchronize(ranks);
    const auto dispatch = elapsed(dispatch_start);

    const auto ff_start = Clock::now();
    for (const auto& rank : ranks) {
        if (rank->host_experts.empty()) continue;
        check(cudaSetDevice(static_cast<int>(rank->id)));
        expert_ffn<<<rank->host_experts.size(), hidden, 0, rank->stream>>>(
            rank->receive.data, rank->expert_ids.data, rank->up.data,
            rank->down.data, rank->expert_output.data);
        check(cudaGetLastError());
    }
    synchronize(ranks);
    const auto feed_forward = elapsed(ff_start);

    const auto combine_start = Clock::now();
    exchange(work, ranks, true);
    for (const auto& rank : ranks) {
        if (rank->host_returns.empty()) continue;
        check(cudaSetDevice(static_cast<int>(rank->id)));
        check(cudaMemcpyAsync(rank->host_returns.data(), rank->returned.data,
                              rank->host_returns.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, rank->stream));
    }
    synchronize(ranks);
    std::vector<float> result(work.activations.size(), 0.0f);
    for (const auto& rank : ranks) {
        for (std::size_t dst = 0; dst < work.ranks; ++dst) {
            const auto& rows = work.routes[rank->id * work.ranks + dst];
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const auto& route = rows[i];
                const float* output = rank->host_returns.data() + (rank->outgoing[dst] + i) * width;
                for (std::size_t d = 0; d < width; ++d) {
                    result[route.token * width + d] += route.weight * output[d];
                }
            }
        }
    }
    const auto combine = elapsed(combine_start);
    communicator_scope.finalize();
    return {std::move(result), dispatch, feed_forward, combine};
}

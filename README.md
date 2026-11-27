# MoE expert parallel router

Each rank owns two experts. Every token selects two distinct experts and combines their two-layer feed-forward outputs using normalized gate weights. The input generator skews the gate toward rank 0. The balanced policy subtracts a running load penalty before choosing the two experts. Weighting still uses the original gate scores for the chosen experts.

## Measure a local transport baseline

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/moe_router --ranks 4 --tokens-per-rank 1024 --trials 9
```

Both wire modes send the same off-rank FP32 activation rows and expert outputs through a real Unix-domain socket pair. Rowwise issues a send for each activation row. Packed sends one contiguous buffer per source and destination pair. The transport runs serially over one socket pair. It excludes socket creation and receiver-buffer allocation from timing. A separate receiver thread drains the stream.

The program rotates the mode order, drops one warmup pair, and verifies exact output agreement for both modes at each trial. One CPU thread per destination runs that rank's experts. The output separates packing, dispatch socket time, expert wall time, longest rank expert time, return socket time and weighted combine. The modeled message and payload-byte totals come from routes. The observed send count records successful send calls, which can differ from peer-pair counts after a short send.

Use `--strict-top-k` to skip the balanced policy. The comparison between policies counts changed expert sets and measures output drift. A lower load ratio comes with different model output; the activation differences are not a language quality score.

### Recorded local run

Apple M3 Pro, macOS 27.2.1, Apple Clang 21.0.0, Release build. Four ranks, 1,024 tokens per rank, nine measured pairs per policy. The [raw output](measurements/m3-pro-socketpair.txt) includes phase timings, rank loads and observed sends.

| Policy | Max expert load / mean | Rowwise median socket round trip | Packed median socket round trip | Packed median longest-rank compute |
|---|---:|---:|---:|---:|
| Strict | 2.354 | 12.466 ms | 1.946 ms | 5.095 ms |
| Balanced | 1.386 | 10.257 ms | 1.302 ms | 3.303 ms |

With this input, balancing changed 1,830 of 4,096 top-two expert sets (44.7%) and produced an activation RMSE of 0.0217. Packing preserved the output exactly within each policy. These timing differences quantify one machine's local socket and scheduling overhead. They are not a prediction for GPU or fabric performance.

## CUDA and NCCL

On Linux with CUDA, NCCL and at least two visible NVIDIA GPUs:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMOE_ENABLE_NCCL=ON
cmake --build build-gpu
./build-gpu/moe_router --nccl --ranks 2 --tokens-per-rank 1024
```

One process assigns one GPU per rank and places each rank's weights on its device. Grouped NCCL sends and receives exchange packed buffers before and after the CUDA feed-forward pass. The optional run compares each policy's CUDA output with the corresponding CPU result and rejects nonfinite values or excessive error. Its phase times include host transfers and synchronization. Set `CMAKE_CUDA_ARCHITECTURES` for the devices being used.

No NCCL measurements are included. To compare a real fabric, record GPU topology, ranks, workload shape, precision, expert load, and both the host-inclusive and device-only transfer intervals. Both paths use synthetic activations and weights.

[NCCL point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)

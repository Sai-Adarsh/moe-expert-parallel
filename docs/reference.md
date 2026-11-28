# MoE expert parallel router

Run the commands in this guide from the repository root.

Each rank owns two experts. By default, a token selects two distinct experts and combines their two-layer feed-forward outputs using normalized gate weights. The input generator skews the gate toward rank 0. The balanced policy subtracts a running load penalty before choosing experts, then weights them using the original scores.

## Bound the work per expert

Build the CPU example with the CMake commands in the local transport section below.

`--expert-capacity N` caps each expert at `N` activation rows for the batch. Routing sweeps the source ranks round-robin and tries to give each token one expert before assigning second experts. A token with one available expert gets weight 1. A token with no route gets a zero feed-forward contribution, and the CLI counts it under `dropped_tokens`. Duplicated experts are excluded. The bounded router stores the batch's gate scores for its second pass.

```sh
for cap in 100 128 192; do
  ./build/moe_router --ranks 4 --tokens-per-rank 256 --trials 5 --expert-capacity "$cap"
done
```

Both policies produced these counts on the 1,024-token CPU workload on an Apple M3 Pro. Rowwise and packed outputs agreed exactly within each policy for all five measured pairs.

| Capacity per expert | Assignments | Single route | Two routes | Dropped |
|---|---:|---:|---:|---:|
| 100 | 800 | 800 | 0 | 224 |
| 128 | 1,024 | 1,024 | 0 | 0 |
| 192 | 1,536 | 512 | 512 | 0 |

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

Apple M3 Pro, macOS 27.2.1, Apple Clang 21.0.0, Release build. Four ranks, 1,024 tokens per rank, nine measured pairs per policy. The [raw output](../measurements/m3-pro-socketpair.txt) includes phase timings, rank loads and observed sends.

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

## Tests and coverage

The suite verifies the routing invariants and both socket transports against an independently calculated expert output. Capacity tests check per-expert bounds, first-route admission across ranks, renormalization, and zero contribution for dropped tokens. The suite also checks socket open failure, malformed CLI options, and nonfinite or mismatched results.

Both execution backends check the rank-pair grid, tensor dimensions, route ownership, and finite gate weights before dispatch. CPU tests cover the shared validation, including a route sent to a rank that does not own its expert.

```sh
cmake -S . -B build/coverage -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping"
cmake --build build/coverage -j
profiles="$(mktemp -d)"
LLVM_PROFILE_FILE="$profiles/%m-%p.profraw" ctest --test-dir build/coverage --output-on-failure
xcrun llvm-profdata merge -sparse "$profiles"/*.profraw -o "$profiles/merged.profdata"
xcrun llvm-cov report build/coverage/moe_router \
  -object build/coverage/moe_router_tests -instr-profile="$profiles/merged.profdata" \
  --show-branch-summary main.cpp cpu_backend.cpp router.cpp moe.hpp
```

On Linux, use `llvm-profdata` and `llvm-cov` from the same Clang toolchain without the `xcrun` prefix. The instrumented AppleClang CPU build covered 461/470 lines (98.09%) and 164/182 branches (90.11%). It covered all 134 instrumented lines and 44 branches in `router.cpp`. The NCCL/CUDA backend requires a separate Linux multi-GPU build and run.

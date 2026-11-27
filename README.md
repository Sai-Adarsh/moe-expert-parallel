# MoE expert parallel router

Each rank owns two experts. The router picks two distinct experts per token, then combines their two-layer feed-forward outputs using normalized gate weights. Packing rows by source and destination limits each peer pair to one message per phase.

## CPU

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/moe_router --ranks 4 --tokens-per-rank 1024
```

The program generates skewed gate scores and compares strict top-two routing with a policy that penalizes already busy experts. Use `--strict-top-k` to run only strict routing. Output includes the expert histogram, peak-to-mean load ratio, remote bytes and message counts, and time spent on dispatch, expert computation, and return and combine.

## CUDA and NCCL

On Linux with CUDA, NCCL, and at least two visible NVIDIA GPUs:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMOE_ENABLE_NCCL=ON
cmake --build build-gpu
./build-gpu/moe_router --nccl --ranks 2 --tokens-per-rank 1024
```

One process assigns one GPU per rank and places each rank's weights on its device. Grouped NCCL sends and receives exchange variable-length buffers before and after the CUDA feed-forward pass. The program compares output with the CPU implementation. GPU phase times include host transfers and synchronization. Set `CMAKE_CUDA_ARCHITECTURES` when targeting a specific GPU architecture.

Both runs use synthetic activations and weights. The load penalty can change which experts are selected, so measure output quality before applying it to a pretrained gate. [NCCL point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)

# MoE expert parallel router

Compare rowwise and packed communication for a mixture-of-experts router.

## Run locally

Run from the repo root on macOS or Linux with CMake 3.24+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/moe_router
```

[Routing, benchmarks, and CUDA setup](docs/reference.md)

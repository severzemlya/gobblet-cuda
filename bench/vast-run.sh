#!/bin/bash
# Build gobblet-cuda on a fresh vast.ai instance (image nvidia/cuda:12.8.1-devel-ubuntu24.04)
# and time the GPU solver. Copy a source tarball to /workspace/gobblet-cuda.tgz first.
set -euo pipefail
cd /workspace
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
command -v cc >/dev/null || (apt-get update -qq && apt-get install -y -qq build-essential >/dev/null)
[ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q
. ~/.cargo/env
tar xzf gobblet-cuda.tgz
cd gobblet-cuda
cargo build --release --features cuda -q
B=./target/release/gobblet-cuda
echo "== machine"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv
nproc; nvcc --version | tail -1
echo "== gpu (repeated)"
for i in 1 2 3; do $B solve --backend gpu | grep -E "checksum|time"; done
echo "== gpu detail"
$B solve --backend gpu --passes

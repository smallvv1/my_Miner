#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_TORCH_ROOT="${PPU_TORCH_ROOT:-/tmp/pytorch-sail-converted}"

source /usr/local/PPU_SDK/envsetup.sh ppu >/dev/null
export MINER_PPU_RUNTIME=1
export PYTHONPATH="${PPU_TORCH_ROOT}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_PRELOAD="/tmp/libacsparse_stub.so:/tmp/libgreen_context_stub.so${LD_PRELOAD:+:${LD_PRELOAD}}"

cd "${PROJECT_DIR}"
exec "$@"

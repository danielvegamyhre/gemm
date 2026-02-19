#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_debug
rm -rf /tmp/torch_extensions_debug
python -u test.py 2>&1

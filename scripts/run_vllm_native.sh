#!/usr/bin/env bash
# Script to launch vLLM serving natively on Host / WSL2 environment

set -e

MODEL_NAME="${MODEL_NAME:-Vishva007/gemma-4-E2B-it-W4A16-AutoRound}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

echo "Starting vLLM serving for model: $MODEL_NAME"
echo "Configuration: GPU Util=$GPU_UTIL, Max Len=$MAX_MODEL_LEN, Max Num Seqs=$MAX_NUM_SEQS, Port=$PORT"

vllm serve "$MODEL_NAME" \
  --quantization gptq \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --port "$PORT" \
  --trust-remote-code


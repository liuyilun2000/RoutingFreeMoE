#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:59:00
#SBATCH --mail-type=ALL

export WANDB_API_KEY="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"
#export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"
export WANDB_SILENT="true"
#export WANDB_RUN_ID="edw0sl27"
#export WANDB_RESUME="must"

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310


#!/usr/bin/env bash
set -euo pipefail

# Benchmark baseline vs routing-free model on 1, 2, 3, and 4 GPUs.
# It calls benchmark_prefill_decode_ep.py and logs each run to a file.
#
# Usage:
#   BASELINE_HF_REPO=org/baseline-final-model \
#   RF_HF_REPO=org/rf-final-model \
#   bash benchmark_prefill_decode_ep.sh [baseline_model_ref] [rf_model_ref] [GPU_POOL] [OUT_DIR]
#
# Example:
#   bash benchmark_prefill_decode_ep.sh \
#     my-org/mixtral-baseline-final-model \
#     my-org/mixtral-rf-final-model \
#     "0,1,2,3" \
#     ./benchmark_logs
#
# Model refs can be either:
# - local path to a model directory
# - Hugging Face repo id (e.g., org/model-name)
BASELINE_MODEL_REF="${1:-${BASELINE_HF_REPO:-}}"
RF_MODEL_REF="${2:-${RF_HF_REPO:-}}"
GPU_POOL="${3:-}"
OUT_DIR="${4:-./benchmark_logs}"

if [[ -z "${BASELINE_MODEL_REF}" || -z "${RF_MODEL_REF}" ]]; then
  echo "Error: missing required args."
  echo "Usage: BASELINE_HF_REPO=<repo> RF_HF_REPO=<repo> bash benchmark_prefill_decode_ep.sh [baseline_model_ref] [rf_model_ref] [gpu_pool] [out_dir]"
  exit 1
fi

if [[ -z "${GPU_POOL}" ]]; then
  # Auto-detect all currently available CUDA device indices.
  # If user already set CUDA_VISIBLE_DEVICES, reuse that ordering/subset.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_POOL="${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_POOL="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    echo "Error: cannot auto-detect GPUs (nvidia-smi not found). Pass GPU_POOL explicitly."
    exit 1
  fi
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_POOL}"
GPU_COUNT="${#GPU_IDS[@]}"
if (( GPU_COUNT < 1 )); then
  echo "Error: no GPUs detected from GPU_POOL='${GPU_POOL}'."
  exit 1
fi

mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${OUT_DIR}/benchmark_prefill_decode_ep_${TS}.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="/home/hk-project-p0024767/hgf_mxv5488/RoutingFreeMoE/benchmark_prefill_decode_ep.py"

# Common benchmark knobs (override through environment if needed)
BATCH_SIZE="${BATCH_SIZE:-1}"
PREFILL_LENGTH="${PREFILL_LENGTH:-1024}"
DECODE_STEPS="${DECODE_STEPS:-128}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
REPEATS="${REPEATS:-10}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
RF_MODEL_TYPE="${RF_MODEL_TYPE:-routing_free_mixtral}"
BASELINE_MODEL_TYPE="${BASELINE_MODEL_TYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

echo "==== Benchmark Start $(date) ====" | tee -a "${MASTER_LOG}"
echo "BASELINE_MODEL_REF=${BASELINE_MODEL_REF}" | tee -a "${MASTER_LOG}"
echo "RF_MODEL_REF=${RF_MODEL_REF}" | tee -a "${MASTER_LOG}"
echo "GPU_POOL=${GPU_POOL}" | tee -a "${MASTER_LOG}"
echo "DETECTED_GPU_COUNT=${GPU_COUNT}" | tee -a "${MASTER_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${MASTER_LOG}"
echo "BATCH_SIZE=${BATCH_SIZE}, PREFILL_LENGTH=${PREFILL_LENGTH}, DECODE_STEPS=${DECODE_STEPS}" | tee -a "${MASTER_LOG}"
echo "WARMUP_ITERS=${WARMUP_ITERS}, REPEATS=${REPEATS}, DTYPE=${DTYPE}, SEED=${SEED}" | tee -a "${MASTER_LOG}"
echo | tee -a "${MASTER_LOG}"

run_case() {
  local ngpu="$1"
  local devices="$2"
  local case_log="${OUT_DIR}/benchmark_${ngpu}gpu_${TS}.log"

  echo "---- Running ${ngpu} GPU(s): CUDA_VISIBLE_DEVICES=${devices}" | tee -a "${MASTER_LOG}"
  echo "Log file: ${case_log}" | tee -a "${MASTER_LOG}"

  local cmd=(
    python "${PY_SCRIPT}"
    --baseline-model "${BASELINE_MODEL_REF}"
    --rf-model "${RF_MODEL_REF}"
    --baseline-model-type "${BASELINE_MODEL_TYPE}"
    --rf-model-type "${RF_MODEL_TYPE}"
    --cuda-devices "${devices}"
    --batch-size "${BATCH_SIZE}"
    --prefill-length "${PREFILL_LENGTH}"
    --decode-steps "${DECODE_STEPS}"
    --warmup-iters "${WARMUP_ITERS}"
    --repeats "${REPEATS}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
  )

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi

  "${cmd[@]}" 2>&1 | tee "${case_log}" | tee -a "${MASTER_LOG}"
  echo | tee -a "${MASTER_LOG}"
}

# Run 1/2/3/4 GPU benchmarks (skip counts not available).
for ngpu in 1 2 3 4; do
  if (( GPU_COUNT >= ngpu )); then
    devices_csv="$(printf "%s," "${GPU_IDS[@]:0:ngpu}")"
    devices_csv="${devices_csv%,}"
    run_case "${ngpu}" "${devices_csv}"
  else
    echo "Skip ${ngpu}-GPU benchmark: only ${GPU_COUNT} GPU(s) available." | tee -a "${MASTER_LOG}"
  fi
done

echo "==== Benchmark Done $(date) ====" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}"

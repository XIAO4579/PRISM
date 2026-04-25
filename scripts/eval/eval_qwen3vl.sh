# ============================================================================
# lmms-eval reference script for PRISM.
#
# Custom task overrides (the ones we modified) live in $SCRIPT_DIR/tasks/
# and are loaded via --include_path. No need to patch the upstream lmms-eval.
#
# Architecture:
#   GPU 0      -> local vLLM server acting as LLM judge (OpenAI-compatible)
#   GPU 1..7   -> data-parallel evaluation of the target model
# ============================================================================

set -o pipefail

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_TASKS_DIR="${SCRIPT_DIR}/tasks"

LMMS_EVAL_ROOT="/path/to/lmms-eval"
ENV_PATH="/path/to/lmms_eval_env"
HF_CACHE="/path/to/hf_cache"
OUTPUT_PATH="${LMMS_EVAL_ROOT}/log"

MODEL="/path/to/models/Qwen3-VL-4B-Instruct"
JUDGE_MODEL="/path/to/models/Qwen2.5-32B-Instruct"

# ---- runtime env ----
cd "${LMMS_EVAL_ROOT}"
source "${ENV_PATH}/bin/activate"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000

# ============================================================================
# Judge server (GPU 0)
# ============================================================================
JUDGE_PORT=8000
JUDGE_HOST="0.0.0.0"
JUDGE_GPU="0"
JUDGE_LOG="${OUTPUT_PATH}/judge_server_${SLURM_JOB_ID:-local}.log"

mkdir -p "${OUTPUT_PATH}"

echo "Starting vLLM judge server on GPU ${JUDGE_GPU} (${JUDGE_MODEL})"
CUDA_VISIBLE_DEVICES=${JUDGE_GPU} vllm serve "${JUDGE_MODEL}" \
    --port ${JUDGE_PORT} \
    --host ${JUDGE_HOST} \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!
echo "Judge server PID: ${JUDGE_PID} (log: ${JUDGE_LOG})"

cleanup() {
    echo "Stopping judge server (PID: ${JUDGE_PID})..."
    kill ${JUDGE_PID} 2>/dev/null
    wait ${JUDGE_PID} 2>/dev/null
}
trap cleanup EXIT

echo "Waiting for judge server to be ready..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:${JUDGE_PORT}/health" > /dev/null 2>&1; then
        echo "Judge server ready after ${WAITED}s."
        break
    fi
    if ! kill -0 ${JUDGE_PID} 2>/dev/null; then
        echo "ERROR: judge server died before becoming ready."
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: judge server failed to start within ${MAX_WAIT}s"
    exit 1
fi

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_URL="http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions"
export OPENAI_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export JUDGE_API_KEY="EMPTY"
export JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export JUDGE_MODEL_NAME="${JUDGE_MODEL}"
export MODEL_VERSION="${JUDGE_MODEL}"
export USE_LLM_JUDGE="True"

# ============================================================================
# Evaluation (GPUs 1..7)
# ============================================================================
EVAL_GPUS="1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=${EVAL_GPUS}

TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=7
GPU_MEMORY_UTILIZATION=0.75
BATCH_SIZE=16

TASKS="mathvista_testmini,mathvision_testmini,mathverse_testmini,wemath_testmini_reasoning,mmmu_val,mmmu_pro,hallusion_bench_image"

CHAT_TEMPLATE="${SCRIPT_DIR}/chat_template/qwen3vl_bridge_eval.jinja"
LOG_SUFFIX="qwen3vl_vllm"

MODEL_ARGS="model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},chat_template=${CHAT_TEMPLATE}"

if [ "${DATA_PARALLEL_SIZE}" -gt 1 ]; then
    LAUNCHER="python -m torch.distributed.run --standalone --nproc_per_node=$((TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE)) -m lmms_eval"
    MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${DATA_PARALLEL_SIZE}"
else
    LAUNCHER="python -m lmms_eval"
fi

echo "=========================================="
echo "Model:             ${MODEL}"
echo "Custom tasks dir:  ${CUSTOM_TASKS_DIR}"
echo "Tasks:             ${TASKS}"
echo "TP / DP:           ${TENSOR_PARALLEL_SIZE} / ${DATA_PARALLEL_SIZE}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Output path:       ${OUTPUT_PATH}"
echo "=========================================="

${LAUNCHER} \
    --model vllm \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --include_path "${CUSTOM_TASKS_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --use_cache "${OUTPUT_PATH}/qwen3vl_prism.db" \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --gen_kwargs temperature=1.0,top_p=0.7,top_k=-1 \
    --log_samples_suffix "${LOG_SUFFIX}"

echo "Evaluation complete. Results saved to: ${OUTPUT_PATH}"

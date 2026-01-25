#!/bin/bash

# 评估脚本：使用vLLM测试Qwen2.5-7B-Instruct在AIME24、AIME25和MMLUPro上的准确度

# 设置默认参数
MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
DATASETS=${2:-"aime24,aime25,mmlupro"}
OUTPUT_DIR=${3:-"./results"}
NUM_SAMPLES=${4:-""}
TENSOR_PARALLEL_SIZE=${5:-1}
GPU_MEMORY_UTILIZATION=${6:-0.9}

# 如果NUM_SAMPLES为空，不传递该参数
if [ -z "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_ARG=""
else
    NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi

echo "=========================================="
echo "Evaluation Configuration:"
echo "  Model: $MODEL"
echo "  Datasets: $DATASETS"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Number of Samples: ${NUM_SAMPLES:-"All"}"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "=========================================="

# 运行评估
python evaluate_with_judge.py \
    --model "$MODEL" \
    --datasets "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    $NUM_SAMPLES_ARG \
    --use_judge \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION

echo "Evaluation completed!"

#!/usr/bin/env python3
"""
使用示例：展示如何使用评估脚本
"""

import subprocess
import sys

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("基本使用示例")
    print("=" * 60)
    
    cmd = [
        "python", "evaluate_with_judge.py",
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--datasets", "aime24",
        "--output_dir", "./results",
        "--num_samples", "10",
        "--use_judge"
    ]
    
    print("运行命令:")
    print(" ".join(cmd))
    print("\n或者使用shell脚本:")
    print("bash run_eval.sh Qwen/Qwen2.5-7B-Instruct aime24 ./results 10")


def example_multi_dataset():
    """多数据集评估示例"""
    print("\n" + "=" * 60)
    print("多数据集评估示例")
    print("=" * 60)
    
    cmd = [
        "python", "evaluate_with_judge.py",
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--datasets", "aime24,aime25,mmlupro",
        "--output_dir", "./results",
        "--tensor_parallel_size", "2",
        "--gpu_memory_utilization", "0.85"
    ]
    
    print("运行命令:")
    print(" ".join(cmd))
    print("\n或者使用shell脚本:")
    print("bash run_eval.sh Qwen/Qwen2.5-7B-Instruct aime24,aime25,mmlupro ./results")


def example_custom_judge():
    """使用自定义judge模型示例"""
    print("\n" + "=" * 60)
    print("使用自定义judge模型示例")
    print("=" * 60)
    
    cmd = [
        "python", "evaluate_with_judge.py",
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--datasets", "aime24",
        "--output_dir", "./results",
        "--judge_model", "Qwen/Qwen2.5-14B-Instruct",  # 使用更大的模型作为judge
        "--use_judge"
    ]
    
    print("运行命令:")
    print(" ".join(cmd))


if __name__ == "__main__":
    example_basic_usage()
    example_multi_dataset()
    example_custom_judge()
    
    print("\n" + "=" * 60)
    print("注意事项:")
    print("=" * 60)
    print("1. 确保已安装所有依赖: pip install -r requirements.txt")
    print("2. 确保有足够的GPU内存")
    print("3. 数据集文件应位于 ../math_eval/eval/data/{dataset}/test.jsonl")
    print("4. 如果使用多GPU，设置tensor_parallel_size参数")
    print("=" * 60)

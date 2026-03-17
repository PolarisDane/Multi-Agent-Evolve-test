# vLLM Evaluation Scripts

这个文件夹包含使用vLLM评估模型在AIME24、AIME25和MMLUPro数据集上的脚本。

## 功能特性

1. **使用Solver Prompt生成回答**：使用指定的Solver prompt引导模型生成高质量回答
2. **String Match评估**：使用字符串匹配判断答案是否正确
3. **Judge评估**：使用judge prompt（输出true/false）判断答案正确性
4. **结果对比**：对比string match和judge两种方法的准确率

## 文件说明

- `evaluate_with_judge.py`: 主评估脚本
- `run_eval.sh`: 便捷运行脚本
- `README.md`: 本文件

## 使用方法

### 基本使用

```bash
# 使用默认参数（Qwen2.5-7B-Instruct，评估所有数据集）
bash run_eval.sh

# 指定模型和数据集
bash run_eval.sh "Qwen/Qwen2.5-7B-Instruct" "aime24,aime25,mmlupro" "./results"

# 限制样本数量（例如只评估前10个样本）
bash run_eval.sh "Qwen/Qwen2.5-7B-Instruct" "aime24" "./results" 10
```

### 直接使用Python脚本

```bash
python evaluate_with_judge.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --datasets "aime24,aime25,mmlupro" \
    --output_dir "./results" \
    --num_samples 100 \
    --use_judge \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9
```

### 参数说明

- `--model`: 模型名称或路径（默认：Qwen/Qwen2.5-7B-Instruct）
- `--datasets`: 要评估的数据集，逗号分隔（默认：aime24,aime25,mmlupro）
- `--output_dir`: 结果输出目录（默认：./results）
- `--num_samples`: 评估的样本数量（None表示全部）
- `--use_judge`: 是否使用judge prompt进行评估
- `--judge_model`: Judge模型（如果与主模型不同）
- `--tensor_parallel_size`: vLLM的tensor parallel大小（默认：1）
- `--gpu_memory_utilization`: GPU内存利用率（默认：0.9）

## 输出格式

评估完成后，会在输出目录生成以下文件：

1. `{dataset}_results.jsonl`: 每个数据集的详细结果
   - 包含每个样本的问题、ground truth、预测答案、完整响应
   - 包含string match和judge的判断结果

2. `summary.json`: 总结文件
   - 包含所有数据集的准确率统计

## Solver Prompt

脚本使用以下Solver prompt：

```
## Task: Generate a High-Quality Response to a Given Task

You will be given a cognitive, creative, logical、mathematical、or planning-related task. Your job is to generate a complete, high-quality response that satisfies the task's constraints and demonstrates clear, structured reasoning or creativity.

### Instructions:
- Carefully read and understand the task.
- Think step by step — break down the task, simulate it mentally if needed, and reason through constraints.
- Then directly write your final response inside a pair of <answer></answer> tags(no need to restate or reformat the task).
- Your output should:
  * Be **correct** or **plausibly optimal**, given the task
  * **Fulfill all constraints** in the task
  * Be **clear** and **structured**
  * Avoid any vagueness or randomness

### Good Response Traits:
- For reasoning tasks: shows logical progression and result
- For generation tasks: respects the given constraints
- For math/logic/planning: includes a final answer that could be evaluated
- For creative tasks: coherent and original
```

## Judge Prompt

脚本使用修改后的judge prompt，输出true/false而不是1-10分：

```
Please evaluate the following solution to a question/problem following a strict rubric.

Question/Problem: {question}

Generated Solution: {answer}

First, analyze the solution in the <think> and </think> tags below:

<think>
STRICT RUBRIC:
- Factual correctness is mandatory. ANY factual error (including arithmetic mistakes, violated common sense, wrong units, invalid assumptions, or faulty reasoning steps) => final judgment MUST be false.
- Meaningless or excessive repetition, rambling, or filler content => final judgment MUST be false.
- Hallucinated references, fabricated data, unsupported claims, or contradictions => final judgment MUST be false.
- Completeness & directness: The answer must fully address the asked question. Missing key steps, incomplete derivations, or only partially addressing the prompt => judgment should be false unless the answer is entirely correct.

[Write your detailed analysis here: identify any factual errors, faulty reasoning steps or repetition issues. If ANY such issue exists, note that the judgment must be false. If none, justify why it qualifies as correct.]
</think>

Then provide your judgment between <judgment> and </judgment> tags:
- true: The solution is entirely correct, follows instructions, is concise (no redundant repetition), and fully answers the question.
- false: ANY factual/logic/common-sense/calculation error OR hallucination OR large meaningless repetition OR severe irrelevance OR incomplete answer.

<judgment>X</judgment> (where X is either "true" or "false")
```

## 评估方法

### String Match评估

- **AIME24/AIME25**: 
  - 使用temperature=0.6，每个问题采样64次
  - 计算Mean@64：所有rollout的平均准确率
  - 对于每个问题，计算64次采样中正确的比例
  - 最终准确率 = 所有问题的平均准确率（所有rollout中正确的比例的平均值）
  - 提取答案中的数字，与ground truth进行比较
- **MMLUPro**: 提取答案中的选项字母（A-J），与ground truth进行比较

### Judge评估

- 使用judge prompt让模型分析答案的正确性
- 从响应中提取`<judgment>true</judgment>`或`<judgment>false</judgment>`
- 如果没有找到标签，根据响应中的关键词判断
- **Judge正确性定义**：judge的判断与string match的判断一致才算正确
  - 即：如果string match认为正确，judge也认为正确 → judge正确
  - 如果string match认为错误，judge也认为错误 → judge正确
  - 如果两者不一致 → judge错误

## 注意事项

1. 确保已安装vLLM和相关依赖
2. 确保有足够的GPU内存
3. 数据集文件应位于 `../math_eval/eval/data/{dataset}/test.jsonl`
4. 如果模型支持chat template，脚本会自动使用

## 示例输出

```
============================================================
Evaluating dataset: aime24
============================================================
Loaded 100 examples
Generating responses (temperature=0.6, n_sampling=64)...
Processing results...

Results for aime24:
  Sampling: 64 times with temperature=0.6
  String Match Accuracy: 45.00% (45/100)
  Judge Accuracy (agreement with string match): 42.00% (42/100)
Results saved to ./results/aime24_results.jsonl

============================================================
SUMMARY
============================================================
Dataset         String Match    Judge          
------------------------------------------------------------
aime24           45.00%         42.00%
aime25           38.00%         35.00%
mmlupro          52.00%         50.00%
```

**注意**：
- AIME24和AIME25数据集使用temperature=0.6，每个问题采样64次
- Judge准确率表示judge判断与string match判断一致的比例

## 分析Judge一致性

运行评估后，可以使用 `analyze_judge_consistency.py` 分析judge的一致性：

```bash
# 分析结果文件中的judge一致性（自动查找所有数据集）
python analyze_judge_consistency.py --results_dir ./results

# 或指定具体文件路径
python analyze_judge_consistency.py \
    --aime24_file ./results/aime24_results.jsonl \
    --aime25_file ./results/aime25_results.jsonl \
    --mmlupro_file ./results/mmlupro_results.jsonl
```

该脚本会统计：
- **当string match正确时**：judge也判断为正确的比例（一致性）
- **当string match错误时**：judge也判断为错误的比例（一致性）
- **总体一致性**：所有rollout中judge与string match一致的比例

输出示例：
```
Statistics for aime24:
  Total questions: 100
  Total rollouts: 6400
  String match correct rollouts: 2880 (45.00%)
  String match wrong rollouts: 3520 (55.00%)

  When string match is CORRECT (2880 rollouts):
    Judge agrees (also judges as correct): 85.42% (2460/2880)
    Judge disagrees (judges as wrong): 14.58% (420/2880)

  When string match is WRONG (3520 rollouts):
    Judge agrees (also judges as wrong): 72.16% (2540/3520)
    Judge disagrees (judges as correct): 27.84% (980/3520)

  Overall judge agreement rate: 78.13% (5000/6400)
```

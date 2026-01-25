#!/usr/bin/env python3
"""
评估脚本：使用vLLM测试模型在AIME24、AIME25和MMLUPro上的准确度
- 使用Solver prompt生成回答
- 使用string match判断正确率
- 使用judge prompt判断答案正确性（true/false）
- 对比两种方法的准确率
"""

import os
import json
import argparse
import re
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys

# 添加math_eval路径以便导入工具函数
current_dir = os.path.dirname(os.path.abspath(__file__))
math_eval_path = os.path.join(current_dir, '..', 'math_eval', 'eval')
if os.path.exists(math_eval_path):
    sys.path.insert(0, math_eval_path)
    try:
        from utils import load_jsonl, save_jsonl
    except ImportError:
        # 如果导入失败，定义本地版本
        def load_jsonl(file):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except:
                        print("Error in loading:", line)
                        continue
        
        def save_jsonl(samples, save_path):
            folder = os.path.dirname(save_path)
            os.makedirs(folder, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print("Saved to", save_path)
else:
    # 定义本地版本
    def load_jsonl(file):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except:
                    print("Error in loading:", line)
                    continue
    
    def save_jsonl(samples, save_path):
        folder = os.path.dirname(save_path)
        os.makedirs(folder, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print("Saved to", save_path)


# Solver prompt
SOLVER_PROMPT = """## Task: Generate a High-Quality Response to a Given Task

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
- For creative tasks: coherent and original"""

# Judge prompt (修改为true/false输出)
JUDGE_PROMPT_TEMPLATE = """Please evaluate the following solution to a question/problem following a strict rubric.

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

<judgment>X</judgment> (where X is either "true" or "false")"""


def load_dataset(data_name: str, data_dir: str = None) -> List[Dict]:
    """加载数据集"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'math_eval', 'eval', 'data')
    
    data_file = os.path.join(data_dir, data_name, 'test.jsonl')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    examples = list(load_jsonl(data_file))
    
    # 添加索引
    for i, example in enumerate(examples):
        if 'idx' not in example:
            example['idx'] = i
    
    return examples


def extract_answer_from_response(response: str) -> str:
    """从模型响应中提取答案（在<answer></answer>标签中）"""
    # 尝试提取<answer>标签中的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        # 清理答案中的额外标签
        answer = re.sub(r'</?answer>', '', answer, flags=re.IGNORECASE)
        if answer:
            return answer
    
    # 如果没有找到标签，尝试提取\boxed{}中的内容
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 尝试提取\framebox{}中的内容
    framebox_match = re.search(r'\\framebox\{([^}]+)\}', response)
    if framebox_match:
        return framebox_match.group(1).strip()
    
    # 如果都没有，尝试提取最后一行作为答案
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # 如果最后一行看起来像答案（包含数字或字母）
        if re.search(r'[A-J]|\d+', last_line):
            return last_line
    
    # 如果都没有，返回整个响应（去除前后空白）
    return response.strip()


def string_match_evaluation(predicted: str, ground_truth: str, data_name: str) -> bool:
    """使用string match判断答案是否正确"""
    # 清理答案
    def clean_answer(ans: str) -> str:
        ans = ans.strip()
        # 移除常见的LaTeX格式
        ans = re.sub(r'\$', '', ans)
        ans = re.sub(r'\\boxed\{([^}]+)\}', r'\1', ans)
        ans = re.sub(r'\\framebox\{([^}]+)\}', r'\1', ans)
        ans = ans.strip()
        return ans
    
    pred_clean = clean_answer(predicted)
    gt_clean = clean_answer(ground_truth)
    
    # 对于MMLUPro，答案通常是单个字母
    if data_name == 'mmlupro':
        # 提取第一个字母
        pred_letter = re.search(r'\b([A-J])\b', pred_clean.upper())
        gt_letter = gt_clean.upper().strip()
        if pred_letter:
            return pred_letter.group(1) == gt_letter
        return False
    
    # 对于AIME，答案通常是数字
    # 直接比较清理后的字符串
    if pred_clean == gt_clean:
        return True
    
    # 尝试提取数字进行比较
    pred_numbers = re.findall(r'\d+', pred_clean)
    gt_numbers = re.findall(r'\d+', gt_clean)
    
    if pred_numbers and gt_numbers:
        return pred_numbers[-1] == gt_numbers[-1]  # 比较最后一个数字
    
    return False


def judge_evaluation(llm, tokenizer, question: str, answer: str, judge_model: str = None) -> bool:
    """使用judge prompt判断答案是否正确"""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    
    # 如果judge_model与主模型不同，需要单独加载
    # 这里假设使用同一个模型作为judge
    if tokenizer:
        # 使用chat template
        messages = [{"role": "user", "content": judge_prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 如果chat template失败，使用原始prompt
            formatted_prompt = judge_prompt
    else:
        formatted_prompt = judge_prompt
    
    # 生成judge响应
    try:
        outputs = llm.generate(
            [formatted_prompt],
            SamplingParams(
                temperature=0.0,
                max_tokens=512,
                stop=["</judgment>", "\n\nQuestion:", "<|im_end|>", "</s>", "<|im_start|>"]
            ),
            use_tqdm=False
        )
        
        judge_response = outputs[0].outputs[0].text
        
        # 提取judgment
        judgment_match = re.search(r'<judgment>\s*(true|false)\s*</judgment>', judge_response, re.IGNORECASE | re.DOTALL)
        if judgment_match:
            judgment = judgment_match.group(1).lower().strip()
            return judgment == 'true'
        
        # 如果没有找到标签，尝试查找true/false关键词（更严格的匹配）
        # 查找最后一个明确的true/false判断
        response_lower = judge_response.lower()
        # 检查是否有明确的true/false判断
        if re.search(r'\btrue\b', response_lower) and not re.search(r'\bfalse\b', response_lower):
            return True
        if re.search(r'\bfalse\b', response_lower):
            return False
        
        # 默认返回False
        return False
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        return False


def evaluate_dataset(
    llm,
    tokenizer,
    data_name: str,
    solver_prompt: str,
    output_dir: str,
    num_samples: int = None,
    use_judge: bool = True,
    judge_model: str = None
) -> Dict[str, Any]:
    """评估单个数据集"""
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {data_name}")
    print(f"{'='*60}")
    
    # 加载数据
    examples = load_dataset(data_name)
    if num_samples:
        examples = examples[:num_samples]
    
    print(f"Loaded {len(examples)} examples")
    
    # 准备prompts
    prompts = []
    for example in examples:
        # 构建问题文本
        if data_name == 'mmlupro':
            question = example.get('question', '')
        else:
            question = example.get('question', example.get('problem', ''))
        
        # 构建完整prompt
        full_prompt = f"{solver_prompt}\n\nTask:\n{question}"
        
        if tokenizer:
            # 使用chat template
            messages = [{"role": "user", "content": full_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = full_prompt
        
        prompts.append(formatted_prompt)
    
    # 判断是否为AIME数据集，需要采样多次
    is_aime = data_name in ['aime24', 'aime25']
    temperature = 0.6 if is_aime else 0.0
    n_sampling = 64 if is_aime else 1
    
    # 生成回答
    print(f"Generating responses (temperature={temperature}, n_sampling={n_sampling})...")
    
    # 对于AIME数据集，每个问题需要采样64次
    if is_aime:
        # 重复每个prompt 64次
        repeated_prompts = []
        prompt_indices = []  # 记录每个prompt的原始索引
        for i, prompt in enumerate(prompts):
            for _ in range(n_sampling):
                repeated_prompts.append(prompt)
                prompt_indices.append(i)
        
        outputs = llm.generate(
            repeated_prompts,
            SamplingParams(
                temperature=temperature,
                max_tokens=2048,
                stop=["</answer>", "\n\nTask:", "<|im_end|>", "</s>", "<|im_start|>"],
                n=1
            ),
            use_tqdm=True
        )
        
        # 按原始索引分组
        grouped_outputs = {}
        for idx, output in zip(prompt_indices, outputs):
            if idx not in grouped_outputs:
                grouped_outputs[idx] = []
            grouped_outputs[idx].append(output.outputs[0].text)
    else:
        # 非AIME数据集，正常生成
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=temperature,
                max_tokens=2048,
                stop=["</answer>", "\n\nTask:", "<|im_end|>", "</s>", "<|im_start|>"],
                n=1
            ),
            use_tqdm=True
        )
        grouped_outputs = {i: [output.outputs[0].text] for i, output in enumerate(outputs)}
    
    # 处理结果
    results = []
    # 对于AIME数据集，使用累加准确率；对于其他数据集，使用累加正确数
    string_match_correct = 0.0 if is_aime else 0
    judge_correct = 0.0 if is_aime else 0
    total = len(examples)
    
    print("Processing results...")
    for i in tqdm(range(total), total=total):
        example = examples[i]
        responses = grouped_outputs[i]
        
        # 获取ground truth
        if data_name == 'mmlupro':
            ground_truth = example.get('answer', '').strip()
            question = example.get('question', '')
        else:
            ground_truth = example.get('answer', '').strip()
            question = example.get('question', example.get('problem', ''))
        
        # 对于AIME数据集，计算mean@64（所有rollout的平均准确率）
        if is_aime:
            string_match_results = []
            judge_results = []
            predicted_answers = []
            
            for response in responses:
                # 提取答案
                predicted_answer = extract_answer_from_response(response)
                predicted_answers.append(predicted_answer)
                
                # String match评估
                string_match_result = string_match_evaluation(predicted_answer, ground_truth, data_name)
                string_match_results.append(string_match_result)
                
                # Judge评估
                if use_judge:
                    try:
                        judge_result = judge_evaluation(llm, tokenizer, question, response, judge_model)
                        judge_results.append(judge_result)
                    except Exception as e:
                        print(f"Error in judge evaluation for example {i}: {e}")
                        judge_results.append(False)
            
            # 计算该题的准确率（64次中正确的比例）
            question_string_match_acc = sum(string_match_results) / len(string_match_results)
            question_judge_acc = sum(judge_results) / len(judge_results) if use_judge else None
            
            # 累加准确率（用于计算所有题目的平均）
            string_match_correct += question_string_match_acc
            if use_judge and question_judge_acc is not None:
                # Judge正确性：judge的判断和string match的判断一致
                # 对于每个rollout，判断judge和string match是否一致
                judge_agreement_count = sum(
                    judge_results[j] == string_match_results[j] 
                    for j in range(len(string_match_results))
                )
                judge_correct += judge_agreement_count / len(string_match_results)
            
            # 保存结果（保存第一个响应作为代表）
            result = {
                'idx': example.get('idx', i),
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answers[0],  # 保存第一个预测答案
                'all_predicted_answers': predicted_answers,  # 保存所有64个答案
                'full_response': responses[0],  # 保存第一个完整响应
                'all_responses': responses,  # 保存所有64个响应
                'string_match_accuracy': question_string_match_acc,  # 该题的准确率（64次中正确的比例）
                'judge_accuracy': question_judge_acc if use_judge else None,  # 该题的judge准确率
                'n_sampling': n_sampling,
                'n_correct_samples': sum(string_match_results)  # 64次中正确的次数
            }
        else:
            # 非AIME数据集，单次生成
            response = responses[0]
            
            # 提取答案
            predicted_answer = extract_answer_from_response(response)
            
            # String match评估
            string_match_result = string_match_evaluation(predicted_answer, ground_truth, data_name)
            if string_match_result:
                string_match_correct += 1
            
            # Judge评估
            judge_result = False
            if use_judge:
                try:
                    judge_result = judge_evaluation(llm, tokenizer, question, response, judge_model)
                except Exception as e:
                    print(f"Error in judge evaluation for example {i}: {e}")
                    judge_result = False
            
            # Judge正确性：judge的判断和string match的判断一致
            if use_judge:
                if judge_result == string_match_result:
                    judge_correct += 1
            
            # 保存结果
            result = {
                'idx': example.get('idx', i),
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'full_response': response,
                'string_match_correct': string_match_result,
                'judge_correct': (judge_result == string_match_result) if use_judge else None,
                'judge_result': judge_result if use_judge else None
            }
        
        results.append(result)
    
    # 计算准确率
    if is_aime:
        # mean@64: 所有rollout的平均准确率
        string_match_acc = (string_match_correct / total) * 100
        judge_acc = (judge_correct / total) * 100 if use_judge else None
    else:
        # 普通准确率
        string_match_acc = string_match_correct / total * 100
        judge_acc = judge_correct / total * 100 if use_judge else None
    
    print(f"\nResults for {data_name}:")
    if is_aime:
        print(f"  Sampling: {n_sampling} times with temperature={temperature}")
        print(f"  Evaluation: Mean@64 (average accuracy across all {n_sampling} rollouts)")
        print(f"  String Match Accuracy (Mean@64): {string_match_acc:.2f}%")
        if use_judge:
            print(f"  Judge Accuracy (Mean@64, agreement with string match): {judge_acc:.2f}%")
    else:
        print(f"  String Match Accuracy: {string_match_acc:.2f}% ({string_match_correct}/{total})")
        if use_judge:
            print(f"  Judge Accuracy (agreement with string match): {judge_acc:.2f}% ({judge_correct}/{total})")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{data_name}_results.jsonl")
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_jsonl(results, output_file)
    print(f"Results saved to {output_file}")
    
    return {
        'dataset': data_name,
        'total': total,
        'string_match_correct': string_match_correct,
        'string_match_accuracy': string_match_acc,
        'judge_correct': judge_correct if use_judge else None,
        'judge_accuracy': judge_acc if use_judge else None,
        'temperature': temperature,
        'n_sampling': n_sampling if is_aime else 1
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on AIME24, AIME25, and MMLUPro')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Model name or path')
    parser.add_argument('--datasets', type=str, default='aime24,aime25,mmlupro',
                       help='Comma-separated list of datasets to evaluate')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--use_judge', action='store_true', default=True,
                       help='Use judge prompt for evaluation')
    parser.add_argument('--judge_model', type=str, default=None,
                       help='Judge model (if different from main model)')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU memory utilization')
    
    args = parser.parse_args()
    
    # 初始化模型
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=32768,
    )
    
    # 加载tokenizer（如果需要chat template）
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
    
    # 评估每个数据集
    datasets = args.datasets.split(',')
    all_results = []
    
    for dataset in datasets:
        dataset = dataset.strip()
        try:
            result = evaluate_dataset(
                llm=llm,
                tokenizer=tokenizer,
                data_name=dataset,
                solver_prompt=SOLVER_PROMPT,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                use_judge=args.use_judge,
                judge_model=args.judge_model or args.model
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} {'String Match':<15} {'Judge':<15}")
    print("-"*60)
    for result in all_results:
        dataset = result['dataset']
        sm_acc = result['string_match_accuracy']
        judge_acc = result.get('judge_accuracy', 'N/A')
        if isinstance(judge_acc, float):
            print(f"{dataset:<15} {sm_acc:>6.2f}%       {judge_acc:>6.2f}%")
        else:
            print(f"{dataset:<15} {sm_acc:>6.2f}%       {judge_acc}")
    
    # 保存总结
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == '__main__':
    main()

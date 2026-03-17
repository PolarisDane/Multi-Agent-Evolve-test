#!/usr/bin/env python3
"""
分析judge一致性：统计在string match正确和错误的情况下，judge判断的一致性概率
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


def load_results(file_path: str) -> List[Dict]:
    """加载结果文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_judge_consistency(results: List[Dict], dataset_name: str) -> Dict:
    """
    分析judge一致性
    
    对于AIME数据集，每个问题有64个rollout，需要分别统计：
    1. 当string match正确时，judge也判断为正确的比例
    2. 当string match错误时，judge也判断为错误的比例
    """
    stats = {
        'dataset': dataset_name,
        'total_questions': len(results),
        'total_rollouts': 0,
        'string_match_correct_rollouts': 0,
        'string_match_wrong_rollouts': 0,
        'judge_agree_when_correct': 0,  # string match正确时，judge也判断正确
        'judge_agree_when_wrong': 0,     # string match错误时，judge也判断错误
        'judge_disagree_when_correct': 0,  # string match正确时，judge判断错误
        'judge_disagree_when_wrong': 0,    # string match错误时，judge判断正确
    }
    
    for result in results:
        # 检查是否有多个rollout（AIME数据集）
        if 'all_responses' in result and 'all_predicted_answers' in result:
            # AIME数据集：有64个rollout
            n_sampling = result.get('n_sampling', 64)
            all_responses = result.get('all_responses', [])
            all_predicted_answers = result.get('all_predicted_answers', [])
            ground_truth = result.get('ground_truth', '')
            question = result.get('question', '')
            data_name = 'aime24' if 'aime24' in dataset_name.lower() else 'aime25'
            
            # 需要重新评估每个rollout的string match和judge
            # 这里我们需要重新计算，因为结果文件中可能没有保存每个rollout的详细判断
            # 或者我们可以从string_match_accuracy和judge_accuracy推断
            
            # 从accuracy推断：string_match_accuracy * n_sampling = 正确的rollout数
            string_match_acc = result.get('string_match_accuracy', 0.0)
            judge_acc = result.get('judge_accuracy', 0.0)
            n_correct_samples = result.get('n_correct_samples', 0)
            
            n_correct = int(round(string_match_acc * n_sampling))
            n_wrong = n_sampling - n_correct
            
            stats['total_rollouts'] += n_sampling
            stats['string_match_correct_rollouts'] += n_correct
            stats['string_match_wrong_rollouts'] += n_wrong
            
            # 如果judge_accuracy存在，说明judge也评估了
            if judge_acc is not None and 'judge_accuracy' in result:
                # judge_accuracy表示judge判断为正确的比例
                # 但我们不知道judge在正确和错误情况下的分布
                # 需要从结果中推断或重新计算
                # 这里我们假设judge_accuracy是整体准确率
                # 实际上我们需要知道：当string match正确时judge的正确率，和当string match错误时judge的错误率
                pass
        else:
            # 非AIME数据集：单个rollout
            string_match_correct = result.get('string_match_correct', False)
            judge_result = result.get('judge_result', None)
            judge_correct = result.get('judge_correct', None)
            
            stats['total_rollouts'] += 1
            
            if string_match_correct:
                stats['string_match_correct_rollouts'] += 1
                if judge_result is True:
                    stats['judge_agree_when_correct'] += 1
                elif judge_result is False:
                    stats['judge_disagree_when_correct'] += 1
            else:
                stats['string_match_wrong_rollouts'] += 1
                if judge_result is False:
                    stats['judge_agree_when_wrong'] += 1
                elif judge_result is True:
                    stats['judge_disagree_when_wrong'] += 1
    
    return stats


def analyze_judge_consistency_detailed(results: List[Dict], dataset_name: str, 
                                      string_match_eval_func=None, judge_eval_func=None) -> Dict:
    """
    详细分析judge一致性（需要重新评估每个rollout）
    对于AIME数据集，需要重新评估每个rollout的string match和judge结果
    """
    stats = {
        'dataset': dataset_name,
        'total_questions': len(results),
        'total_rollouts': 0,
        'string_match_correct_rollouts': 0,
        'string_match_wrong_rollouts': 0,
        'judge_agree_when_correct': 0,  # string match正确时，judge也判断正确
        'judge_agree_when_wrong': 0,     # string match错误时，judge也判断错误
        'judge_disagree_when_correct': 0,  # string match正确时，judge判断错误
        'judge_disagree_when_wrong': 0,    # string match错误时，judge判断正确
    }
    
    for result in results:
        # 检查是否有多个rollout（AIME数据集）
        if 'all_responses' in result and 'all_predicted_answers' in result:
            all_responses = result.get('all_responses', [])
            all_predicted_answers = result.get('all_predicted_answers', [])
            ground_truth = result.get('ground_truth', '')
            question = result.get('question', '')
            data_name = 'aime24' if 'aime24' in dataset_name.lower() else 'aime25'
            
            n_sampling = len(all_responses)
            stats['total_rollouts'] += n_sampling
            
            # 评估每个rollout
            for predicted_answer, response in zip(all_predicted_answers, all_responses):
                # String match评估
                if string_match_eval_func:
                    string_match_correct = string_match_eval_func(predicted_answer, ground_truth, data_name)
                else:
                    # 简单的字符串匹配
                    string_match_correct = (predicted_answer.strip() == ground_truth.strip())
                
                if string_match_correct:
                    stats['string_match_correct_rollouts'] += 1
                else:
                    stats['string_match_wrong_rollouts'] += 1
                
                # Judge评估（如果有judge结果）
                # 注意：这里我们需要judge对每个rollout的判断
                # 如果结果文件中没有保存每个rollout的judge结果，我们需要重新计算
                # 但重新计算需要模型，所以这里我们先尝试从结果中读取
                pass  # 暂时跳过，因为需要模型来重新评估
        else:
            # 单个rollout的情况
            string_match_correct = result.get('string_match_correct', False)
            judge_result = result.get('judge_result', None)
            
            stats['total_rollouts'] += 1
            
            if string_match_correct:
                stats['string_match_correct_rollouts'] += 1
                if judge_result is True:
                    stats['judge_agree_when_correct'] += 1
                elif judge_result is False:
                    stats['judge_disagree_when_correct'] += 1
            else:
                stats['string_match_wrong_rollouts'] += 1
                if judge_result is False:
                    stats['judge_agree_when_wrong'] += 1
                elif judge_result is True:
                    stats['judge_disagree_when_wrong'] += 1
    
    return stats


def analyze_from_aggregated_stats(results: List[Dict], dataset_name: str) -> Dict:
    """
    从结果文件中分析judge一致性
    如果结果文件中保存了每个rollout的judge结果，直接使用
    否则需要重新评估
    """
    stats = {
        'dataset': dataset_name,
        'total_questions': len(results),
        'total_rollouts': 0,
        'string_match_correct_rollouts': 0,
        'string_match_wrong_rollouts': 0,
        'judge_agree_when_correct': 0,
        'judge_agree_when_wrong': 0,
        'judge_disagree_when_correct': 0,
        'judge_disagree_when_wrong': 0,
    }
    
    has_judge_results = False
    missing_judge_count = 0
    
    for result in results:
        # 检查是否有保存每个rollout的结果
        if 'all_string_match_results' in result and 'all_judge_results' in result:
            # 有保存每个rollout的结果，直接使用
            string_match_results = result.get('all_string_match_results', [])
            judge_results = result.get('all_judge_results', [])
            
            if judge_results is None:
                missing_judge_count += 1
                continue
            
            has_judge_results = True
            n_sampling = len(string_match_results)
            stats['total_rollouts'] += n_sampling
            
            # 统计每个rollout
            for string_match_correct, judge_result in zip(string_match_results, judge_results):
                if string_match_correct:
                    stats['string_match_correct_rollouts'] += 1
                    if judge_result is True:
                        stats['judge_agree_when_correct'] += 1
                    elif judge_result is False:
                        stats['judge_disagree_when_correct'] += 1
                else:
                    stats['string_match_wrong_rollouts'] += 1
                    if judge_result is False:
                        stats['judge_agree_when_wrong'] += 1
                    elif judge_result is True:
                        stats['judge_disagree_when_wrong'] += 1
        elif 'all_responses' in result and 'all_predicted_answers' in result:
            # AIME数据集：有多个rollout，检查是否有judge结果
            if 'all_judge_results' not in result:
                missing_judge_count += 1
                # 仍然统计string match的结果
                all_predicted_answers = result.get('all_predicted_answers', [])
                n_sampling = len(all_predicted_answers)
                stats['total_rollouts'] += n_sampling
                
                # 从string_match_accuracy推断
                string_match_acc = result.get('string_match_accuracy', 0.0)
                n_correct = int(round(string_match_acc * n_sampling))
                n_wrong = n_sampling - n_correct
                
                stats['string_match_correct_rollouts'] += n_correct
                stats['string_match_wrong_rollouts'] += n_wrong
            else:
                # 有judge结果
                has_judge_results = True
                string_match_results = result.get('all_string_match_results', [])
                judge_results = result.get('all_judge_results', [])
                
                if judge_results is None:
                    missing_judge_count += 1
                    continue
                
                n_sampling = len(string_match_results)
                stats['total_rollouts'] += n_sampling
                
                # 统计每个rollout
                for string_match_correct, judge_result in zip(string_match_results, judge_results):
                    if string_match_correct:
                        stats['string_match_correct_rollouts'] += 1
                        if judge_result is True:
                            stats['judge_agree_when_correct'] += 1
                        elif judge_result is False:
                            stats['judge_disagree_when_correct'] += 1
                    else:
                        stats['string_match_wrong_rollouts'] += 1
                        if judge_result is False:
                            stats['judge_agree_when_wrong'] += 1
                        elif judge_result is True:
                            stats['judge_disagree_when_wrong'] += 1
        else:
            # 单个rollout（如MMLUPro）
            string_match_correct = result.get('string_match_correct', False)
            judge_result = result.get('judge_result', None)
            
            stats['total_rollouts'] += 1
            
            # 检查是否有judge结果
            if judge_result is not None:
                has_judge_results = True
            
            if string_match_correct:
                stats['string_match_correct_rollouts'] += 1
                if judge_result is True:
                    stats['judge_agree_when_correct'] += 1
                elif judge_result is False:
                    stats['judge_disagree_when_correct'] += 1
                elif judge_result is None:
                    missing_judge_count += 1
            else:
                stats['string_match_wrong_rollouts'] += 1
                if judge_result is False:
                    stats['judge_agree_when_wrong'] += 1
                elif judge_result is True:
                    stats['judge_disagree_when_wrong'] += 1
                elif judge_result is None:
                    missing_judge_count += 1
    
    # 添加警告信息
    stats['has_judge_results'] = has_judge_results
    stats['missing_judge_count'] = missing_judge_count
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze judge consistency in AIME and MMLUPro results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing result files')
    parser.add_argument('--aime24_file', type=str, default=None,
                       help='Path to aime24_results.jsonl (auto-detect if not specified)')
    parser.add_argument('--aime25_file', type=str, default=None,
                       help='Path to aime25_results.jsonl (auto-detect if not specified)')
    parser.add_argument('--mmlupro_file', type=str, default=None,
                       help='Path to mmlupro_results.jsonl (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # 自动查找结果文件
    if args.aime24_file is None:
        aime24_file = os.path.join(args.results_dir, 'aime24_results.jsonl')
    else:
        aime24_file = args.aime24_file
    
    if args.aime25_file is None:
        aime25_file = os.path.join(args.results_dir, 'aime25_results.jsonl')
    else:
        aime25_file = args.aime25_file
    
    if args.mmlupro_file is None:
        mmlupro_file = os.path.join(args.results_dir, 'mmlupro_results.jsonl')
    else:
        mmlupro_file = args.mmlupro_file
    
    # 加载结果
    all_stats = []
    
    for file_path, dataset_name in [(aime24_file, 'aime24'), (aime25_file, 'aime25'), (mmlupro_file, 'mmlupro')]:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {dataset_name}")
        print(f"{'='*60}")
        
        results = load_results(file_path)
        print(f"Loaded {len(results)} questions")
        
        # 分析judge一致性
        stats = analyze_from_aggregated_stats(results, dataset_name)
        
        # 检查是否有judge结果
        if not stats['has_judge_results']:
            print(f"\n⚠️  WARNING: No judge results found in {dataset_name} results file!")
            print(f"   This dataset has {stats['missing_judge_count']} questions without judge results.")
            print(f"\n   To fix this, you have two options:")
            print(f"   1. Re-run the evaluation (recommended):")
            print(f"      python evaluate_with_judge.py --model <model> --datasets {dataset_name} --output_dir ./results --use_judge")
            print(f"   2. The results file contains 'all_responses' but no judge evaluations.")
            print(f"      You would need to re-evaluate judge for each response (requires model).")
            print(f"\n   Skipping judge consistency analysis for {dataset_name}...")
            continue
        
        if stats['missing_judge_count'] > 0:
            print(f"\n⚠️  WARNING: {stats['missing_judge_count']} questions are missing judge results.")
            print(f"   These questions will be excluded from judge consistency analysis.")
        
        all_stats.append(stats)
        
        # 打印统计结果
        print(f"\nStatistics for {dataset_name}:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Total rollouts: {stats['total_rollouts']}")
        print(f"  String match correct rollouts: {stats['string_match_correct_rollouts']} ({stats['string_match_correct_rollouts']/stats['total_rollouts']*100:.2f}%)")
        print(f"  String match wrong rollouts: {stats['string_match_wrong_rollouts']} ({stats['string_match_wrong_rollouts']/stats['total_rollouts']*100:.2f}%)")
        
        if stats['string_match_correct_rollouts'] > 0:
            judge_agree_rate_when_correct = stats['judge_agree_when_correct'] / stats['string_match_correct_rollouts'] * 100
            judge_disagree_rate_when_correct = stats['judge_disagree_when_correct'] / stats['string_match_correct_rollouts'] * 100
            print(f"\n  When string match is CORRECT ({stats['string_match_correct_rollouts']} rollouts):")
            print(f"    Judge agrees (also judges as correct): {judge_agree_rate_when_correct:.2f}% ({stats['judge_agree_when_correct']}/{stats['string_match_correct_rollouts']})")
            print(f"    Judge disagrees (judges as wrong): {judge_disagree_rate_when_correct:.2f}% ({stats['judge_disagree_when_correct']}/{stats['string_match_correct_rollouts']})")
        else:
            print(f"\n  When string match is CORRECT: No correct rollouts")
        
        if stats['string_match_wrong_rollouts'] > 0:
            judge_agree_rate_when_wrong = stats['judge_agree_when_wrong'] / stats['string_match_wrong_rollouts'] * 100
            judge_disagree_rate_when_wrong = stats['judge_disagree_when_wrong'] / stats['string_match_wrong_rollouts'] * 100
            print(f"\n  When string match is WRONG ({stats['string_match_wrong_rollouts']} rollouts):")
            print(f"    Judge agrees (also judges as wrong): {judge_agree_rate_when_wrong:.2f}% ({stats['judge_agree_when_wrong']}/{stats['string_match_wrong_rollouts']})")
            print(f"    Judge disagrees (judges as correct): {judge_disagree_rate_when_wrong:.2f}% ({stats['judge_disagree_when_wrong']}/{stats['string_match_wrong_rollouts']})")
        else:
            print(f"\n  When string match is WRONG: No wrong rollouts")
        
        # 计算总体一致性
        total_agreements = stats['judge_agree_when_correct'] + stats['judge_agree_when_wrong']
        total_disagreements = stats['judge_disagree_when_correct'] + stats['judge_disagree_when_wrong']
        overall_agreement_rate = total_agreements / stats['total_rollouts'] * 100 if stats['total_rollouts'] > 0 else 0
        print(f"\n  Overall judge agreement rate: {overall_agreement_rate:.2f}% ({total_agreements}/{stats['total_rollouts']})")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'When Correct':<25} {'When Wrong':<25} {'Overall':<15}")
    print("-" * 80)
    
    for stats in all_stats:
        dataset = stats['dataset']
        
        if stats['string_match_correct_rollouts'] > 0:
            judge_agree_correct = stats['judge_agree_when_correct'] / stats['string_match_correct_rollouts'] * 100
        else:
            judge_agree_correct = 0.0
        
        if stats['string_match_wrong_rollouts'] > 0:
            judge_agree_wrong = stats['judge_agree_when_wrong'] / stats['string_match_wrong_rollouts'] * 100
        else:
            judge_agree_wrong = 0.0
        
        total_agreements = stats['judge_agree_when_correct'] + stats['judge_agree_when_wrong']
        overall_agreement = total_agreements / stats['total_rollouts'] * 100 if stats['total_rollouts'] > 0 else 0
        
        print(f"{dataset:<15} {judge_agree_correct:>6.2f}% agree      {judge_agree_wrong:>6.2f}% agree      {overall_agreement:>6.2f}%")


if __name__ == '__main__':
    main()

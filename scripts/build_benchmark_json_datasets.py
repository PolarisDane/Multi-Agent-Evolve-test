#!/usr/bin/env python3
"""
Build two JSON datasets from MAE benchmark table:
1. fixed_benchmark_1per.json: 1 sample per benchmark (22 benchmarks)
2. fixed_benchmark_25per.json: 25 samples per benchmark (550 samples total)

Format follows fixed_fusionbench_1000.json:
- data_source, prompt, question, ability, reward_model (style, ground_truth), extra_info
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATH_EVAL_DATA = PROJECT_ROOT / "evaluation" / "math_eval" / "eval" / "data"
CODE_EVAL_DATA = PROJECT_ROOT / "evaluation" / "code_eval" / "data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "fixed_datasets"


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def to_fusionbench_format(
    content: str,
    ground_truth: str,
    benchmark: str,
    idx: int,
) -> Dict:
    """Convert single sample to fixed_fusionbench format."""
    return {
        "data_source": "gen_general",
        "prompt": [{"role": "user", "content": content}],
        "question": content,
        "ability": "general",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": "train",
            "index": idx,
            "metric": "gen_general",
            "chosen_references": [],
            "benchmark": benchmark,
        },
    }


def load_gsm8k(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "gsm8k" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_math(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "math" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_gpqa(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "gpqa" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_amc(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "amc23" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_olympiad(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "olympiadbench" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_minerva(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "minerva_math" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_mmlupro(n: int) -> List[Dict]:
    path = MATH_EVAL_DATA / "mmlupro" / "test.jsonl"
    return load_jsonl(path)[:n]


def load_humanevalplus(n: int) -> List[Dict]:
    path = CODE_EVAL_DATA / "HumanEvalPlus.jsonl"
    return load_jsonl(path)[:n]


def load_mbppplus(n: int) -> List[Dict]:
    path = CODE_EVAL_DATA / "MbppPlus.jsonl"
    return load_jsonl(path)[:n]


def load_from_hf(benchmark: str, n: int) -> List[Dict]:
    """Load from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] 'datasets' not installed. Skipping HuggingFace benchmarks. Run: pip install datasets")
        return []

    loaders = {
        "arc_c": ("allenai/ai2_arc", "ARC-Challenge", "test"),
        "mmlu": ("cais/mmlu", "abstract_algebra", "test"),
        "commonsenseqa": ("tau/commonsense_qa", None, "validation"),
        "openbookqa": ("allenai/openbookqa", "main", "test"),
        "nq": ("nq_open", None, "validation"),
        "triviaqa": ("mandarjoshi/trivia_qa", "rc", "validation"),
        "squad": ("squad_v2", None, "validation"),
        "boolq": ("google/boolq", None, "validation"),
        "hellaswag": ("Rowan/hellaswag", None, "validation"),
        "truthfulqa": ("truthfulqa/truthful_qa", "generation", "validation"),
        "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
        "bbh": ("SaylorTwift/bbh", "boolean_expressions", "test"),
        "livebench_reasoning": ("livebench/reasoning", None, "test"),
        "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
    }
    if benchmark not in loaders:
        return []
    ds_name, config, split = loaders[benchmark]
    try:
        if config:
            ds = load_dataset(ds_name, config, split=split)
        else:
            ds = load_dataset(ds_name, split=split)
    except Exception as e:
        print(f"[WARN] Failed to load {benchmark}: {e}")
        return []

    items = list(ds)[:n]
    return items


def convert_gsm8k(raw: Dict) -> tuple:
    q = raw.get("question", "")
    ans = raw.get("answer", "")
    if "####" in ans:
        ans = ans.split("####")[-1].strip()
    return q, ans


def convert_math(raw: Dict) -> tuple:
    q = raw.get("problem", raw.get("question", ""))
    ans = raw.get("answer", "")
    return q, ans


def convert_gpqa(raw: Dict) -> tuple:
    q = raw.get("question", "")
    ans = raw.get("answer", "")
    return q, ans


def convert_amc(raw: Dict) -> tuple:
    q = raw.get("question", raw.get("problem", ""))
    ans = str(raw.get("answer", ""))
    return q, ans


def convert_olympiad(raw: Dict) -> tuple:
    q = raw.get("question", "")
    fa = raw.get("final_answer", [""])
    ans = fa[0] if isinstance(fa, list) else str(fa)
    return q, ans


def convert_minerva(raw: Dict) -> tuple:
    q = raw.get("problem", "")
    sol = raw.get("solution", "")
    import re
    m = re.search(r"\\boxed\{([^}]+)\}", sol)
    ans = m.group(1) if m else sol
    return q, ans


def convert_mmlupro(raw: Dict) -> tuple:
    q = raw.get("question", "")
    ans = raw.get("answer", "")
    return q, ans


def convert_humanevalplus(raw: Dict) -> tuple:
    prompt = raw.get("prompt", "")
    content = f"Complete the following Python function. Write only the function body (the part after the docstring), no extra explanation.\n\n```python\n{prompt}\n```"
    sol = raw.get("canonical_solution", "")
    return content, sol


def convert_mbppplus(raw: Dict) -> tuple:
    prompt = raw.get("prompt", "")
    content = f"Complete the following Python function. Write only the function body, no extra explanation.\n\n```python\n{prompt}\n```"
    sol = raw.get("canonical_solution", "")
    return content, sol


def convert_arc_c(raw: Dict) -> tuple:
    q = raw["question"]
    choices = raw["choices"]["text"]
    labels = raw["choices"]["label"]
    txt = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices)])
    content = f"Question: {q}\n\nChoices:\n{txt}\n\nChoose the correct answer:"
    return content, raw["answerKey"]


def convert_mmlu(raw: Dict) -> tuple:
    q = raw["question"]
    choices = raw["choices"]
    txt = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
    content = f"Question: {q}\n\nChoices:\n{txt}\n\nChoose the correct answer (A-D):"
    ans = chr(65 + raw["answer"])
    return content, ans


def convert_commonsenseqa(raw: Dict) -> tuple:
    q = raw["question"]
    choices = raw["choices"]["text"]
    labels = raw["choices"]["label"]
    txt = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices)])
    content = f"Question: {q}\n\nChoices:\n{txt}\n\nChoose the correct answer:"
    return content, raw["answerKey"]


def convert_openbookqa(raw: Dict) -> tuple:
    q = raw["question_stem"]
    choices = raw["choices"]["text"]
    labels = raw["choices"]["label"]
    txt = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices)])
    content = f"Question: {q}\n\nChoices:\n{txt}\n\nChoose the correct answer:"
    return content, raw["answerKey"]


def convert_nq(raw: Dict) -> tuple:
    q = raw["question"]
    ans = raw["answer"]
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    content = f"Answer the following question concisely:\n\nQuestion: {q}"
    return content, ans


def convert_triviaqa(raw: Dict) -> tuple:
    q = raw["question"]
    ans = raw["answer"]
    if isinstance(ans, dict):
        ans = ans.get("value", "")
    content = f"Answer the following trivia question:\n\nQuestion: {q}"
    return content, ans


def convert_squad(raw: Dict) -> tuple:
    ctx = raw["context"]
    q = raw["question"]
    ans = raw.get("answers", {})
    if isinstance(ans, dict) and "text" in ans:
        ans = ans["text"][0] if ans["text"] else ""
    else:
        ans = ""
    if not ans:
        return None, None
    content = f"Read the following passage and answer the question.\n\nContext: {ctx}\n\nQuestion: {q}"
    return content, ans


def convert_boolq(raw: Dict) -> tuple:
    passage = raw["passage"]
    q = raw["question"]
    ans = "yes" if raw["answer"] else "no"
    content = f"Passage: {passage}\n\nQuestion: {q}\n\nAnswer with only 'yes' or 'no':"
    return content, ans


def convert_hellaswag(raw: Dict) -> tuple:
    ctx = raw["ctx"]
    endings = raw["endings"]
    txt = "\n".join([f"{chr(65+j)}. {e}" for j, e in enumerate(endings)])
    content = f"Context: {ctx}\n\nChoices:\n{txt}\n\nChoose the most appropriate continuation (A-D):"
    ans = chr(65 + int(raw["label"]))
    return content, ans


def convert_truthfulqa(raw: Dict) -> tuple:
    q = raw["question"]
    ans = raw.get("best_answer", "")
    content = f"Answer the following question truthfully:\n\n{q}"
    return content, ans


def convert_winogrande(raw: Dict) -> tuple:
    sent = raw["sentence"]
    o1, o2 = raw["option1"], raw["option2"]
    ans = "A" if raw["answer"] == "1" else "B"
    content = f"Complete the sentence by choosing the most appropriate option.\n\nSentence: {sent}\n\nA. {o1}\nB. {o2}\n\nChoose (A or B):"
    return content, ans


def convert_bbh(raw: Dict) -> tuple:
    q = raw["input"]
    ans = raw["target"]
    return q, ans


def convert_livebench_reasoning(raw: Dict) -> tuple:
    turns = raw.get("turns", [])
    q = turns[0] if turns else "No problem"
    ans = raw.get("ground_truth", "")
    return q, ans


def convert_mmlu_pro(raw: Dict) -> tuple:
    q = raw.get("question", "")
    opts = raw.get("options", [])
    ans_idx = raw.get("answer_index", 0)
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    txt = "\n".join([f"{labels[j]}. {o}" for j, o in enumerate(opts[:10])])
    content = f"Question: {q}\n\nChoices:\n{txt}\n\nChoose the correct answer (A-J):"
    ans = labels[ans_idx] if ans_idx < len(labels) else raw.get("answer", "")
    return content, ans


# Map benchmark name -> (loader_func, converter_func, use_hf)
BENCHMARK_CONFIG = {
    "gsm8k": ("local", load_gsm8k, convert_gsm8k),
    "math": ("local", load_math, convert_math),
    "arc_c": ("hf", load_from_hf, convert_arc_c),
    "mmlu": ("hf", load_from_hf, convert_mmlu),
    "gpqa": ("local", load_gpqa, convert_gpqa),
    "commonsenseqa": ("hf", load_from_hf, convert_commonsenseqa),
    "openbookqa": ("hf", load_from_hf, convert_openbookqa),
    "nq": ("hf", load_from_hf, convert_nq),
    "triviaqa": ("hf", load_from_hf, convert_triviaqa),
    "squad": ("hf", load_from_hf, convert_squad),
    "boolq": ("hf", load_from_hf, convert_boolq),
    "hellaswag": ("hf", load_from_hf, convert_hellaswag),
    "mbpp_plus": ("local", load_mbppplus, convert_mbppplus),
    "humaneval_plus": ("local", load_humanevalplus, convert_humanevalplus),
    "truthfulqa": ("hf", load_from_hf, convert_truthfulqa),
    "bbh": ("hf", load_from_hf, convert_bbh),
    "livebench_reasoning": ("hf", load_from_hf, convert_livebench_reasoning),
    "amc": ("local", load_amc, convert_amc),
    "minerva": ("local", load_minerva, convert_minerva),
    "winogrande": ("hf", load_from_hf, convert_winogrande),
    "olympiad": ("local", load_olympiad, convert_olympiad),
    "mmlu_pro": ("local", load_mmlupro, convert_mmlupro),
}


def load_benchmark_samples(benchmark: str, n: int) -> List[Dict]:
    """Load n samples from benchmark, return list of (content, ground_truth) tuples."""
    cfg = BENCHMARK_CONFIG.get(benchmark)
    if not cfg:
        return []
    source, loader, converter = cfg
    if source == "local":
        raw_list = loader(n)
    else:
        raw_list = loader(benchmark, n)
    out = []
    for r in raw_list:
        try:
            content, gt = converter(r)
            if content is not None and gt is not None:
                out.append((content, gt))
        except Exception as e:
            print(f"[WARN] Failed to convert {benchmark} sample: {e}")
    return out


def build_dataset(samples_per_benchmark: int, seed: int = 42) -> List[Dict]:
    """Build dataset with given samples per benchmark."""
    random.seed(seed)
    examples = []
    idx = 0
    for bench in BENCHMARK_CONFIG:
        samples = load_benchmark_samples(bench, samples_per_benchmark * 2)
        if not samples:
            print(f"[WARN] No samples for {bench}")
            continue
        if len(samples) > samples_per_benchmark:
            samples = random.sample(samples, samples_per_benchmark)
        for content, gt in samples:
            ex = to_fusionbench_format(content, gt, bench, idx)
            examples.append(ex)
            idx += 1
    return examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import time
    ts = str(int(time.time()))

    # Dataset 1: 1 sample per benchmark
    print("Building fixed_benchmark_1per.json (1 sample per benchmark)...")
    d1 = build_dataset(1)
    out1 = {
        "metadata": {
            "num_samples": len(d1),
            "source": "MAE Benchmarks",
            "split": "train",
            "seed": 42,
            "per_benchmark": 1,
            "extraction_date": ts,
            "description": "1 sample per benchmark from MAE evaluation table",
        },
        "examples": d1,
    }
    out_path_1 = OUTPUT_DIR / "fixed_benchmark_1per.json"
    with open(out_path_1, "w", encoding="utf-8") as f:
        json.dump(out1, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(d1)} samples to {out_path_1}")

    # Dataset 2: 25 samples per benchmark
    print("Building fixed_benchmark_25per.json (25 samples per benchmark)...")
    d2 = build_dataset(25)
    out2 = {
        "metadata": {
            "num_samples": len(d2),
            "source": "MAE Benchmarks",
            "split": "train",
            "seed": 42,
            "per_benchmark": 25,
            "extraction_date": ts,
            "description": "25 samples per benchmark from MAE evaluation table",
        },
        "examples": d2,
    }
    out_path_2 = OUTPUT_DIR / "fixed_benchmark_25per.json"
    with open(out_path_2, "w", encoding="utf-8") as f:
        json.dump(out2, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(d2)} samples to {out_path_2}")


if __name__ == "__main__":
    main()

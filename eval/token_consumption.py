from pathlib import Path
import json


def run_dir(path: str, name: str):
    json_files = Path(path).glob("*.json")
    all_prompt_tokens = []
    all_completion_tokens = []
    for file in json_files:
        if name in file.name:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            prompt_tokens = data["prompt_tokens"]
            completion_tokens = data["completion_tokens"]

            if prompt_tokens > 0:
                all_prompt_tokens.append(prompt_tokens)
            if completion_tokens > 0:
                all_completion_tokens.append(completion_tokens)

    print(f"{name}: average prompt_tokens: {sum(all_prompt_tokens)/len(all_prompt_tokens)} "
          f"average completion_tokens: {sum(all_completion_tokens)/len(all_completion_tokens)}")


run_dir("./token_consumption", "locomo")
run_dir("./token_consumption", "longmemeval")


import json
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np


def simple_tokenize(text):
    """简单的分词与文本清洗函数（用于计算 F1）"""
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def compute_f1(prediction, reference):
    """基于词重叠（Overlap Token Level）计算 F1 Score"""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)

    if (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def calculate_bleu_scores(prediction: str, reference: str):
    """使用 NLTK 计算 BLEU 1-4 分数"""
    try:
        pred_tokens = nltk.word_tokenize(str(prediction).lower())
        ref_tokens = [nltk.word_tokenize(str(reference).lower())]
    except Exception:
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = [simple_tokenize(reference)]

    weights_list = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smooth,
            )
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = score

    return scores


def process_eval_file(file_path: str, dataset_name: str):
    """读取单文件 JSON 列表并统计 Token、F1、Accuracy 和 BLEU 1-4"""
    file_p = Path(file_path)
    if not file_p.exists():
        print(f"Error: File {file_path} does not exist.")
        return

    try:
        with open(file_p, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if not isinstance(data_list, list):
        print(f"Error: Expected a JSON array in {file_path}")
        return

    all_prompt_tokens = []
    all_completion_tokens = []

    metrics_by_category = defaultdict(
        lambda: {
            "f1": [],
            "judge_correct": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
        }
    )
    global_f1s = []
    global_judge_scores = []
    global_bleus = defaultdict(list)

    for item in data_list:
        # 1. 收集 Token 消耗
        p_tok = item.get("prompt_tokens", 0)
        c_tok = item.get("completion_tokens", 0)
        if p_tok > 0:
            all_prompt_tokens.append(p_tok)
        if c_tok > 0:
            all_completion_tokens.append(c_tok)

        # 2. 获取 Category、Prediction、Reference 和 Correct 判定
        if dataset_name.lower() == "longmemeval":
            cat_key = item.get("question_type", "LongMemEval QA")
            pred = item.get("system_answer", "")
            ref = item.get("golden_answer", "")
        else:  # locomo 数据集
            cat = item.get("category") or item.get("question_type") or "uncategorized"
            cat_key = f"Category {cat}" if str(cat).isdigit() else str(cat)
            pred = item.get("system_answer") or item.get("prediction", "")
            ref = item.get("original_answer") or item.get("reference", "")

        correct_val = item.get("correct")
        judge_score = None
        if correct_val is not None:
            judge_score = 1.0 if correct_val is True else (0.0 if correct_val is False else float(correct_val))

        # 计算指标
        f1_score = compute_f1(pred, ref)
        bleu_scores = calculate_bleu_scores(pred, ref)

        # 记录分类指标
        metrics_by_category[cat_key]["f1"].append(f1_score)
        global_f1s.append(f1_score)

        for b_key in ["bleu1", "bleu2", "bleu3", "bleu4"]:
            b_val = bleu_scores.get(b_key, 0.0)
            metrics_by_category[cat_key][b_key].append(b_val)
            global_bleus[b_key].append(b_val)

        if judge_score is not None:
            metrics_by_category[cat_key]["judge_correct"].append(judge_score)
            global_judge_scores.append(judge_score)

    # 3. 打印统计结果
    avg_prompt = np.mean(all_prompt_tokens) if all_prompt_tokens else 0.0
    avg_comp = np.mean(all_completion_tokens) if all_completion_tokens else 0.0

    print(f"\n================ [{dataset_name.upper()}] Evaluation Summary ================")
    print(
        f"Average Prompt Tokens    : {avg_prompt:.2f}\n"
        f"Average Completion Tokens: {avg_comp:.2f}"
    )

    if metrics_by_category:
        has_judge = len(global_judge_scores) > 0

        header = f"{'Category / Type':<28} | {'F1':<8} | {'BLEU-1':<8} | {'BLEU-2':<8} | {'BLEU-3':<8} | {'BLEU-4':<8}"
        if has_judge:
            header += f" | {'Accuracy':<10}"
        header += f" | {'Count':<6}"

        print("\n" + "-" * len(header))
        print(header)
        print("-" * len(header))

        for cat_key in sorted(metrics_by_category.keys()):
            cat_data = metrics_by_category[cat_key]

            avg_f1 = np.mean(cat_data["f1"]) if cat_data["f1"] else 0.0
            avg_b1 = np.mean(cat_data["bleu1"]) if cat_data["bleu1"] else 0.0
            avg_b2 = np.mean(cat_data["bleu2"]) if cat_data["bleu2"] else 0.0
            avg_b3 = np.mean(cat_data["bleu3"]) if cat_data["bleu3"] else 0.0
            avg_b4 = np.mean(cat_data["bleu4"]) if cat_data["bleu4"] else 0.0

            line = (
                f"{str(cat_key):<28} | {avg_f1:<8.4f} | {avg_b1:<8.4f} | "
                f"{avg_b2:<8.4f} | {avg_b3:<8.4f} | {avg_b4:<8.4f}"
            )

            if has_judge:
                judges = cat_data["judge_correct"]
                avg_judge = np.mean(judges) if judges else None
                judge_str = f"{avg_judge:<10.4f}" if avg_judge is not None else f"{'N/A':<10}"
                line += f" | {judge_str}"

            line += f" | {len(cat_data['f1']):<6}"
            print(line)

        # 输出 OVERALL 总平均
        print("-" * len(header))
        ov_f1 = np.mean(global_f1s) if global_f1s else 0.0
        ov_b1 = np.mean(global_bleus["bleu1"]) if global_bleus["bleu1"] else 0.0
        ov_b2 = np.mean(global_bleus["bleu2"]) if global_bleus["bleu2"] else 0.0
        ov_b3 = np.mean(global_bleus["bleu3"]) if global_bleus["bleu3"] else 0.0
        ov_b4 = np.mean(global_bleus["bleu4"]) if global_bleus["bleu4"] else 0.0

        ov_line = (
            f"{'OVERALL (Total Avg)':<28} | {ov_f1:<8.4f} | {ov_b1:<8.4f} | "
            f"{ov_b2:<8.4f} | {ov_b3:<8.4f} | {ov_b4:<8.4f}"
        )

        if has_judge:
            overall_judge = np.mean(global_judge_scores) if global_judge_scores else 0.0
            ov_line += f" | {overall_judge:<10.4f}"

        ov_line += f" | {len(global_f1s):<6}"
        print(ov_line)
        print("=" * len(header))
    else:
        print("No metrics found.")


if __name__ == "__main__":
    # 配置你的 JSON 数据文件路径
    tasks = [
        ("./results/longmemeval_result.json", "longmemeval"),
        ("./results/locomo_result.json", "locomo"),
    ]

    for file_path, name in tasks:
        process_eval_file(file_path, name)
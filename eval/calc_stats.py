#!/usr/bin/env python3
"""
统计 JSON 列表中每个类别的准确率，并输出总体平均准确率。
输入：一个 JSON 数组，每个元素包含 "category" 和 "correct" 字段。
输出：每个类别的总数、正确数、准确率，以及总体准确率。
"""

import json
import sys
from collections import defaultdict

def calculate_accuracy(data):
    """
    统计每个 category 的准确率，同时累计总样本数和总正确数。
    返回：
        results: 列表，元素为 (category, total, correct, accuracy)
        total_all: 总样本数
        correct_all: 总正确数
    """
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total_all = 0
    correct_all = 0

    for item in data:
        if "category" not in item or "correct" not in item:
            continue
        cat = item["category"]
        stats[cat]["total"] += 1
        total_all += 1
        if item["correct"]:
            stats[cat]["correct"] += 1
            correct_all += 1

    # 计算每个类别的准确率
    results = []
    for cat, counts in stats.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracy = correct / total if total > 0 else 0.0
        results.append((cat, total, correct, accuracy))

    # 按类别排序（统一转为字符串以便混合类型）
    results.sort(key=lambda x: str(x[0]))

    return results, total_all, correct_all

def main():
    # 读取 JSON
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取文件失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            data = json.load(sys.stdin)
        except Exception as e:
            print(f"读取标准输入失败: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, list):
        print("错误：输入应为 JSON 数组", file=sys.stderr)
        sys.exit(1)

    results, total_all, correct_all = calculate_accuracy(data)

    if not results:
        print("未找到有效数据（缺少 category 或 correct 字段）")
        return

    # 打印每个类别的统计表
    print(f"{'Category':<15} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 45)
    for cat, total, correct, acc in results:
        print(f"{str(cat):<15} {total:<8} {correct:<8} {acc:.2%}")

    # 打印总体准确率
    overall_acc = correct_all / total_all if total_all > 0 else 0.0
    print("\n" + "=" * 45)
    print(f"总体准确率 (Total Accuracy) : {overall_acc:.2%}  ({correct_all}/{total_all})")

    # 如需输出各类别准确率的算术平均值（宏平均），可取消下面注释
    # macro_acc = sum(acc for _, _, _, acc in results) / len(results) if results else 0.0
    # print(f"类别平均准确率 (Macro Avg) : {macro_acc:.2%}")

if __name__ == "__main__":
    main()
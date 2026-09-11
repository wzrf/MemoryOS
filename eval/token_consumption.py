from pathlib import Path
import json
import re
import hashlib
from openai import OpenAI
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np

SYSTEM = """You are a strict, method-blind evaluator of question answering. Judge only whether the candidate answer is semantically correct
  according to the question and reference answer. Do not infer which system produced it."""
TEMPLATE = """Decide whether the candidate answer is correct.

  Rules:
  1. Accept concise paraphrases, equivalent names, equivalent date/number formats, and a correct answer embedded in harmless extra explanation.
  2. Reject a wrong person, entity, event, date, ordering, count, amount, or polarity; a contradiction; a refusal when the reference answers the question; or an answer missing a required list item, comparison, calculation, or event.
  3. Extra text is harmless only if it does not add a materially false answer claim.
  4. For open-ended preference or recommendation questions, the answer need not copy every example in the reference, but it must correctly use the core personal information required by the reference.
  5. Treat the reference as the scoring ground truth. Do not use outside knowledge.

  Question:
  {question}

  Reference answer:
  {reference}

  Candidate answer:
  {prediction}

  Do not REASON. JUST GIVE THE RESULT.
  Return exactly one JSON object with one boolean field and no other text:
  {{"correct": true}}
  or
  {{"correct": false}}"""

def text_sha256(text: str) -> str:
    """Return SHA256 hex digest of the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

PROMPT_SHA256 = text_sha256(SYSTEM + "\n\0\n" + TEMPLATE)

# Cache for LLM judge results
CACHE_FILE = Path(".llm_judge_cache.json")
_cache_lock = threading.RLock()
_judge_cache = {}

def load_cache():
    """Load judge cache from disk."""
    global _judge_cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _judge_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache file: {e}")
            _judge_cache = {}
    else:
        _judge_cache = {}

def save_cache():
    """Save judge cache to disk."""
    with _cache_lock:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_judge_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache file: {e}")

def get_cache_key(question: str, reference: str, prediction: str) -> str:
    """Generate a cache key from question, reference, prediction."""
    # Use SHA256 of concatenated strings
    content = f"{question}|{reference}|{prediction}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# Load cache on module import
load_cache()

def parse_correct(content: object) -> bool:
    text = str(content or "").strip()
    text = re.sub(r"^(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*$", "", text)
    try:
        value = json.loads(text)
    except Exception as e:
        print(f"parse error: {e} text: {text}")
        raise ValueError("judge response is not valid JSON")
    if not isinstance(value, dict) or set(value) != {"correct"} or not isinstance(value["correct"], bool):
        raise ValueError("judge response is not strict correct:boolean JSON")
    return value["correct"]

def request_once(api_key: str, endpoint: str, model: str, item: dict, timeout: int = 30, max_tokens: int = 100) -> tuple[bool, dict, str]:
    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=timeout,
    )
    # 调用 Chat Completions API
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(**item)},
        ],
        max_tokens=2048,
        stream=False,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}}
    )
    # 解析响应数据
    choice = completion.choices[0]
    message = choice.message
    content = message.content or ""
    # 如果 content 为空，尝试获取推理链内容（兼容深度思考模型）
    if not content and hasattr(message, "reasoning_content"):
        content = message.reasoning_content or ""
    # 获取 token 使用量字典
    usage = completion.usage.model_dump() if completion.usage else {}
    return parse_correct(content), usage, str(completion.model or "")

def llm_judge(question: str, reference: str, prediction: str,
              api_key: str = None, endpoint: str = None, model: str = None,
              max_retries: int = 3, use_cache: bool = True) -> bool:
    """
    Judge correctness using LLM with retry logic and caching.
    Returns True if correct, False otherwise.

    Args:
        question: The question text
        reference: Reference/gold answer
        prediction: Predicted/system answer
        api_key: API key, defaults to "sk-dummy"
        endpoint: API endpoint, defaults to "http://127.0.0.1:30002/v1"
        model: Model name, defaults to "GLM-5.3"
        max_retries: Maximum number of retry attempts
        use_cache: Whether to use cache for previously judged items
    """
    # Set defaults
    if api_key is None:
        api_key = "sk-dummy"
    if endpoint is None:
        endpoint = "http://127.0.0.1:30002/v1"
    if model is None:
        model = "GLM-5.3"

    # Check cache if enabled
    if use_cache:
        cache_key = get_cache_key(question, reference, prediction)
        with _cache_lock:
            if cache_key in _judge_cache:
                cached_result = _judge_cache[cache_key]
                # print(f"Cache hit for judgment (key: {cache_key[:16]}...): {cached_result}")
                return cached_result

    item = {"question": question, "reference": reference, "prediction": prediction}

    for attempt in range(max_retries):
        try:
            correct, usage, model_name = request_once(api_key, endpoint, model, item)

            # Store in cache if enabled
            if use_cache:
                with _cache_lock:
                    _judge_cache[cache_key] = correct
                    # Save cache periodically or on exit, but we'll save immediately
                    save_cache()

            # Optionally log usage
            # print(f"Judge usage: {usage}")
            return correct
        except Exception as e:
            print(f"Judge attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return False


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


def process_eval_file(file_path: str, dataset_name: str, use_llm_judge: bool = False,
                       api_key: str = None, endpoint: str = None, model: str = None,
                       max_workers: int = 32):
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

    # LLM judge configuration
    if use_llm_judge:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", "sk-dummy")
        if endpoint is None:
            endpoint = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30002/v1")
        if model is None:
            model = os.environ.get("LLM_MODEL", "GLM-5.3")
        print(f"LLM judge configured: endpoint={endpoint}, model={model}")

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

    # Collect all items data for processing
    item_data_list = []
    for idx, item in enumerate(data_list):
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

        # 获取问题文本
        q_text = item.get("question", cat_key)

        # 存储项数据
        item_data_list.append({
            "idx": idx,
            "cat_key": cat_key,
            "pred": pred,
            "ref": ref,
            "q_text": q_text,
            "item": item,
        })

    # 并发执行 LLM judge（如果需要）
    judge_results = [None] * len(item_data_list)
    if use_llm_judge:
        print(f"Starting concurrent LLM judge with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for item_data in item_data_list:
                idx = item_data["idx"]
                future = executor.submit(
                    llm_judge,
                    question=item_data["q_text"],
                    reference=item_data["ref"],
                    prediction=item_data["pred"],
                    api_key=api_key,
                    endpoint=endpoint,
                    model=model,
                    use_cache=True
                )
                future_to_idx[future] = idx

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    correct = future.result()
                    judge_results[idx] = 1.0 if correct else 0.0
                except Exception as e:
                    print(f"LLM judge failed for item {idx}: {e}")
                    judge_results[idx] = None
        print("LLM judge completed.")

    # Process each item to compute metrics
    for item_data, judge_score in zip(item_data_list, judge_results):
        cat_key = item_data["cat_key"]
        pred = item_data["pred"]
        ref = item_data["ref"]
        item = item_data["item"]

        # If not using LLM judge, get correct field from item
        if not use_llm_judge:
            correct_val = item.get("correct")
            judge_score = 1.0 if correct_val is True else (0.0 if correct_val is False else float(correct_val)) if correct_val is not None else None

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


def process_halumem_dir(dir_path: str):
    """处理 results_memoryos_halumem 目录，统计每个问题类型的平均 token 消耗和指标"""
    import json
    from pathlib import Path
    from collections import defaultdict
    import numpy as np

    dir_p = Path(dir_path)
    if not dir_p.exists() or not dir_p.is_dir():
        print(f"Error: Directory {dir_path} does not exist or is not a directory.")
        return

    json_files = list(dir_p.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return

    # 按问题类型分组存储数据
    groups = defaultdict(list)

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if not isinstance(data, list):
            print(f"Error: Expected a JSON array in {file}")
            continue

        for item in data:
            question_type = item.get("question_type", "unknown")

            # 收集 token 相关字段
            turn_build_memory_prompt_tokens = item.get("turn_build_memory_prompt_tokens", 0)
            turn_build_memory_completion_tokens = item.get("turn_build_memory_completion_tokens", 0)
            retrieval_prompt_tokens = item.get("retrieval_prompt_tokens", 0)
            retrieval_completion_tokens = item.get("retrieval_completion_tokens", 0)
            answer_prompt_tokens = item.get("answer_prompt_tokens", 0)
            answer_completion_tokens = item.get("answer_completion_tokens", 0)

            # 计算总和
            retrieval_prompt_total = retrieval_prompt_tokens + answer_prompt_tokens
            retrieval_completion_total = retrieval_completion_tokens + answer_completion_tokens

            # 获取预测和参考文本
            prediction = item.get("answer", "")
            reference = item.get("reference", "")

            # 计算指标
            f1_score = item.get("metrics", {}).get("f1", 0.0)
            correct = item.get("metrics", {}).get("correct", False)
            correct_score = 1.0 if correct else 0.0

            # 计算 BLEU 分数
            bleu_scores = calculate_bleu_scores(prediction, reference)

            groups[question_type].append({
                "turn_build_memory_prompt_tokens": turn_build_memory_prompt_tokens,
                "turn_build_memory_completion_tokens": turn_build_memory_completion_tokens,
                "retrieval_prompt_total": retrieval_prompt_total,
                "retrieval_completion_total": retrieval_completion_total,
                "f1": f1_score,
                "correct": correct_score,
                "bleu1": bleu_scores.get("bleu1", 0.0),
                "bleu2": bleu_scores.get("bleu2", 0.0),
                "bleu3": bleu_scores.get("bleu3", 0.0),
                "bleu4": bleu_scores.get("bleu4", 0.0),
            })

    if not groups:
        print("No data found.")
        return

    print(f"\n================ [HALUMEM] Evaluation Summary ================")
    print(f"Total files processed: {len(json_files)}")
    print(f"Total question entries: {sum(len(items) for items in groups.values())}")

    # 打印表头 - 添加 BLEU 1-4 列
    header = f"{'Question Type':<25} | {'TBM Pr':>8} | {'TBM Comp':>8} | {'Ret+Ans Pr':>11} | {'Ret+Ans Comp':>13} | {'F1':>6} | {'Acc':>6} | {'BLEU-1':>7} | {'BLEU-2':>7} | {'BLEU-3':>7} | {'BLEU-4':>7} | {'Count':>6}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    # 计算每个组的平均值
    for qtype in sorted(groups.keys()):
        items = groups[qtype]
        count = len(items)

        avg_turn_build_prompt = np.mean([i["turn_build_memory_prompt_tokens"] for i in items])
        avg_turn_build_completion = np.mean([i["turn_build_memory_completion_tokens"] for i in items])
        avg_prompt_total = np.mean([i["retrieval_prompt_total"] for i in items])
        avg_completion_total = np.mean([i["retrieval_completion_total"] for i in items])
        avg_f1 = np.mean([i["f1"] for i in items])
        avg_correct = np.mean([i["correct"] for i in items])
        avg_bleu1 = np.mean([i["bleu1"] for i in items])
        avg_bleu2 = np.mean([i["bleu2"] for i in items])
        avg_bleu3 = np.mean([i["bleu3"] for i in items])
        avg_bleu4 = np.mean([i["bleu4"] for i in items])

        line = f"{qtype:<25} | {avg_turn_build_prompt:>8.0f} | {avg_turn_build_completion:>8.0f} | {avg_prompt_total:>11.0f} | {avg_completion_total:>13.0f} | {avg_f1:>6.4f} | {avg_correct:>6.4f} | {avg_bleu1:>7.4f} | {avg_bleu2:>7.4f} | {avg_bleu3:>7.4f} | {avg_bleu4:>7.4f} | {count:>6}"
        print(line)

    # 计算整体平均值
    all_items = [item for sublist in groups.values() for item in sublist]
    if all_items:
        overall_turn_build_prompt = np.mean([i["turn_build_memory_prompt_tokens"] for i in all_items])
        overall_turn_build_completion = np.mean([i["turn_build_memory_completion_tokens"] for i in all_items])
        overall_prompt_total = np.mean([i["retrieval_prompt_total"] for i in all_items])
        overall_completion_total = np.mean([i["retrieval_completion_total"] for i in all_items])
        overall_f1 = np.mean([i["f1"] for i in all_items])
        overall_correct = np.mean([i["correct"] for i in all_items])
        overall_bleu1 = np.mean([i["bleu1"] for i in all_items])
        overall_bleu2 = np.mean([i["bleu2"] for i in all_items])
        overall_bleu3 = np.mean([i["bleu3"] for i in all_items])
        overall_bleu4 = np.mean([i["bleu4"] for i in all_items])
        total_count = len(all_items)

        print("-" * len(header))
        line = f"{'OVERALL (Total Avg)':<25} | {overall_turn_build_prompt:>8.0f} | {overall_turn_build_completion:>8.0f} | {overall_prompt_total:>11.0f} | {overall_completion_total:>13.0f} | {overall_f1:>6.4f} | {overall_correct:>6.4f} | {overall_bleu1:>7.4f} | {overall_bleu2:>7.4f} | {overall_bleu3:>7.4f} | {overall_bleu4:>7.4f} | {total_count:>6}"
        print(line)
        print("=" * len(header))


if __name__ == "__main__":
    # 配置你的 JSON 数据文件路径
    run_dir("./token_consumption", "locomo")
    run_dir("./token_consumption", "longmemeval")
    run_dir("./token_consumption_GLM-4.5-Air", "locomo")
    run_dir("./token_consumption_GLM-4.5-Air", "longmemeval")
    run_dir("./token_consumption_GLM-4.5-Air", "locomo")
    run_dir("./token_consumption_GLM-4.5-Air", "longmemeval")
    tasks = [
        ("./results/locomo_result.json", "locomo-qwen3"),
        ("./results_GLM-4.5-Air/locomo_result.json", "locomo-glm"),
        ("./results_Kimi-K2.6/locomo_result.json", "locomo-kimi"),
        ("./results/longmemeval_result.json", "longmemeval-qwen3"),
        ("./results_GLM-4.5-Air/longmemeval_result.json", "longmemeval-glm"),
        ("./results_Kimi-K2.6/longmemeval_result.json", "longmemeval-kimi"),
    ]

    max_workers = int(os.environ.get("LLM_JUDGE_MAX_WORKERS", "32"))
    for file_path, name in tasks:
        process_eval_file(file_path, name, use_llm_judge=True, max_workers=max_workers)

    # 处理 HALUMEM 结果目录
    # process_halumem_dir("./results_memoryos_halumem")
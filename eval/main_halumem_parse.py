import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入 MemoryOS 核心模块
from short_term_memory import ShortTermMemory
from mid_term_memory import MidTermMemory
from long_term_memory import LongTermMemory
from dynamic_update import DynamicUpdate
from retrieval_and_answer import RetrievalAndAnswer
from utils import OpenAIClient, get_timestamp
from judge import AnswerJudge
from main_loco_parse import update_user_profile_from_top_segment

# ============================================================================
# 全局配置与 LLM 客户端初始化
# ============================================================================

LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-11ce7640e46049a6977c0d96ba855ffb")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:30004/v1/")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "sk-11ce7640e46049a6977c0d96ba855ffb")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2")

client = OpenAIClient(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    recomputation_rate=float(os.environ.get("recomputation_rate", 0.0)),
    sglang_url="http://127.0.0.1:30003/v1/completions",
    sglang_url_prefiller="http://127.0.0.1:30003/v1/completions",
)


# ============================================================================
# HaluMem 数据结构与 JSONL 加载器
# ============================================================================

@dataclass
class HaluMemSample:
    sample_id: str
    persona_info: str
    sessions: List[Dict]


def load_halumem_dataset(path_input: Union[str, Path]) -> List[HaluMemSample]:
    """加载 HaluMem 数据集 (.jsonl 格式或包含 jsonl 的目录)"""
    path_input = Path(path_input)
    if not path_input.exists():
        raise FileNotFoundError(f"Path not found at {path_input}")

    files = sorted(list(path_input.glob("*.jsonl"))) if path_input.is_dir() else [path_input]

    samples = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sid = data.get("uuid", f"{fpath.stem}_{line_idx}")
                persona = data.get("persona_info", "")
                sessions = data.get("sessions", [])
                samples.append(HaluMemSample(sample_id=str(sid), persona_info=persona, sessions=sessions))

    print(f"Successfully loaded {len(samples)} samples from {path_input}")
    return samples


def parse_session_dialogs(session_messages: List[Dict]) -> List[Dict]:
    """解析 HaluMem Session 中的消息列表为 MemoryOS 接受的 QA 对"""
    processed = []
    i = 0
    while i < len(session_messages):
        msg = session_messages[i]
        role = msg.get("speaker", msg.get("role", ""))
        content = msg.get("content", "")
        ts = msg.get("timestamp", msg.get("time_stamp", ""))

        if role == "user":
            user_text = content
            agent_text = ""
            if i + 1 < len(session_messages):
                next_role = session_messages[i + 1].get("speaker", session_messages[i + 1].get("role", ""))
                if next_role == "assistant":
                    agent_text = session_messages[i + 1].get("content", "")
                    i += 1
            processed.append({
                "user_input": user_text,
                "agent_response": agent_text,
                "timestamp": ts,
            })
        elif role == "assistant":
            processed.append({
                "user_input": "",
                "agent_response": content,
                "timestamp": ts,
            })
        i += 1
    return processed


def calculate_simple_f1(prediction: str, reference: str) -> float:
    """简单的 Token 级 F1 计算"""
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)


# ============================================================================
# MemoryOS HaluMem 增量测试器
# ============================================================================

class HaluMemMemoryOSTester:
    def __init__(self, use_llm_judge: bool = False, mem_dir: str = "./mem_tmp_halumem"):
        self.use_llm_judge = use_llm_judge
        self.mem_dir = mem_dir
        os.makedirs(self.mem_dir, exist_ok=True)
        if use_llm_judge:
            self.judge_client = AnswerJudge(
                api_key=JUDGE_API_KEY,
                api_url=JUDGE_BASE_URL,
                model="deepseek-v3.2"
            )
        else:
            self.judge_client = None

    def run_single_sample(self, sample: HaluMemSample, sample_idx: int, embedding_model,
                          save_dir: str = "./results_memoryos_halumem"):
        safe_sample_id = sample.sample_id.replace(" ", "_")

        # 检查结果文件是否已存在且包含所有问题
        result_file = os.path.join(save_dir, f"{safe_sample_id}.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)

                # 收集已存在的question_id
                existing_question_ids = set()
                for res in existing_results:
                    if isinstance(res, dict) and 'question_id' in res:
                        existing_question_ids.add(res['question_id'])

                # 计算当前样本的所有question_id
                expected_question_ids = set()
                for s_idx, session in enumerate(sample.sessions):
                    questions = session.get("questions", [])
                    for q_idx, _ in enumerate(questions):
                        question_id = f"{safe_sample_id}_s{s_idx}_q{q_idx}"
                        expected_question_ids.add(question_id)

                # 如果已存在的question_id包含所有预期问题，跳过处理
                if expected_question_ids.issubset(existing_question_ids):
                    print(f"[{sample_idx}] Sample {safe_sample_id} already processed with all {len(expected_question_ids)} questions. Skipping.")
                    return existing_results
                else:
                    missing = expected_question_ids - existing_question_ids
                    print(f"[{sample_idx}] Sample {safe_sample_id} partially processed. Missing {len(missing)} questions. Re-processing entire sample.")
            except Exception as e:
                print(f"[{sample_idx}] Error reading existing results for {safe_sample_id}: {e}. Re-processing.")

        # 定义 3 个级别的记忆文件存储路径
        short_mem_path = os.path.join(self.mem_dir, f"{safe_sample_id}_short_term.json")
        mid_mem_path = os.path.join(self.mem_dir, f"{safe_sample_id}_mid_term.json")
        long_mem_path = os.path.join(self.mem_dir, f"{safe_sample_id}_long_term.json")

        # 清理之前可能存在的残余缓存文件，保证测试隔离
        for p in [short_mem_path, mid_mem_path, long_mem_path]:
            if os.path.exists(p):
                os.remove(p)

        # 初始化 MemoryOS 三级存储架构与动态更新器
        short_mem = ShortTermMemory(max_capacity=5, file_path=short_mem_path)
        mid_mem = MidTermMemory(max_capacity=2000, file_path=mid_mem_path, embedding_model=embedding_model)
        long_mem = LongTermMemory(file_path=long_mem_path, embedding_model=embedding_model)
        dynamic_updater = DynamicUpdate(
            short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client
        )
        retrieval_system = RetrievalAndAnswer(
            short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10
        )

        sample_results = []
        accumulated_history_turns = 0
        total_build_prompt_tokens = 0
        total_build_completion_tokens = 0

        # 遍历每个 Session
        for s_idx, session in enumerate(sample.sessions):
            print(f"running {sample_idx} {s_idx}/{len(sample.sessions)}")
            dialogue_turns = session.get("dialogue", session.get("messages", []))
            questions = session.get("questions", [])

            # ----------------------------------------------------------------
            # 步骤 1：将当前 Session 对话插入 MemoryOS，并计算构建 Token 开销
            # ----------------------------------------------------------------
            parsed_dialogs = parse_session_dialogs(dialogue_turns)

            # 记录插入前的 Token 消耗统计
            stats_before = dynamic_updater.get_stats()

            for dialog in parsed_dialogs:
                short_mem.add_qa_pair(dialog)
                if short_mem.is_full():
                    dynamic_updater.bulk_evict_and_update_mid_term()
                update_user_profile_from_top_segment(mid_mem, long_mem, safe_sample_id, client, dynamic_updater)

            accumulated_history_turns += len(parsed_dialogs)

            # 记录插入后的 Token 消耗统计，计算本轮 (Diff)
            stats_after = dynamic_updater.get_stats()

            turn_build_prompt_tokens = stats_after.get("prompt_tokens", 0) - stats_before.get("prompt_tokens", 0)
            turn_build_completion_tokens = stats_after.get("completion_tokens", 0) - stats_before.get("completion_tokens", 0)

            total_build_prompt_tokens += turn_build_prompt_tokens
            total_build_completion_tokens += turn_build_completion_tokens

            # ----------------------------------------------------------------
            # 步骤 2：对当前 Session 内的 Question 进行检索与回答
            # ----------------------------------------------------------------
            for q_idx, q_item in enumerate(questions):
                question = q_item.get("question", "")
                reference = str(q_item.get("answer", ""))
                question_type = q_item.get("question_type", "default")
                question_id = f"{safe_sample_id}_s{s_idx}_q{q_idx}"
                question_date = q_item.get("date", "")

                # 2.1 检索阶段开销统计
                retrieval_start = time.time()
                retrieval_result = retrieval_system.retrieve(
                    question,
                    segment_threshold=0.1,
                    page_threshold=0.1,
                    knowledge_threshold=0.1,
                    client=client,
                    embedding_model=embedding_model
                )
                retrieval_time = time.time() - retrieval_start

                # 2.2 生成回答阶段开销统计
                answer_start = time.time()

                history = short_mem.get_all()
                history_text = "\n".join([
                    f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')}\nTime: ({qa.get('timestamp', '')})"
                    for qa in history
                ])
                retrieval_text = "\n".join([
                    f"【Historical Memory】 User: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nTime:({page.get('timestamp', '')})\n"
                    for page in retrieval_result["retrieval_queue"]
                ])

                profile_obj = long_mem.get_user_profile(safe_sample_id)
                user_profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"

                background = f"【User Profile】\n{user_profile_text}\n\n"
                for kn in retrieval_result.get("long_term_knowledge", []):
                    background += f"{kn.get('knowledge', '')}\n"

                system_prompt = "You are a helpful AI assistant. Answer questions based on past long-term memory in an extremely concise manner."
                user_prompt = (
                    f"<CONTEXT>\nRecent conversation:\n{history_text}\n\n"
                    f"<MEMORY>\nRelevant past conversations:\n{retrieval_text}\n\n"
                    f"<CHARACTER TRAITS>\n{background}\n\n"
                    f"Question Date: {question_date}\n"
                    f"Question: {question}\n\n"
                    f"Please directly provide a short and concise answer without extra wordiness."
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                answer, answer_prompt_tokens, answer_completion_tokens = client.chat_completion_with_usage(
                    model="qwen3-8b", messages=messages, temperature=0.7, max_tokens=2000
                )
                answer_time = time.time() - answer_start

                # 2.3 指标计算与 Judge
                f1_score = calculate_simple_f1(answer, reference)
                is_correct = False
                if self.use_llm_judge and self.judge_client:
                    judge_res = self.judge_client.judge(
                        question=question,
                        golden_answer=reference,
                        generated_answer=answer
                    )
                    is_correct = (judge_res.lower() == "correct")

                metrics = {
                    "f1": f1_score,
                    "correct": is_correct
                }

                # 组装结果对象 (保持与其他 Memory 框架对齐的 Output JSON 规范)
                query_res = {
                    'sample_id': safe_sample_id,
                    'question_id': question_id,
                    'session_index': s_idx,
                    'question_type': question_type,
                    'difficulty': q_item.get("difficulty", "normal"),
                    'question': question,
                    'answer': answer,
                    'reference': reference,

                    # 1. 记忆构建/更新 Token 开销
                    'session_dialogue_turns_added': len(parsed_dialogs),
                    'turn_build_memory_prompt_tokens': turn_build_prompt_tokens,
                    'turn_build_memory_completion_tokens': turn_build_completion_tokens,
                    'turn_build_memory_total_tokens': turn_build_prompt_tokens + turn_build_completion_tokens,

                    'accumulated_history_turns': accumulated_history_turns,
                    'total_build_memory_prompt_tokens': total_build_prompt_tokens,
                    'total_build_memory_completion_tokens': total_build_completion_tokens,

                    # 2. 检索阶段 Token 开销 (MemoryOS Retrieve 主要基于嵌入与语义打分)
                    'retrieval_prompt_tokens': 0,
                    'retrieval_completion_tokens': 0,
                    'retrieval_total_tokens': 0,

                    # 3. 回答生成阶段 Token 开销
                    'answer_prompt_tokens': answer_prompt_tokens,
                    'answer_completion_tokens': answer_completion_tokens,
                    'answer_total_tokens': answer_prompt_tokens + answer_completion_tokens,

                    # 4. QA 环节总 Token 消耗
                    'query_total_prompt_tokens': answer_prompt_tokens,
                    'query_total_completion_tokens': answer_completion_tokens,
                    'query_total_tokens': answer_prompt_tokens + answer_completion_tokens,

                    'retrieval_time': retrieval_time,
                    'answer_time': answer_time,
                    'total_time': retrieval_time + answer_time,
                    'num_retrieved': len(retrieval_result.get("retrieval_queue", [])),
                    'metrics': metrics
                }
                sample_results.append(query_res)

        # 保存当前 Sample 的完整测试结果
            os.makedirs(save_dir, exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(sample_results, f, indent=2, ensure_ascii=False)

        avg_f1 = sum(r['metrics']['f1'] for r in sample_results) / len(sample_results) if sample_results else 0
        print(f"[{sample_idx}] Sample ID: {safe_sample_id} | Queries: {len(sample_results)} | Avg F1: {avg_f1:.3f}")
        return sample_results


# ============================================================================
# 多 Sample 并发入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run MemoryOS on HaluMem dataset')
    parser.add_argument('--dataset', type=str, default='data/HaluMem-Medium.jsonl',
                        help='Path to HaluMem jsonl file or directory')
    parser.add_argument('--llm-judge', action='store_true', help='Enable LLM-as-judge evaluation')
    parser.add_argument('--output-dir', type=str, default='./results_memoryos_halumem',
                        help='Directory to save evaluation results')
    parser.add_argument('--mem-dir', type=str, default='./mem_tmp_halumem',
                        help='Directory to save temporary memory files')
    args = parser.parse_args()

    samples = load_halumem_dataset(args.dataset)

    max_workers = 16
    if os.environ.get("DEBUG") == "1":
        max_workers = 1

    # 加载多副本 Embedding Model，避免多线程重叠调用引起冲突
    from sentence_transformers import SentenceTransformer
    if not os.path.exists(MODEL_PATH):
        model_path = "all-MiniLM-L6-v2"
    else:
        model_path = MODEL_PATH

    embedding_model = SentenceTransformer(model_path)

    def _worker(idx_sample):
        idx, sample = idx_sample
        tester = HaluMemMemoryOSTester(
            use_llm_judge=args.llm_judge,
            mem_dir=args.mem_dir
        )
        emb_model = embedding_model
        return tester.run_single_sample(sample, idx, emb_model, save_dir=args.output_dir)

    all_flattened_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, (i, s)) for i, s in enumerate(samples)]
        for future in as_completed(futures):
            try:
                sample_res = future.result()
                all_flattened_results.extend(sample_res)
            except Exception as e:
                import traceback
                print(f"Sample execution failed: {e}")
                traceback.print_exc()

    if all_flattened_results:
        avg_f1 = sum(r['metrics']['f1'] for r in all_flattened_results) / len(all_flattened_results)
        print("\n" + "=" * 80)
        print(" HaluMem + MemoryOS Test Summary ".center(80, "="))
        print(f"Total Queries Evaluated Across All Samples: {len(all_flattened_results)}")
        print(f"Overall Average F1: {avg_f1:.4f}")
        if args.llm_judge:
            accuracy = sum(1 for r in all_flattened_results if r['metrics']['correct']) / len(all_flattened_results)
            print(f"Overall LLM Judge Accuracy: {accuracy:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
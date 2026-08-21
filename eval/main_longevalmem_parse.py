import json
import os
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from short_term_memory import ShortTermMemory
from mid_term_memory import MidTermMemory
from long_term_memory import LongTermMemory
from dynamic_update import DynamicUpdate
from retrieval_and_answer import RetrievalAndAnswer
from utils import (
    OpenAIClient,
    get_timestamp,
    gpt_personality_analysis,
    gpt_update_profile,
)
from judge import AnswerJudge
from main_loco_parse import update_user_profile_from_top_segment

# 初始化 OpenAI 客户端
client = OpenAIClient(
    api_key='sk-11ce7640e46049a6977c0d96ba855ffb',
    base_url='http://127.0.0.1:30004/v1/',
    recomputation_rate=float(os.environ.get("recomputation_rate", 0.0)),
    sglang_url="http://127.0.0.1:30003/v1/completions",
    sglang_url_prefiller="http://127.0.0.1:30003/v1/completions",
)


def parse_session_dialogs(session_messages, session_date):
    """解析单次 haystack session 中的 message 数组为标准化对话对。"""
    processed = []
    i = 0
    while i < len(session_messages):
        msg = session_messages[i]
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            user_text = content
            agent_text = ""
            if i + 1 < len(session_messages) and session_messages[i + 1].get("role") == "assistant":
                agent_text = session_messages[i + 1].get("content", "")
                i += 1
            processed.append({
                "user_input": user_text,
                "agent_response": agent_text,
                "timestamp": session_date,
            })
        elif role == "assistant":
            processed.append({
                "user_input": "",
                "agent_response": content,
                "timestamp": session_date,
            })
        i += 1
    return processed


# ==========================================
# 2. 单个 Sample 构建与解答逻辑
# ==========================================

def build_memory_for_sample(sample, embedding_model):
    """
    Build 阶段：为单个 sample 顺序构建记忆系统。
    Sample 级别已经在外部并发，这里直接采用干净高效的单线程顺序解析。
    """
    sample_id = sample.get("question_id", "unknown_id")
    sessions = sample.get("haystack_sessions", [])
    dates = sample.get("haystack_dates", [])
    print(f"len(sessions): {len(sessions)}, len(dates): {len(dates)}")

    short_mem_path = f"{MEM_DIR}/{sample_id}_short_term.json"
    mid_mem_path = f"{MEM_DIR}/{sample_id}_mid_term.json"
    long_mem_path = f"{MEM_DIR}/{sample_id}_long_term.json"

    short_mem = ShortTermMemory(max_capacity=5, file_path=short_mem_path)
    mid_mem = MidTermMemory(max_capacity=2000, file_path=mid_mem_path, embedding_model=embedding_model)
    long_mem = LongTermMemory(file_path=long_mem_path, embedding_model=embedding_model)
    dynamic_updater = DynamicUpdate(
        short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client
    )

    dialogs = []
    for idx, msgs in enumerate(sessions):
        s_date = dates[idx] if idx < len(dates) else ""
        dialogs.extend(parse_session_dialogs(msgs, s_date))

    save_token_consumption = True
    if len(short_mem.memory) > 0:
        start_sign = short_mem.memory[-1]
        for start_idx, dialog in enumerate(dialogs):
            if dialog["agent_response"] == start_sign["agent_response"] and dialog["user_input"] == start_sign["user_input"] and dialog["timestamp"] == start_sign["timestamp"]:
                dialogs = dialogs[start_idx + 1:]
                save_token_consumption = False ##mengyao_debug 如果是从一半开始build/跳过build 就不写入了
                break


        # 2. 依次写入记忆系统
    for dialog in dialogs:
        short_mem.add_qa_pair(dialog)
        if short_mem.is_full():
            dynamic_updater.bulk_evict_and_update_mid_term()
        update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client, dynamic_updater)
        dynamic_updater.get_stats()

    if save_token_consumption:
        with open(f"./token_consumption/longmemeval_{sample_id}.json", "w") as f:
            json.dump(dynamic_updater.get_stats(), f)

    return short_mem, mid_mem, long_mem, dynamic_updater


def generate_system_response_longmemeval(query, query_date, short_mem, long_mem, retrieval_queue, long_knowledge,
                                         client_inst, sample_id):
    """根据检索到的长短期记忆生成最终简短回答。"""
    history = short_mem.get_all()
    history_text = "\n".join([
        f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')}\nTime: ({qa.get('timestamp', '')})"
        for qa in history
    ])

    retrieval_text = "\n".join([
        f"【Historical Memory】 User: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nTime:({page.get('timestamp', '')})\n"
        for page in retrieval_queue
    ])

    profile_obj = long_mem.get_user_profile(sample_id)
    user_profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"

    background = f"【User Profile】\n{user_profile_text}\n\n"
    for kn in long_knowledge:
        background += f"{kn['knowledge']}\n"

    system_prompt = (
        "You are a helpful AI assistant. Answer questions based on past long-term memory in an extremely concise manner."
    )

    user_prompt = (
        f"<CONTEXT>\nRecent conversation:\n{history_text}\n\n"
        f"<MEMORY>\nRelevant past conversations:\n{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\n{background}\n\n"
        f"Question Date: {query_date}\n"
        f"Question: {query}\n\n"
        f"Please directly provide a short and concise answer without extra wordiness."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response, prompt_tokens, completion_tokens = client_inst.chat_completion_with_usage(
        model="qwen3-8b", messages=messages, temperature=0.7, max_tokens=2000
    )
    return response, system_prompt, user_prompt, prompt_tokens, completion_tokens


def answer_single_sample(sample, embedding_model=None):
    """Answer 阶段：检索并解答。"""
    sample_id = sample.get("question_id", "unknown_id")
    question = sample.get("question", "")
    question_date = sample.get("question_date", "")
    golden_answer = sample.get("answer", "")
    question_type = sample.get("question_type", "")

    short_mem_path = f"{MEM_DIR}/{sample_id}_short_term.json"
    mid_mem_path = f"{MEM_DIR}/{sample_id}_mid_term.json"
    long_mem_path = f"{MEM_DIR}/{sample_id}_long_term.json"

    short_mem = ShortTermMemory(max_capacity=5, file_path=short_mem_path)
    mid_mem = MidTermMemory(max_capacity=2000, file_path=mid_mem_path, embedding_model=embedding_model)
    long_mem = LongTermMemory(file_path=long_mem_path, embedding_model=embedding_model)
    dynamic_updater = DynamicUpdate(
        short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client
    )

    retrieval_system = RetrievalAndAnswer(
        short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10
    )

    retrieval_result = retrieval_system.retrieve(
        question,
        segment_threshold=0.1,
        page_threshold=0.1,
        knowledge_threshold=0.1,
        client=client,
        embedding_model=embedding_model
    )

    sys_answer, sys_prompt, user_prompt, p_tokens, c_tokens = generate_system_response_longmemeval(
        question,
        question_date,
        short_mem,
        long_mem,
        retrieval_result["retrieval_queue"],
        retrieval_result["long_term_knowledge"],
        client,
        sample_id
    )

    aj = AnswerJudge(
        api_key="sk-11ce7640e46049a6977c0d96ba855ffb",
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-v3.2"
    )
    judge_res = aj.judge(
        question=question,
        golden_answer=golden_answer,
        generated_answer=sys_answer,
    )

    return {
        "question_id": sample_id,
        "question_type": question_type,
        "question": question,
        "question_date": question_date,
        "golden_answer": golden_answer,
        "system_answer": sys_answer,
        "correct": judge_res.lower() == "correct",
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "timestamp": get_timestamp()
    }


def process_single_longmemeval_sample(sample, embedding_model):
    """
    单个 Sample 的执行流水线（运行于独立的子线程中）：
    1. Build 阶段：构建属于该 sample_id 的记忆（多线程无锁运行）
    2. Answer 阶段：检索解答并进行 Judge 评估
    3. 返回完整结果包给主线程
    """
    sample_id = sample.get("question_id", "unknown_id")
    try:
        # 1. 构建 Memory
        build_memory_for_sample(sample, embedding_model)

        # 2. 检索并解答
        res = answer_single_sample(sample, embedding_model=embedding_model)

        print(f"✅ Sample [{sample_id}] 处理完成 | 预测: '{res['system_answer']}' | 正确: {res['correct']}")
        return res
    except Exception as e:
        print(f"❌ Sample [{sample_id}] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# 3. 多 Sample 并发主控入口
# ==========================================

def main_parallel_longmemeval(
        data_path="longmemeval.json",
        output_file="./results/longmemeval_result.json",
        sample_max_workers=8
):
    """
    主控函数：按 Sample 并发调度
    """
    os.makedirs(MEM_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载 LongMemEval 数据集，包含 {len(dataset)} 个 Sample。")
    except Exception as e:
        print(f"读取数据集失败: {e}")
        return

    # 预先加载 Embedding 模型
    from sentence_transformers import SentenceTransformer
    model_path = "/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2"
    if not os.path.exists(model_path):
        model_path = "all-MiniLM-L6-v2"
    embedding_models = []
    for _ in range(sample_max_workers):
        embedding_models.append(SentenceTransformer(model_path))

    results = []
    total_samples = len(dataset)
    completed_count = 0

    # 多 Sample 并发池
    with ThreadPoolExecutor(max_workers=sample_max_workers) as executor:
        future_to_sample = {
            executor.submit(
                process_single_longmemeval_sample,
                sample,
                embedding_models[i % sample_max_workers],
            ): sample.get("question_id", f"idx_{i}")
            for i, sample in enumerate(dataset)
        }

        # 主线程接收各 Sample 返回的结果并写入文件（绝无写冲突）
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            completed_count += 1

            res = future.result()
            if res is not None:
                results.append(res)

                # 实时更新落盘
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"💾 进度 [{completed_count}/{total_samples}]: Sample [{sample_id}] 已写入 {output_file}")

    print(f"\n🎉 处理完毕！共计完成 {len(results)}/{total_samples} 条数据，结果已保存至 {output_file}")


##mengyao_debug LLM配置： 搜索 http://127.0.0.1:30004 即可
if __name__ == "__main__":
    MAX_WORKERS = 32
    if os.environ.get("DEBUG") == "1":
        MAX_WORKERS = 1
    MEM_DIR = "mem_tmp_longmemeval"
    main_parallel_longmemeval(
        data_path="data/longmemeval_mixed.json",
        output_file="./results/longmemeval_result.json",
        sample_max_workers=MAX_WORKERS,  # 同时并发处理 8 个 Sample
    )
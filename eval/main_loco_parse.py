import json
from datetime import datetime, timedelta
from short_term_memory import ShortTermMemory
from mid_term_memory import MidTermMemory
from long_term_memory import LongTermMemory
from dynamic_update import DynamicUpdate
from retrieval_and_answer import RetrievalAndAnswer
from utils import OpenAIClient, gpt_generate_answer, gpt_extract_theme, gpt_update_profile, gpt_generate_multi_summary, get_timestamp, llm_extract_keywords, gpt_personality_analysis
import re
from judge import AnswerJudge
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# import tiktoken
import os
import argparse

total_tokens = 0
num_samples=0

# Heat threshold
H_THRESHOLD = 5.0
all_memory_summarize_percentage = []
all_history_len = []
all_history_stored_len = []
all_answer_time = []
os.environ["OMP_NUM_THREADS"] = "4"

def update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client, dynamic_updater: DynamicUpdate):
    """
    Update user profile if heat exceeds threshold and extract assistant knowledge.
    """
    if not mid_mem.heap:
        return
    
    neg_heat, sid = mid_mem.heap[0]
    mid_mem.rebuild_heap()
    current_heat = -neg_heat
    
    if current_heat >= H_THRESHOLD:
        session = mid_mem.sessions.get(sid)
        if not session:
            return
        
        un_analyzed = [p for p in session["details"] if not p.get("analyzed", False)]
        if un_analyzed:
            print(f"Updating user profile: Segment {sid} heat {current_heat:.2f} exceeds threshold, starting profile update...")
            
            old_profile = long_mem.get_raw_user_profile(sample_id)
            
            result, prompt_tokens, completion_tokens, fusionrag_stats_list = gpt_personality_analysis(un_analyzed, client)
            dynamic_updater.calls += 1
            new_profile = result["profile"]
            new_private = result["private"]
            assistant_knowledge = result["assistant_knowledge"]
            
            if old_profile:
                updated_profile, prompt_tokens_1, completion_tokens_1, fusionrag_stats = gpt_update_profile(old_profile, new_profile, client)
                fusionrag_stats["reuse_type"] = "reuse_decode"
                prompt_tokens += prompt_tokens_1
                completion_tokens += completion_tokens_1
                dynamic_updater.fusionrag_stats.append(fusionrag_stats)
                dynamic_updater.calls += 1
            else:
                updated_profile = new_profile
                
            long_mem.update_user_profile(sample_id, updated_profile)

            dynamic_updater.prompt_tokens += prompt_tokens
            dynamic_updater.completion_tokens += completion_tokens
            dynamic_updater.fusionrag_stats.extend(fusionrag_stats_list)
            
            # 修改点：拆分 new_private 并逐个存储
            if new_private and new_private != "- None":
                # 按行拆分，过滤空行和非事实行（如 "【User Data】" 或注释）
                facts = [line.strip() for line in new_private.split("\n")]
                for fact in facts:
                    long_mem.add_knowledge(fact)  # 逐条添加
            
            if assistant_knowledge and assistant_knowledge != "None":
                long_mem.add_assistant_knowledge(assistant_knowledge)
            
            for p in session["details"]:
                p["analyzed"] = True
            session["N_visit"] = 0
            session["L_interaction"] = 0
            session["R_recency"] = 1.0
            session["H_segment"] = 0.0
            session["last_visit_time"] = get_timestamp()
            mid_mem.rebuild_heap()
            mid_mem.save()
            print(f"Update complete: Segment {sid} heat has been reset.")

def generate_system_response_with_meta(query, short_mem, long_mem, retrieval_queue, long_konwledge, client, sample_id, speaker_a, speaker_b, meta_data):
    """
    Generate system response with speaker roles clearly defined.
    """
    history = short_mem.get_all()
    history_text_list = [
        f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')}\nTime: ({qa.get('timestamp', '')})" 
        for qa in history
    ]
    history_text = "\n".join(history_text_list)
    
    retrieval_text_list = [
        f"【Historical Memory】 {speaker_a}: {page.get('user_input', '')}\n{speaker_b}: {page.get('agent_response', '')}\nTime:({page.get('timestamp', '')})\nConversation chain overview:({page.get('meta_info', '')})\n" 
        for page in retrieval_queue
    ]
    retrieval_text = "\n".join(retrieval_text_list)
    
    profile_obj = long_mem.get_user_profile(sample_id)
    user_profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"
    
    background = f"【User Profile】\n{user_profile_text}\n\n"
    for kn in long_konwledge:
        background += f"{kn['knowledge']}\n"
    background = re.sub(r'(?i)\buser\b', speaker_a, background)
    background= re.sub(r'(?i)\bassistant\b', speaker_b, background)
    assistant_knowledge = long_mem.get_assistant_knowledge()
    assistant_knowledge_text = "【Assistant Knowledge】\n"
    for ak in assistant_knowledge:
        assistant_knowledge_text += f"- {ak['knowledge']} ({ak['timestamp']})\n"
    #meta_data_text = f"【Conversation Meta Data】\n{json.dumps(meta_data, ensure_ascii=False, indent=2)}\n\n"
    assistant_knowledge_text = re.sub(r'\bI\b', speaker_b, assistant_knowledge_text)
    
    system_prompt = (
        f"You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}. "
        f"Here are some of your character traits and knowledge:\n{assistant_knowledge_text}\n"
        f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
    )

    user_prompt = (
        f"<CONTEXT>\n"
        f"Recent conversation between {speaker_a} and {speaker_b}:\n"
        f"{history_text}\n\n"
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\n"
        f"Characteristics of {speaker_a}:\n"
        f"{background}\n\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )

    time_start = time.time()

    prefix = "Here are the CONTEXT, MEMORY and CHARACTER TRAITS."
    question = (
        f"the question is: {query}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )
    fusionrag_list = []
    history_text_list[0] = f"<CONTEXT>\nRecent conversation between" + history_text_list[0]
    retrieval_text_list[0] = f"<MEMORY>\nRelevant past conversations:\n" + retrieval_text_list[0]
    background = f"<CHARACTER TRAITS>\nCharacteristics of {speaker_a}:\n" + background
    fusionrag_list.extend(history_text_list)
    fusionrag_list.extend(retrieval_text_list)
    fusionrag_list.append(background)
    question_origin = {
        "system_prompt": system_prompt,
        "prefix": prefix,
        "query_prompt": question,
        "question" : question,
        "fusionrag_list": fusionrag_list,
    }
    if os.getenv("DUMP_QUESTIONS", "").lower() == "true":
        response = "dummy"
        prompt_tokens = completion_tokens = 0
    else:
        if os.environ.get("FUSIONRAG", "").lower() == "true":
            fusionrag_list = fusionrag_list[:12]
            response, prompt_tokens, completion_tokens, chat_completion_fusionrag = client.chat_completion_fusionrag(
                model="kimi-k2.6",
                system_prompt=system_prompt,
                prefix=prefix,
                fusionrag_cache_list=fusionrag_list,
                query_prompt=question,
            )
        else:

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response, prompt_tokens, completion_tokens, _ = client.chat_completion_with_usage(model=LLM_MODEL, messages=messages, temperature=0.7, max_tokens=2000)
    time_answer = time.time() - time_start
    return response, system_prompt, user_prompt, prompt_tokens, completion_tokens, time_answer, question_origin

def process_conversation(conversation_data):
    """
    Process conversation data from locomo10 format into memory system format.
    Handles both text-only and image-containing messages.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]
    
    # Find all session keys
    session_keys = [key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")]
    
    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        
        for dialog in conversation_data[session_key]:
            speaker = dialog["speaker"]
            text = dialog["text"]
            
            # Handle image content if present
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"
            
            # Alternate between speakers as user and assistant
            if speaker == speaker_a:
                processed.append({
                    "user_input": text,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": text,
                        "timestamp": timestamp
                    })
    
    return processed


from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os


def filter_qa(qa_list: list):
    return qa_list
    """过滤 QA 列表"""
    try:
        with open(
            "./result_simplerag_fusion_rag_1.0_Qwen2.5-3B-Instruct_qwen2.5-7B_retrieve_5.json"
        ) as f:
            simplerag_res = json.load(f)
            target_ids = {0, 2, 3, 4, 6, 7, 9}

            simplerag_res_q = {
                x["question"]: x["sample_id"]
                for x in simplerag_res
                if x["sample_id"] in target_ids
            }
            qa_list = [
                qa for qa in qa_list if qa["question"] in simplerag_res_q
            ]
    except FileNotFoundError:
        print("警告：未找到 filter_qa 对应的 JSON 文件，保留原始 QA 列表")
    return qa_list


def process_single_qa_worker(
    qa_idx, qa, sample_id, speaker_a, speaker_b, client, embedding_model
):
    """单个 QA 处理任务的通用 Worker 函数

    处理前重新根据 sample_id 从磁盘装载记忆组件，确保线程间隔离
    """
    question = qa.get("question")
    original_answer = qa.get("answer", "")
    category = qa.get("category")
    evidence = qa.get("evidence", "")

    if not original_answer:
        original_answer = qa.get("adversarial_answer", "")

    # 定义当前 sample 的记忆文件路径
    short_mem_path = f"{mem_dir}/{sample_id}_short_term.json"
    mid_mem_path = f"{mem_dir}/{sample_id}_mid_term.json"
    long_mem_path = f"{mem_dir}/{sample_id}_long_term.json"

    # 重新装载记忆系统组件，保证线程独立隔离
    local_short_mem = ShortTermMemory(
        max_capacity=5, file_path=short_mem_path
    )
    local_mid_mem = MidTermMemory(
        max_capacity=2000, file_path=mid_mem_path,
        client=client,
        embedding_model=embedding_model
    )
    local_long_mem = LongTermMemory(file_path=long_mem_path, embedding_model=embedding_model)

    local_dynamic_updater = DynamicUpdate(
        local_short_mem,
        local_mid_mem,
        local_long_mem,
        topic_similarity_threshold=0.6,
        client=client,
    )
    local_retrieval_system = RetrievalAndAnswer(
        local_short_mem,
        local_mid_mem,
        local_long_mem,
        local_dynamic_updater,
        queue_capacity=10,
    )

    # 检索与答案生成
    retrieval_result = local_retrieval_system.retrieve(
        question,
        segment_threshold=0.1,
        page_threshold=0.1,
        knowledge_threshold=0.1,
        client=client,
        embedding_model=embedding_model
    )


    meta_data = {
        "sample_id": sample_id,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "category": category,
        "evidence": evidence,
    }

    system_answer, system_prompt, user_prompt, prompt_tokens, completion_tokens, time_answer, question_origin = (
        generate_system_response_with_meta(
            question,
            local_short_mem,
            local_long_mem,
            retrieval_result["retrieval_queue"],
            retrieval_result["long_term_knowledge"],
            client,
            sample_id,
            speaker_a,
            speaker_b,
            meta_data,
        )
    )

    all_answer_time.append(time_answer)
    print(f"average qa time={sum(all_answer_time)/len(all_answer_time)}")

    print(f"\033[93muser_prompt = {user_prompt}\033[0m")

    if os.getenv("DUMP_QUESTIONS", "").lower() == "true":
        result = "dummpy"
    else:
        aj = AnswerJudge(
            api_key="sk-11ce7640e46049a6977c0d96ba855ffb",
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="deepseek-v3.2"
        )
        result = aj.judge(
            question=question,
            golden_answer=original_answer,
            generated_answer=system_answer,
        )


    return qa_idx, {
        "sample_id": sample_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "question": question,
        "system_answer": system_answer,
        "original_answer": original_answer,
        "category": category,
        "evidence": evidence,
        "timestamp": get_timestamp(),
        "time_answer": time_answer,
        "question_origin": question_origin,
        "correct": result.lower() == "correct",
    }


def process_qa_in_parallel(
    qa_pairs, sample_id, speaker_a, speaker_b, client, qa_max_workers=5, embedding_model=None,
    output_file=None, result_lock=None
):
    """辅助函数：针对确定的 QA 列表开启多线程并发处理，并恢复原始顺序"""
    qa_results_indexed = []

    def append_result_to_file(result):
        """将单个结果追加到输出文件，使用锁保证线程安全"""
        if not output_file:
            return
        try:
            # 如果提供了锁，则在锁内执行整个读取-追加-写入序列
            if result_lock:
                with result_lock:
                    # 读取现有文件内容
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                existing_data = json.loads(content)
                            else:
                                existing_data = []
                    else:
                        existing_data = []

                    # 追加新结果
                    existing_data.append(result)

                    # 写入文件
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
            else:
                # 没有锁，直接执行（可能存在竞争）
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                        else:
                            existing_data = []
                else:
                    existing_data = []

                existing_data.append(result)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

            print(f"样本 {sample_id}: 问题 '{result['question'][:50]}...' 结果已写入文件")
        except Exception as e:
            print(f"样本 {sample_id}: 写入结果文件时出错: {e}")

    with ThreadPoolExecutor(max_workers=qa_max_workers) as qa_executor:
        futures = []

        for qa_idx, qa in enumerate(qa_pairs):
            worker_id = qa_idx % qa_max_workers

            future = qa_executor.submit(
                process_single_qa_worker,
                qa_idx,
                qa,
                sample_id,
                speaker_a,
                speaker_b,
                client,
                embedding_model,
            )

            futures.append(future)

        for future in as_completed(futures):
            try:
                qa_idx, result = future.result()
                qa_results_indexed.append((qa_idx, result))
                # 每完成一个 QA 就写入文件
                append_result_to_file(result)
            except Exception as e:
                print(
                    f"样本 {sample_id} 处理 QA 时出错: {e}"
                )
                import traceback
                traceback.print_exc()

    # 按原始 qa_idx 排序保持顺序一致
    qa_results_indexed.sort(key=lambda x: x[0])
    return [res for _, res in qa_results_indexed]


def process_single_sample(sample, client, embedding_model, qa_max_workers=5, output_file=None, result_lock=None):
    """单个样本的完整处理逻辑（提供给 main_parallel 调用）：

    1. 顺序构建/更新该 sample 的记忆
    2. 并发处理该 sample 下的所有 QA 问答
    """
    sample_id = sample.get("sample_id", "unknown_sample")
    conversation_data = sample.get("conversation", {})
    qa_pairs = sample.get("qa", [])

    processed_dialogs = process_conversation(conversation_data)
    processed_dialogs_text = " ".join([x["user_input"] + " " + x["agent_response"] for x in processed_dialogs])
    if not processed_dialogs:
        print(f"样本 {sample_id} 没有有效的对话数据，跳过")
        return []

    token_consumption_path = f"./{token_consumption_dir}/locomo_{sample_id}.json"
    if os.path.exists(token_consumption_path):
        print(f"[{sample_id}] already built, skip.")
    else:
        short_mem_path = f"{mem_dir}/{sample_id}_short_term.json"
        mid_mem_path = f"{mem_dir}/{sample_id}_mid_term.json"
        long_mem_path = f"{mem_dir}/{sample_id}_long_term.json"
        for path in [short_mem_path, mid_mem_path, long_mem_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[{sample_id}] removed old memory file: {path}")

        speaker_a = conversation_data.get("speaker_a")
        speaker_b = conversation_data.get("speaker_b")

        # 1. 初始化记忆模块并顺序写入对话历史（写入过程需保持时序）
        short_mem = ShortTermMemory(
            max_capacity=5,
            file_path=short_mem_path,
        )
        mid_mem = MidTermMemory(
            max_capacity=2000,
            file_path=mid_mem_path,
            embedding_model=embedding_model,
            client=client,
        )
        long_mem = LongTermMemory(
            file_path=long_mem_path,
            embedding_model=embedding_model
        )
        dynamic_updater = DynamicUpdate(
            short_mem,
            mid_mem,
            long_mem,
            topic_similarity_threshold=0.6,
            client=client,
        )

        for d_idx, dialog in enumerate(processed_dialogs):
            short_mem.add_qa_pair(dialog)
            if short_mem.is_full():
                dynamic_updater.bulk_evict_and_update_mid_term()
            update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client, dynamic_updater)
            dynamic_updater.get_stats()
            print(f"sample [{sample_id}] finished {d_idx+1} / {len(processed_dialogs)}")

        history_stored_text = ""
        memory_short = " ".join([m["user_input"] + " " + m["agent_response"] for m in short_mem.memory])
        memory_mid = " ".join([v["summary"] for k, v in mid_mem.sessions.items()])
        for _, session in mid_mem.sessions.items():
            for detail in session["details"]:
                if detail["meta_info"] not in memory_mid:
                    memory_mid += detail["meta_info"]
                if detail["user_input"] not in history_stored_text:
                    history_stored_text += detail["user_input"]
                if detail["agent_response"] not in history_stored_text:
                    history_stored_text += detail["agent_response"]

        memory_long = " ".join([v["data"] for k, v in long_mem.user_profiles.items()])
        all_memory = memory_short + memory_mid + memory_long

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/mnt/qjhs-sh-lab-01/models/Qwen3-8B", trust_remote_code=True)
        all_summary = tokenizer.encode(all_memory, add_special_tokens=True)
        all_history = tokenizer.encode(processed_dialogs_text, add_special_tokens=True)
        history_stored = tokenizer.encode(history_stored_text, add_special_tokens=True) ## history_stored 是存储起来的对话，这里就是检验一下是否存储了全部的对话
        all_memory_summarize_percentage.append(len(all_summary)/len(all_history))
        all_history_len.append(len(all_history))
        all_history_stored_len.append(len(history_stored))

        print(f"summarize percentage: {sum(all_memory_summarize_percentage)/len(all_memory_summarize_percentage)} all_history_len={sum(all_history_len)} all_history_stored_len={sum(all_history_stored_len)}")

        with open(token_consumption_path, "w") as f:
            json.dump(dynamic_updater.get_stats(), f, indent=4)


    # 如果提供了输出文件，则加载已有结果，跳过已处理的问题
    if output_file and os.path.exists(output_file):
        try:
            # 如果有锁，则在锁内读取文件，避免并发读写冲突
            if result_lock:
                with result_lock:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
            else:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

            if content:
                existing_results = json.loads(content)
                # 提取当前 sample_id 下已经处理过的问题集合
                processed_questions = {res['question'] for res in existing_results
                                      if res.get('sample_id') == sample_id}
                # 过滤掉已处理的问题
                initial_len = len(qa_pairs)
                qa_pairs = [qa for qa in qa_pairs if qa.get('question') not in processed_questions]
                print(f"样本 {sample_id}: 跳过 {initial_len - len(qa_pairs)} 个已处理的问题，剩余 {len(qa_pairs)} 个问题待处理")
            else:
                print(f"样本 {sample_id}: 结果文件为空，将处理所有问题")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"样本 {sample_id}: 读取结果文件时出错 {output_file}: {e}，将处理所有问题")


    # 2. 过滤并并发处理 QA 对
    filtered_qa_pairs = filter_qa(qa_pairs)
    if not filtered_qa_pairs:
        return []

    sample_results = process_qa_in_parallel(
        filtered_qa_pairs,
        sample_id,
        speaker_a,
        speaker_b,
        client,
        qa_max_workers=qa_max_workers,
        embedding_model=embedding_model,
        output_file=output_file,
        result_lock=result_lock
    )
    print(
        f"样本 {sample_id} 处理完成，共并发完成 {len(sample_results)} 个 QA 对"
    )
    return sample_results


def main_parallel(sample_max_workers=5, qa_max_workers=5, output_file=""):
    """多样本并发处理 (Main Parallel)，且每个 Sample 内部的 QA 也是并发处理"""
    print(
        f"开始运行 [main_parallel]: 多样本并发 (workers={sample_max_workers}) + QA并发 (workers={qa_max_workers})..."
    )

    os.makedirs(mem_dir, exist_ok=True)

    try:
        with open("locomo10.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except FileNotFoundError:
        print("错误：找不到 locomo10.json 文件，请确保文件在当前目录中")
        return
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return

    results = []
    completed_samples = 0
    total_samples = len(dataset)
    if os.environ.get("DEBUG") == "1":
        dataset = dataset[:1]
        dataset[0]['qa'] = dataset[0]['qa'][:1] ##mengyao_debug

    from sentence_transformers import SentenceTransformer
    model_path = "/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2"
    if not os.path.exists(model_path):
        model_path = "all-MiniLM-L6-v2"

    device_ = "cuda:1"
    embedding_model = SentenceTransformer(model_path, device=device_)

    # 创建锁用于保护结果文件写入
    result_lock = threading.Lock()

    # 样本级 ThreadPoolExecutor 并发处理
    with ThreadPoolExecutor(max_workers=sample_max_workers) as executor:
        future_to_sample_id = {
            executor.submit(
                process_single_sample, sample, client, embedding_model, qa_max_workers,
                output_file=output_file, result_lock=result_lock
            ): sample.get("sample_id", f"sample_{idx+1}")
            for idx, sample in enumerate(dataset)
        }

        # 主线程统一收集计算结果并落盘写文件
        for future in as_completed(future_to_sample_id):
            sample_id = future_to_sample_id[future]
            completed_samples += 1

            try:
                sample_results = future.result()
                if sample_results:
                    results.extend(sample_results)
                    print(
                        f"进度 [{completed_samples}/{total_samples}]: 样本 {sample_id} 处理完成，累计 {len(results)} 条结果"
                    )

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"样本 {sample_id} 在子线程处理过程中发生错误：{e}")

    print(
        f"[main_parallel] 全量数据并发处理完成！最终结果保存在 {output_file}，共计 {len(results)} 条记录。"
    )


if __name__ == "__main__":
    # LLM_MODEL = "Kimi-K2.6"

    parser = argparse.ArgumentParser()
    parser.add_argument('--LLM_MODEL', type=str,
                        help='llm model')
    parser.add_argument('--API_BASE_URL', type=str,
                        help='llm model api')

    args = parser.parse_args()
    LLM_MODEL = args.LLM_MODEL
    API_BASE_URL = args.API_BASE_URL

    # Initialize OpenAI client
    client = OpenAIClient(
        api_key='sk-11ce7640e46049a6977c0d96ba855ffb',
        # base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        base_url=API_BASE_URL,
        recomputation_rate=float(os.environ.get("recomputation_rate")),
        sglang_url=f"{API_BASE_URL}/completions",
        sglang_url_prefiller=f"{API_BASE_URL}/completions"
    )

    token_consumption_dir = "./token_consumption"
    mem_dir = "mem_tmp_loco"
    result_dir = "./results"
    if os.getenv("DUMP_QUESTIONS", "").lower() == "true":
        result_dir += "_dump_questions"
    fusionrag_tag = os.environ.get("FUSIONRAG", "false").lower()
    result_file = f"{result_dir}/locomo_result.json"


    if LLM_MODEL.lower() != "qwen3-8b":
        mem_dir += f"_{LLM_MODEL}"
        token_consumption_dir += f"_{LLM_MODEL}"
        result_dir += f"_{LLM_MODEL}"

    if fusionrag_tag == "true":
        mem_dir += "_fusionrag"
        token_consumption_dir += "_fusionrag"
        result_dir += f"_fusionrag"

    result_file = f"{result_dir}/locomo_result.json"

    os.makedirs(mem_dir, exist_ok=True)
    os.makedirs(token_consumption_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"mem_dir: {mem_dir}")
    print(f"token_consumption_dir: {token_consumption_dir}")
    print(f"result_dir: {result_dir}")
    print(f"result file: {result_file}")

    MAX_WORKERS = 10
    qa_max_workers = 32
    if os.environ.get("DEBUG") == "1":
        MAX_WORKERS = 1
        qa_max_workers = 1

    main_parallel(sample_max_workers=MAX_WORKERS, qa_max_workers=qa_max_workers, output_file=result_file)
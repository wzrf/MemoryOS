import time
import uuid
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sglang_kvcache import run_one_question_sglang
from FusionRAG.run_question import FusionRAGModel
import os
import threading
embedding_lock = threading.Lock()


gpt_client = OpenAI(
        api_key='sk-11ce7640e46049a6977c0d96ba855ffb',
        base_url = 'http://127.0.0.1:30004/v1/'  ## qwen3
        # base_url = 'http://127.0.0.1:30003/v1/' ## kimi
)
def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    model_name_ = "/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2"
    if os.path.exists(model_name_):
        model_name = model_name_
    model = SentenceTransformer(model_name)
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return embedding

def get_embedding_with_model(text, model):
    with embedding_lock:
        embedding = model.encode([text], convert_to_numpy=True)[0]
        return embedding

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

class OpenAIClient:
    def __init__(self, api_key, base_url, recomputation_rate: float, sglang_url_prefiller: str, sglang_url: str):
        self.api_key = api_key
        self.base_url = base_url
        openai.api_key = self.api_key
        openai.api_base = self.base_url
        self.sglang_url_prefiller = sglang_url_prefiller
        self.sglang_url = sglang_url
        self.recomputation_rate = recomputation_rate
        draft_model_path = os.environ.get("DRAFT_MODEL_PATH", "")
        if draft_model_path == "":
            draft_model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
        self.fusion_rag_model = FusionRAGModel(
            model_path='',
            use_multi_gpu=True,
            model_type="qwen3",
            model_name="Qwen3-32B",
            draft_model_type="qwen",
            draft_model_name="qwen2.5-3b",
            preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
            draft_model_path=draft_model_path,
            draft_model_url="http://127.0.0.1:30005/v1/completions",
            apikey="xxx",
            use_local_draft_model=False,
        )

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        model = "qwen3-8b"
        # print("调用 GPT 接口，模型:", model)
        response = gpt_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # extra_body={"enable_thinking":False}
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False
                }
            }
        )
        content = response.choices[0].message.content.strip()
        return content

    def chat_completion_with_usage(self, model, messages, temperature=0.7, max_tokens=2000):
        model = "qwen3-8b"
        # print("调用 GPT 接口，模型:", model)
        response = gpt_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False
                }
            }
        )
        content = response.choices[0].message.content.strip()
        return content, response.usage.prompt_tokens, response.usage.completion_tokens

    def chat_completion_fusionrag(self, model, system_prompt: str, prefix: str, fusionrag_cache_list: list[str], query_prompt: str, temperature=0.7, max_tokens=2000):
        if "kimi" in model.lower():
            template = {
                "DEFAULT_SYSTEM_PROMPT": f"""<|im_system|>system<|im_middle|>\n{system_prompt}\n{prefix}""",
                "USER_PROMPT": f"""<|im_end|><|im_user|>user<|im_middle|>{query_prompt}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think> Answer:"""
            }
        else:  ## default: qwen
            template = {
                "DEFAULT_SYSTEM_PROMPT": f"""<|im_start|>system\n{system_prompt}\n{prefix}""",
                "USER_PROMPT": f"""<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer: </think>"""
            }
        print("调用 fusionrag GPT 接口，模型:", model)

        recompute_tokens, recompute_tokens_list, retrieved_docs, recompute_rate, sorted_doc_index, sorted_doc_index_before = self.fusion_rag_model.draft_one_question(
            template["DEFAULT_SYSTEM_PROMPT"],  ## DEFAULT_SYSTEM_PROMPT
            fusionrag_cache_list,
            template["USER_PROMPT"],
            self.recomputation_rate,
            "",
            False,
            False,
            [],
            False,
            False,  ## if do preprocess
            False,
            True
        )

        content, usage, top_logprobs, real_recomputation_rate = run_one_question_sglang(
            DEFAULT_SYSTEM_PROMPT=template["DEFAULT_SYSTEM_PROMPT"],
            USER_PROMPT=template["USER_PROMPT"],
            MODEL=model,
            retrived_docs=fusionrag_cache_list,
            max_tokens=max_tokens,  ## max tokens.
            retrived_docs_relevant_docs=[],
            recompute_tokens=recompute_tokens,
            recompute_tokens_list=recompute_tokens_list,
            max_workers=1,  ## max_workers.
            recomputation_rate=self.recomputation_rate,
            model_use=model,
            endpoint_url=self.sglang_url,
            prefiller_endpoint_url=self.sglang_url_prefiller,
            method_keyword="",
        )

        return content.strip(), usage['prompt_tokens'], usage['completion_tokens']

def gpt_generate_answer(prompt, messages, client):
    return client.chat_completion_with_usage(model="qwen3-8b", messages=messages, temperature=0.7, max_tokens=2000)

def gpt_generate_answer_fusionrag(system_prompt: str, prefix: str, fusionrag_cache_list: list[str], query_prompt: str, client):
    return client.chat_completion_fusionrag(model="qwen3-8b",
                                            system_prompt=system_prompt,
                                            prefix=prefix,
                                            fusionrag_cache_list=fusionrag_cache_list,
                                            query_prompt=query_prompt,
                                            temperature=0.7,
                                            max_tokens=2000)

def analyze_assistant_knowledge(dialogs, client):
    """
    Analyzes conversations to extract knowledge or identity traits about the assistant.
    Returns: {"assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAI: {d['agent_response']}\nTime:{d['timestamp']}\n" for d in dialogs])

    prefix = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That’s interesting. I didn’t know you could do that.
   AI: I’m glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
"""
    prompt = prefix + conversation

    messages = [
        {
            "role": "system",
            "content": """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None".""" 
        },
        {"role": "user", "content": prompt}
    ]

    system_prompt = """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None"."""

    fusionrag_cache_list = [f"User: {d['user_input']}\nAI: {d['agent_response']}\nTime:{d['timestamp']}\n" for d in dialogs]

    query_prompt = "Analyze the conversation and extract any fact or identity traits about the assistant."

    print("Analyzing assistant knowledge...")
    if os.environ.get("FUSIONRAG", "false").lower() == "true":
        result, prompt_tokens, completion_tokens = gpt_generate_answer_fusionrag(system_prompt=system_prompt, prefix=prefix, fusionrag_cache_list=fusionrag_cache_list, query_prompt=query_prompt, client=client)
    else:
        result, prompt_tokens, completion_tokens = gpt_generate_answer(prompt, messages, client)
    
    # Parse output
    assistant_knowledge = result.replace("【Assistant Knowledge】", "").strip()
    return {"assistant_knowledge": assistant_knowledge}, prompt_tokens, completion_tokens

def gpt_summarize(dialogs, client):
    prompt = "Please generate a topic summary based on the following conversation：\n"
    for d in dialogs:
        prompt += f"user: {d.get('user_input','')}\nassiant: {d.get('agent_response','')}\n"
    prompt += "\nSubject Summary："
    messages = [
        {"role": "system", "content": "You are an expert in summarizing dialogue topics, please generate a concise and precise summary."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 生成主题摘要...")
    return gpt_generate_answer(prompt, messages, client)

def gpt_generate_multi_summary(text, client):
    """
    调用 LLM 生成多子主题摘要，返回格式示例如下：
    {
      "input": "对话文本",
      "summaries": [
         {"theme": "出差", "keywords": ["出差", "行程", "工作"], "content": "用户提到出差相关的困扰"},
         {"theme": "健康", "keywords": ["感冒", "难受", "生病"], "content": "用户反馈感冒导致身体不适"}
      ]
    }
    """
    prompt = ("Please analyze the following dialogue and generate multiple subtopic summaries (if applicable), with a maximum of two themes.\n"
              "Each summary should include the subtopic name, keywords (separated by commas), and the summary text, formatted as a JSON array, with an example format as follows:\n"
              "[\n  {\"theme\": \"Business trip\", \"keywords\": [\"Business trip\", \"Itinerary\", \"Work\"], \"content\": \" User mentioned the troubles related to business trips.\"},\n  {\"theme\": \"Health\", \"keywords\": [\"Cold\", \"Uncomfortable\", \"Sick\"], \"content\": \"User reported feeling unwell due to a cold.\"}\n]\n"
              "Please directly output the JSON array, without adding any other content.\n\Conversation content:\n" + text)
    messages = [
        {"role": "system", "content": "You are an expert in analyzing dialogue topics. No more than two topics."},
        {"role": "user", "content": prompt}
    ]
    system_prompt = "You are an expert in analyzing dialogue topics. No more than two topics."
    query_prompt = ("Please analyze the following dialogue and generate multiple subtopic summaries (if applicable), with a maximum of two themes.\n"
              "Each summary should include the subtopic name, keywords (separated by commas), and the summary text, formatted as a JSON array, with an example format as follows:\n"
              "[\n  {\"theme\": \"Business trip\", \"keywords\": [\"Business trip\", \"Itinerary\", \"Work\"], \"content\": \" User mentioned the troubles related to business trips.\"},\n  {\"theme\": \"Health\", \"keywords\": [\"Cold\", \"Uncomfortable\", \"Sick\"], \"content\": \"User reported feeling unwell due to a cold.\"}\n]\n"
              "Please directly output the JSON array, without adding any other content.\n\Conversation content:\n")
    fusionrag_prompt_list = [text]
    prefix = " "
    print("调用 GPT 生成多子主题摘要...")

    ##mengyao_debug fusionrag_bad_case
    if os.environ.get("FUSIONRAG", "false").lower() == "true":
        response_text, prompt_tokens, completion_tokens = gpt_generate_answer_fusionrag(client=client,
                                                      system_prompt=system_prompt,
                                                      fusionrag_cache_list=fusionrag_prompt_list,
                                                      prefix=prefix,
                                                      query_prompt=query_prompt
                                                      )
    else:
        response_text, prompt_tokens, completion_tokens = gpt_generate_answer(prompt, messages, client)
    response_text = clean_json(response_text)
    import json
    try:
        summaries = json.loads(response_text)
    except Exception:
        summaries = []
    return {"input": text, "summaries": summaries}, prompt_tokens, completion_tokens

# def gpt_personality_analysis(dialogs, client):
#     prompt = ("Please analyze the following conversation and extract the user profile information and user private data."
#               "Please output in the following format:\n"
#               "【User Profile】\n"
#               "Areas of Interest:\n"
#               "Response Preferences：\n"
#               "Preferred Content Type：\n"
#               "Short vs. Detailed Responses：\n"
#               "Formal vs. Casual Tone：\n"
#               "Other Notes:：\n"
#               "【User Private Data】\n"
#               "Please list all the private information involved (such as account numbers, passwords, user purchase,etc.). If there is none, please write \"None\"\n\n"
#               "The conversation is as follows:\n")
#     for d in dialogs:
#         prompt += f"User: {d.get('user_input','')}\nAssiant: {d.get('agent_response','')}\n"
#     messages = [
#         {"role": "system", "content": "You are a professional user profile analyst who can also identify user private data. Please strictly follow the template for output."},
#         {"role": "user", "content": prompt}
#     ]
#     print("调用 GPT 分析用户画像和私有数据...")
#     result_text = gpt_generate_answer(prompt, messages, client)
#     profile, private = "", ""
#     parts = result_text.split("【User Private Data】")
#     if len(parts) == 2:
#         profile = parts[0].replace("【User Profile】", "").strip()
#         private = parts[1].strip()
#     else:
#         profile = result_text.strip()
#         private = "None"
#     return {"profile": profile, "private": private}
# def gpt_personality_analysis(dialogs, client):
#     """
#     Analyzes conversations to extract structured personality traits, private knowledge, 
#     and assistant-related knowledge.
#     Returns: {"profile": str, "private": str, "assistant_knowledge": str}
#     """
#     conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}" for d in dialogs])

#     prompt = """
# # Personality Analysis Task
# Analyze the conversation and output in EXACTLY this format:

# 【User Profile】
# 1. Core Psychological Traits:
#    - [Trait]: [Positive/Negative/Neutral] (Evidence)
#    - (Max 5 most prominent traits)

# 2. Content Preferences:
#    - [Topic]: [Like/Dislike/Neutral] (Evidence)
#    - (Max 5 strongest preferences)

# 3. Interaction Style:
#    - [Style]: [Preference] (Evidence)
#    - (e.g., Direct/Indirect, Detailed/Concise)

# 4. Value Alignment:
#    - [Value]: [Strong/Weak] (Evidence)
#    - (e.g., Honesty, Helpfulness)

# 【User Private Data】
# - [Fact 1]
# - [Fact 2]
# - (Or "None" if none found)

# Conversation:
# """ + conversation

#     messages = [
#         {
#             "role": "system",
#             "content": """You are a personality analysis engine. Rules:
# 1. Extract ONLY observable traits with direct evidence
# 2. Use standardized trait names from psychology
# 3. Mark confidence: Positive=explicit preference, Neutral=implied
# 4. Private data includes possessions, habits, and sensitive preferences"""
#         },
#         {"role": "user", "content": prompt}
#     ]

#     print("Running personality analysis...")
#     result = gpt_generate_answer(prompt, messages, client)
    
#     # Parse output
#     profile, private = result.split("【User Private Data】") if "【User Private Data】" in result else (result, "None")
    
#     # Analyze assistant knowledge
#     assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
#     return {
#         "profile": profile.replace("【User Profile】", "").strip(),
#         "private": private.strip(),
#         "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
#     }
def gpt_personality_analysis(dialogs, client):
    """
    Analyzes conversations to extract structured personality traits, general user data, 
    and assistant-related knowledge.
    Returns: {"profile": str, "user_data": str, "assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}\nTime:{d['timestamp']}" for d in dialogs])
    fusionrag_cache_list = [f"User: {d['user_input']}\nAssistant: {d['agent_response']}\nTime:{d['timestamp']}" for d in dialogs]

    prefix = """
# Personality and User Data Analysis Task
Analyze the conversation and output in EXACTLY this format:

【User Profile】
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)
   - (Max 5 most prominent traits)

2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)
   - (Max 5 strongest preferences)

3. Interaction Style:
   - [Style]: [Preference] (Evidence)
   - (e.g., Direct/Indirect, Detailed/Concise)

4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)
   - (e.g., Honesty, Helpfulness)

【User Data】
- [Fact 1]: [Details] (e.g., "User mentioned visiting a park on April 1st, 2025 in New York.")
- [Fact 2]: [Details] (e.g., "User likes pizza, enjoys sci-fi movies, and dislikes rainy weather.")
- (Include events, dates, locations, preferences, or other general or private information explicitly mentioned in the conversation. If none, write "None.")

Conversation:
"""
    prompt = prefix + conversation
    system_prompt = """You are a personality and user data analysis engine. Rules:
1. Extract ONLY observable traits and data with direct evidence.
2. Include general user data such as events, dates, locations, and preferences.
3. Use concise and factual statements.
4. If no relevant information is found, output "None"."""

    query_prompt = "Analyze the conversation and output in EXACTLY the format before."

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {"role": "user", "content": prompt}
    ]

    print("Running personality and user data analysis...")
    if os.environ.get("FUSIONRAG", "false").lower() == "true":
        result, prompt_tokens, completion_tokens = gpt_generate_answer_fusionrag(system_prompt=system_prompt, prefix=prefix,
                                      fusionrag_cache_list=fusionrag_cache_list, query_prompt=query_prompt,
                                      client=client)
    else:
        result, prompt_tokens, completion_tokens = gpt_generate_answer(prompt, messages, client)
    
    # Parse output
    profile, user_data = result.split("【User Data】") if "【User Data】" in result else (result, "None")
    
    # Analyze assistant knowledge
    assistant_knowledge_result, prompt_tokens_1, completion_tokens_1 = analyze_assistant_knowledge(dialogs, client)
    
    return {
        "profile": profile.replace("【User Profile】", "").strip(),
        "private": user_data.strip(),
        "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
    }, prompt_tokens+prompt_tokens_1, completion_tokens+completion_tokens_1

def gpt_update_profile(old_profile, new_analysis, client):
    """
    Dynamically merges old and new profile data
    Args:
        old_profile: Previous profile text (structured)
        new_analysis: New analysis text (same format)
    Returns:
        Merged profile text with conflict resolution
    """
    prefix = """
    # Profile Merge Task
Consolidate these profiles while:
- Preserving all valid observations
- Resolving conflicts
- Adding new dimensions
"""
    query_prompt = """## Rules
1. Keep ALL verified traits from both
2. Resolve conflicts by:
   a) New explicit evidence > old assumptions
   b) Mark as Neutral if contradictory
3. Add new dimensions from new data
4. Maintain EXACT original format

Output ONLY the merged profile (no commentary):
The generated content should not exceed 1500 words
    """


    prompt = f"""{prefix}

## Current Profile
{old_profile}

## New Data
{new_analysis}

{query_prompt}
"""

    fusionrag_cache_list = [
        f"""## Current Profile
{old_profile}""",
        f"""## New Data
{new_analysis}"""
    ]
    

    system_prompt = """You are a profile integration system. Your rules:
1. NEVER discard verified information
2. Conflict resolution hierarchy:
   Explicit statement > Implied trait > Assumption
3. Add timestamps when traits change:
   (Updated: [date]) for modified traits
4. Preserve the 4-category structure"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {"role": "user", "content": prompt}
    ]

    print("Updating user profile dynamically...")

    if os.environ.get("FUSIONRAG", "false").lower() == "true":
        return gpt_generate_answer_fusionrag(system_prompt=system_prompt, prefix=prefix,
                                             fusionrag_cache_list=fusionrag_cache_list, query_prompt=query_prompt,
                                             client=client)
    else:
        return gpt_generate_answer(prompt, messages, client)

def gpt_extract_theme(answer_text, client):
    prompt = f"请从以下回答中提取主题总结，并以【主题提取】：开头输出：\n{answer_text}\n"
    messages = [
        {"role": "system", "content": "You are an expert in extracting conversation topics."},
        {"role": "user", "content": prompt}
    ]
    system_prompt = "You are an expert in extracting conversation topics."
    prefix = "请从以下回答中提取主题总结，并以【主题提取】：开头输出：\n"
    fusionrag_cache_list = [answer_text]
    query_prompt = "主题"
    print("调用 GPT 提取主题总结...")
    if os.environ.get("FUSIONRAG", "false").lower() == "true":
        return gpt_generate_answer_fusionrag(system_prompt=system_prompt, prefix=prefix,
                                             fusionrag_cache_list=fusionrag_cache_list, query_prompt=query_prompt,
                                             client=client)
    else:
        return gpt_generate_answer(prompt, messages, client)

def llm_extract_keywords(text, client):
    prompt = "Please extract the keywords of the conversation topic from the following dialogue, separated by commas, and do not exceed three:\n" + text
    messages = [
        {"role": "system", "content": "You are a keyword extraction expert. Please extract the keywords of the conversation topic."},
        {"role": "user", "content": prompt}
    ]
    system_prompt = "You are a keyword extraction expert. Please extract the keywords of the conversation topic."
    prefix = "Please extract the keywords of the conversation topic from the following dialogue, separated by commas, and do not exceed three:\n"
    fusionrag_cache_list = [text]
    query_prompt = "keywords: "
    print("调用 GPT 提取关键词...")
    if os.environ.get("FUSIONRAG", "false").lower() == "true":
        keywords_text, prompt_tokens, completion_tokens = gpt_generate_answer_fusionrag(system_prompt=system_prompt, prefix=prefix,
                                                      fusionrag_cache_list=fusionrag_cache_list, query_prompt=query_prompt, client=client)
    else:
        keywords_text, prompt_tokens, completion_tokens = gpt_generate_answer(prompt, messages, client)
    keywords = [w.strip() for w in keywords_text.split(",") if w.strip()]
    return set(keywords), prompt_tokens, completion_tokens

def compute_time_decay(session_timestamp, current_timestamp, tau=3600):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = datetime.strptime(session_timestamp, fmt)
    t2 = datetime.strptime(current_timestamp, fmt)
    delta = (t2 - t1).total_seconds()
    return np.exp(-delta/tau)


def clean_json(response: str) -> str:
    """
    Cleans the model response by:
    1. Removing enclosing code block markers (```[language] ... ```).
    2. Parsing the JSON content safely.
    3. Returning the value of the "data" key if present, otherwise trying to return the parsed list/dict.
    """
    import re
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response.strip())
    cleaned = match.group(1).strip() if match else response.strip()

    return cleaned
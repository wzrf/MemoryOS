import os

from utils import gpt_summarize, generate_id, get_timestamp, gpt_update_profile, gpt_generate_multi_summary

class DynamicUpdate:
    def __init__(self, short_term_memory, mid_term_memory, long_term_memory, topic_similarity_threshold=0.8, client=None):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.topic_similarity_threshold = topic_similarity_threshold
        self.client = client
        self.last_evicted_page = None
        self.calls = 0
        self.radix_cache = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.fusionrag_stats = []

    def get_stats(self):
        return {
            "calls": self.calls,
            "radix_cache": self.radix_cache,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "fusionrag_stats": self.fusionrag_stats,
        }

    def _is_conversation_continuing(self, previous_page, current_page):
        if not previous_page:
            return False
            
        prompt = """Determine if these two conversation pages are continuous (true continuation without topic shift).
Return ONLY "true" or "false".

Previous Page:
User: {prev_user}
Assistant: {prev_agent}

Current Page:
User: {curr_user}
Assistant: {curr_agent}

Continuous?""".format(
            prev_user=previous_page.get("user_input", ""),
            prev_agent=previous_page.get("agent_response", ""),
            curr_user=current_page.get("user_input", ""),
            curr_agent=current_page.get("agent_response", "")
        )
        
        messages = [
            {"role": "system", "content": "You are a conversation continuity detector. Return ONLY 'true' or 'false'."},
            {"role": "user", "content": prompt}
        ]

        system_prompt = "You are a conversation continuity detector. Return ONLY 'true' or 'false'."
        prefix = """Determine if these two conversation pages are continuous (true continuation without topic shift).
Return ONLY "true" or "false".

Previous Page:"""

        fusionrag_list = [
            f"""Previous Page:
User: {previous_page.get("user_input", "")}
Assistant: {previous_page.get("agent_response", "")}""",

f"""Current Page:
User: {current_page.get("user_input", "")}
Assistant: {current_page.get("agent_response", "")}"""]
        query_prompt = """
        Continuous?"""

        fusionrag_list[0] = prefix + fusionrag_list[0]

        if os.environ.get("FUSIONRAG", "false").lower() == "true": ##mengyao_debug first call.
            response, prompt_tokens, completion_tokens, fusionrag_stats = self.client.chat_completion_fusionrag(
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                fusionrag_cache_list=fusionrag_list,
                query_prompt=query_prompt,
                prefix="",
                temperature=0.0,
                max_tokens=10

            )
            fusionrag_stats["reuse_type"] = "reuse_prefill"
        else:
            response, prompt_tokens, completion_tokens, fusionrag_stats = self.client.chat_completion_with_usage(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
        self.calls += 1
        self.radix_cache.append(system_prompt + prefix)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.fusionrag_stats.append(fusionrag_stats)

        return response.strip().lower() == "true"

    def _generate_meta_info(self, last_page_meta, current_page):
        """
        基于上一页的meta-info和当前页内容生成新的meta-info
        :param last_page_meta: 上一页的meta-info内容
        :param current_page: 当前页的对话内容
        :return: 更新后的meta-info
        """
        current_conversation = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
        
        prompt = """Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.
        
    Guidelines:
    1. Start from the previous meta-summary (if exists)
    2. Add/update information based on the new dialogue
    3. Keep it concise (1-2 sentences max)
    4. Maintain context coherence

    Previous Meta-summary: {last_meta}
    New Dialogue:
    {new_dialogue}

    Updated Meta-summary:""".format(
            last_meta=last_page_meta if last_page_meta else "None",
            new_dialogue=current_conversation
        )
        
        messages = [
            {"role": "system", "content": """You are a conversation meta-summary updater. Your task is to:
    1. Preserve relevant context from previous meta-summary
    2. Integrate new information from current dialogue
    3. Output ONLY the updated summary (no explanations)"""},
            {"role": "user", "content": prompt}
        ]

        fusionrag_prompt_list = [
            f"""Previous Meta-summary: {last_page_meta if last_page_meta else "None"}
        New Dialogue:
        {current_conversation}"""]

        system_prompt = """You are a conversation meta-summary updater. Your task is to:
    1. Preserve relevant context from previous meta-summary
    2. Integrate new information from current dialogue
    3. Output ONLY the updated summary (no explanations)"""

        prefix = """Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.
        
    Guidelines:
    1. Start from the previous meta-summary (if exists)
    2. Add/update information based on the new dialogue
    3. Keep it concise (1-2 sentences max)
    4. Maintain context coherence
"""

        query_prompt = "Updated Meta-summary:"
        fusionrag_prompt_list[0] = prefix + fusionrag_prompt_list[0]
        if os.environ.get("FUSIONRAG", "false").lower() == "true":## mengyao_debug 首次调用，不会触发
            content, prompt_tokens, completion_tokens, fusionrag_stats = self.client.chat_completion_fusionrag(
                model="qwen3-8b",
                system_prompt=system_prompt,
                prefix="",
                fusionrag_cache_list=fusionrag_prompt_list,
                query_prompt=query_prompt,
                temperature=0.3,
                max_tokens=100
            )
            fusionrag_stats["reuse_type"] = "reuse_prefill"
        else:
            content, prompt_tokens, completion_tokens, fusionrag_stats = self.client.chat_completion_with_usage(
                model="qwen3-8b",
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
        self.calls += 1
        self.radix_cache.append(prefix + system_prompt)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.fusionrag_stats.append(fusionrag_stats)
        return content



    def _update_connected_pages(self, page_id, new_meta_info):
        connected_pages = []
        current_page = self.mid_term_memory.get_page_by_id(page_id)
        
        if not current_page:
            return
            
        prev_page_id = current_page.get("pre_page")
        while prev_page_id:
            prev_page = self.mid_term_memory.get_page_by_id(prev_page_id)
            if prev_page:
                connected_pages.insert(0, prev_page)
                prev_page_id = prev_page.get("pre_page")
            else:
                break
                
        next_page_id = current_page.get("next_page")
        while next_page_id:
            next_page = self.mid_term_memory.get_page_by_id(next_page_id)
            if next_page:
                connected_pages.append(next_page)
                next_page_id = next_page.get("next_page")
            else:
                break
                
        for page in connected_pages:
            page["meta_info"] = new_meta_info
            self.mid_term_memory.update_page_connections(page.get("pre_page"), page.get("next_page"))

    def update_short_term(self, message):
        self.short_term_memory.add_qa_pair(message)

    def bulk_evict_and_update_mid_term(self):
        evicted = []
        # 1. 从短期记忆移除内容（保持不变）
        while self.short_term_memory.is_full():
            msg = self.short_term_memory.pop_oldest()
            if msg and msg.get("user_input") and msg.get("agent_response"):
                evicted.append(msg)
        
        if not evicted:
            return
        
        # 2. 先创建基础页面结构并进行连续性处理
        pages = []
        for qa in evicted:
            page = {
                "page_id": generate_id("page"),
                "user_input": qa.get("user_input", ""),
                "agent_response": qa.get("agent_response", ""),
                "timestamp": qa.get("timestamp"),
                "preloaded": False,
                "analyzed": False,
                "pre_page": None,
                "next_page": None,
                "meta_info": None
            }
            
            # 连续性判断
            is_continuous = self._is_conversation_continuing(self.last_evicted_page, page)
            if is_continuous and self.last_evicted_page:
                page["pre_page"] = self.last_evicted_page["page_id"]
                self.last_evicted_page["next_page"] = page["page_id"]
                
                # 更新元信息
                last_meta = self.last_evicted_page.get("meta_info")
                new_meta_info = self._generate_meta_info(last_meta, page)
                page["meta_info"] = new_meta_info
                self._update_connected_pages(page["pre_page"], new_meta_info)
            else:
                page["meta_info"] = self._generate_meta_info(None, page)
            
            pages.append(page)
            self.last_evicted_page = page
        
        # 3. 将所有用户输入拼接用于主题分析
        input_text = "\n".join([f"User: {page.get('user_input','')}\n" for page in pages])
        # print("动态更新：调用 GPT 生成多子主题摘要...")
        multi_summary, prompt_tokens, completion_tokens, fusionrag_stats, prefix_ = gpt_generate_multi_summary(input_text, self.client)
        fusionrag_stats["reuse_type"] = "reuse_prefill"
        self.calls += 1
        self.radix_cache.append(prefix_)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.fusionrag_stats.append(fusionrag_stats)

        # 4. 按主题分组插入中期记忆
        for summary_dict in multi_summary.get("summaries", []):
            sub_summary = summary_dict.get("content", "")
            sub_key_words = summary_dict.get("keywords", [])
            
            print(f"动态更新：处理子主题【{summary_dict.get('theme','')}】，插入中期记忆...")
            prompt_tokens, completion_tokens, fusiorag_stats_list, radix_cache_list = self.mid_term_memory.insert_pages_into_session(
                sub_summary, 
                sub_key_words, 
                pages,  # 传入已经处理好的完整pages
                self.topic_similarity_threshold
            )
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.radix_cache.extend(radix_cache_list)
            self.fusionrag_stats.extend(fusiorag_stats_list)

    def update_long_term(self, user_id, new_profile_data, knowledge_text):
        print("动态更新：更新长期记忆中的用户画像和私有数据...")
        self.long_term_memory.update_user_profile(user_id, new_profile_data)
        self.long_term_memory.add_knowledge(knowledge_text)
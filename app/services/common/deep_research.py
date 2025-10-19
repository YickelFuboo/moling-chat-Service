#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import re
from functools import partial
from app.rag_core.llm_service import LLMBundle
from app.rag_core.rag.nlp import extract_between
from app.rag_core.rag.prompts import kb_prompt
from app.infrastructure.web_search.tavily import Tavily

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
MAX_SEARCH_LIMIT = 6

REASON_PROMPT = (
        "You are a reasoning assistant with the ability to perform dataset searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        f"- To perform a search: write {BEGIN_SEARCH_QUERY} your query here {END_SEARCH_QUERY}.\n"
        f"Then, the system will search and analyze relevant content, then provide you with helpful information in the format {BEGIN_SEARCH_RESULT} ...search results... {END_SEARCH_RESULT}.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "-- Example 1 --\n" ########################################
        "Question: \"Are both the directors of Jaws and Casino Royale from the same country?\"\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Who is the director of Jaws?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nThe director of Jaws is Steven Spielberg...\n{END_SEARCH_RESULT}\n\n"
        "Continues reasoning with the new information.\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Where is Steven Spielberg from?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nSteven Allan Spielberg is an American filmmaker...\n{END_SEARCH_RESULT}\n\n"
        "Continues reasoning with the new information...\n\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Who is the director of Casino Royale?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nCasino Royale is a 2006 spy film directed by Martin Campbell...\n{END_SEARCH_RESULT}\n\n"
        "Continues reasoning with the new information...\n\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Where is Martin Campbell from?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nMartin Campbell (born 24 October 1943) is a New Zealand film and television director...\n{END_SEARCH_RESULT}\n\n"
        "Continues reasoning with the new information...\n\n"
        "Assistant:\nIt's enough to answer the question\n"

        "-- Example 2 --\n" #########################################
        "Question: \"When was the founder of craigslist born?\"\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Who was the founder of craigslist?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nCraigslist was founded by Craig Newmark...\n{END_SEARCH_RESULT}\n\n"
        "Continues reasoning with the new information.\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY} When was Craig Newmark born?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nCraig Newmark was born on December 6, 1952...\n{END_SEARCH_RESULT}\n\n"
        "Continues reasoning with the new information...\n\n"
        "Assistant:\nIt's enough to answer the question\n"
        "**Remember**:\n"
        f"- You have a dataset to search, so you just provide a proper search query.\n"
        f"- Use {BEGIN_SEARCH_QUERY} to request a dataset search and end with {END_SEARCH_QUERY}.\n"
        "- The language of query MUST be as the same as 'Question' or 'search result'.\n"
        "- If no helpful information can be found, rewrite the search query to be less and precise keywords.\n"
        "- When done searching, continue your reasoning.\n\n"
        'Please answer the following question. You should think step by step to solve it.\n\n'
    )

RELEVANT_EXTRACTION_PROMPT = """**Task Instruction:**

    You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

    **Guidelines:**

    1. **Analyze the Searched Web Pages:**
    - Carefully review the content of each searched web page.
    - Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

    2. **Extract Relevant Information:**
    - Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
    - Ensure that the extracted information is accurate and relevant.

    3. **Output Format:**
    - **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
    - The language of query **MUST BE** as the same as 'Search Query' or 'Web Pages'.\n"
    **Final Information**

    [Helpful information]

    - **If the web pages do not provide any helpful information for current search query:** Output the following text.

    **Final Information**

    No helpful information found.

    **Inputs:**
    - **Previous Reasoning Steps:**  
    {prev_reasoning}

    - **Current Search Query:**  
    {search_query}

    - **Searched Web Pages:**  
    {document}

    """


class DeepResearcher:
    """
    深度研究推理器 (Deep Research Reasoner)
    
    这是一个智能推理引擎，通过多轮检索和分析来回答复杂问题。
    它结合了LLM的推理能力和多源信息检索，能够处理需要深度分析的问题。
    
    主要功能：
    1. 多轮推理 - 基于前一轮结果进行迭代推理
    2. 智能检索 - 从多个信息源获取相关信息
    3. 信息整合 - 合并和去重不同来源的信息
    4. 相关性分析 - 评估检索结果与问题的相关性
    5. 流式输出 - 实时显示推理过程
    
    Attributes:
        chat_mdl (LLMBundle): 聊天模型实例，用于生成推理和总结
        prompt_config (dict): 提示词配置，包含各种检索源配置
        _kb_retrieve (partial): 知识库检索函数
        _kg_retrieve (partial): 知识图谱检索函数
    """
    
    def __init__(self,
                 chat_mdl: LLMBundle,
                 prompt_config: dict,
                 kb_retrieve: partial = None,
                 kg_retrieve: partial = None
                 ):
        """
        初始化深度研究推理器
        
        Args:
            chat_mdl (LLMBundle): 聊天模型实例，用于生成推理和总结
            prompt_config (dict): 提示词配置，包含：
                - tavily_api_key: Tavily网络搜索API密钥
                - use_kg: 是否启用知识图谱检索
            kb_retrieve (partial, optional): 知识库检索函数，用于从本地知识库检索信息
            kg_retrieve (partial, optional): 知识图谱检索函数，用于从知识图谱检索信息
        """
        self.chat_mdl = chat_mdl
        self.prompt_config = prompt_config
        self._kb_retrieve = kb_retrieve
        self._kg_retrieve = kg_retrieve

    @staticmethod
    def _remove_tags(text: str, start_tag: str, end_tag: str) -> str:
        """
        通用标签移除方法
        
        使用正则表达式移除文本中指定标签及其内容。
        用于清理LLM输出中的特殊标记。
        
        Args:
            text (str): 要处理的文本
            start_tag (str): 开始标签
            end_tag (str): 结束标签
            
        Returns:
            str: 移除标签后的文本
        """
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        return re.sub(pattern, "", text)

    @staticmethod
    def _remove_query_tags(text: str) -> str:
        """
        移除查询标签
        
        移除LLM输出中的搜索查询标签，用于清理推理过程中的查询标记。
        
        Args:
            text (str): 包含查询标签的文本
            
        Returns:
            str: 移除查询标签后的文本
        """
        return DeepResearcher._remove_tags(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

    @staticmethod
    def _remove_result_tags(text: str) -> str:
        """
        移除结果标签
        
        移除LLM输出中的搜索结果标签，用于清理推理过程中的结果标记。
        
        Args:
            text (str): 包含结果标签的文本
            
        Returns:
            str: 移除结果标签后的文本
        """
        return DeepResearcher._remove_tags(text, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT)

    async def _generate_reasoning(self, msg_history):
        """
        生成推理步骤
        
        使用LLM基于当前消息历史生成推理步骤。这是深度研究的核心步骤，
        通过分析当前信息和历史上下文，生成下一步的推理方向。
        
        Args:
            msg_history (list): 消息历史列表，包含用户问题和之前的推理步骤
            
        Yields:
            str: 流式生成的推理步骤文本
            
        Returns:
            str: 完整的推理步骤文本
        """
        query_think = ""
        
        # 确保最后一条消息是用户消息，如果不是则添加提示
        if msg_history[-1]["role"] != "user":
            msg_history.append({"role": "user", "content": "Continues reasoning with the new information.\n"})
        else:
            msg_history[-1]["content"] += "\n\nContinues reasoning with the new information.\n"
            
        # 使用LLM生成推理步骤，流式输出
        # 样例：LLM流式输出内容
        # 第1次输出：ans = "<think>我需要搜索一些信息来回答这个问题</think>让我搜索一下："
        # 第2次输出：ans = "<think>我需要搜索一些信息来回答这个问题</think>让我搜索一下：<|begin_search_query|>"
        # 第3次输出：ans = "<think>我需要搜索一些信息来回答这个问题</think>让我搜索一下：<|begin_search_query|>什么是人工智能"
        # 第4次输出：ans = "<think>我需要搜索一些信息来回答这个问题</think>让我搜索一下：<|begin_search_query|>什么是人工智能<|end_search_query|>"
        async for ans in self.chat_mdl.chat_stream(REASON_PROMPT, msg_history, {"temperature": 0.7}):
            # 清理LLM输出，移除思考过程标记
            # 样例：处理前 ans = "<think>我需要搜索一些信息来回答这个问题</think>让我搜索一下：<|begin_search_query|>什么是人工智能<|end_search_query|>"
            # 样例：处理后 ans = "让我搜索一下：<|begin_search_query|>什么是人工智能<|end_search_query|>"
            ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
            if not ans:
                continue
            query_think = ans
            # 样例：yield输出给调用方
            # 第1次：yield "让我搜索一下："
            # 第2次：yield "让我搜索一下：<|begin_search_query|>"
            # 第3次：yield "让我搜索一下：<|begin_search_query|>什么是人工智能"
            # 第4次：yield "让我搜索一下：<|begin_search_query|>什么是人工智能<|end_search_query|>"
            yield query_think
            
        return

    def _extract_search_queries(self, query_think, question, step_index):
        """
        从推理过程中提取搜索查询
        
        分析LLM生成的推理步骤，提取其中包含的搜索查询。
        如果第一步没有找到查询，则使用原始问题作为查询。
        
        Args:
            query_think (str): LLM生成的推理步骤文本
            question (str): 原始用户问题
            step_index (int): 当前推理步骤的索引
            
        Returns:
            list: 提取的搜索查询列表
        """
        # 从推理文本中提取搜索查询标签内的内容
        queries = extract_between(query_think, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
        
        # 如果是第一步且没有找到查询，使用原始问题作为查询
        if not queries and step_index == 0:
            queries = [question]
            
        return queries

    def _truncate_previous_reasoning(self, all_reasoning_steps):
        """
        截断之前的推理步骤以保持合理长度
        
        由于LLM的上下文长度限制，需要截断过长的推理历史。
        保留最重要的步骤：第一步、最后几步、包含搜索查询或结果的步骤。
        
        Args:
            all_reasoning_steps (list): 所有推理步骤的列表
            
        Returns:
            str: 截断后的推理历史文本
        """
        truncated_prev_reasoning = ""
        
        # 为每个步骤添加编号
        for i, step in enumerate(all_reasoning_steps):
            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

        # 按段落分割
        prev_steps = truncated_prev_reasoning.split('\n\n')
        
        # 如果步骤数量较少，直接返回
        if len(prev_steps) <= 5:
            truncated_prev_reasoning = '\n\n'.join(prev_steps)
        else:
            # 智能截断：保留重要步骤
            truncated_prev_reasoning = ''
            for i, step in enumerate(prev_steps):
                # 保留第一步、最后4步、包含搜索标记的步骤
                if (i == 0 or 
                    i >= len(prev_steps) - 4 or 
                    BEGIN_SEARCH_QUERY in step or 
                    BEGIN_SEARCH_RESULT in step):
                    truncated_prev_reasoning += step + '\n\n'
                else:
                    # 添加省略号标记
                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                        truncated_prev_reasoning += '...\n\n'
        
        return truncated_prev_reasoning.strip('\n')

    async def _retrieve_information(self, search_query):
        """
        从多个信息源检索信息
        
        这是深度研究的核心检索功能，支持从多个信息源获取相关信息：
        1. 本地知识库检索
        2. 网络搜索（通过Tavily API）
        3. 知识图谱检索
        
        Args:
            search_query (str): 搜索查询字符串
            
        Returns:
            dict: 包含检索结果的字典，格式为：
                {
                    "chunks": [检索到的文档片段列表],
                    "doc_aggs": [文档聚合信息列表]
                }
        """
        # 初始化结果结构
        kbinfos = {"chunks": [], "doc_aggs": []}
        
        # 1. 知识库检索 - 从本地知识库获取相关信息
        try:
            if self._kb_retrieve:
                kbinfos = await self._kb_retrieve(question=search_query)
            else:
                kbinfos = {"chunks": [], "doc_aggs": []}
        except Exception as e:
            logging.error(f"Knowledge base retrieval error: {e}")

        # 2. 网络搜索 - 通过Tavily API获取网络信息
        try:
            if self.prompt_config.get("tavily_api_key"):
                tav = Tavily()
                tav_res = await tav.retrieve_chunks(search_query)
                # 将网络搜索结果合并到知识库结果中
                kbinfos["chunks"].extend(tav_res["chunks"])
                kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
        except Exception as e:
            logging.error(f"Web retrieval error: {e}")

        # 3. 知识图谱检索 - 从知识图谱获取结构化信息
        try:
            if self.prompt_config.get("use_kg") and self._kg_retrieve:
                ck = await self._kg_retrieve(question=search_query)
                if ck.get("content_with_weight"):
                    # 将知识图谱结果插入到最前面，给予更高优先级
                    kbinfos["chunks"].insert(0, ck)
        except Exception as e:
            logging.error(f"Knowledge graph retrieval error: {e}")

        return kbinfos

    def _update_chunk_info(self, chunk_info, kbinfos):
        """
        更新文档片段信息用于引用管理
        
        将新检索到的信息合并到现有的文档片段信息中，避免重复。
        这是多轮检索中的关键步骤，确保引用信息的完整性和准确性。
        
        Args:
            chunk_info (dict): 现有的文档片段信息，会被更新
            kbinfos (dict): 新检索到的信息
        """
        if not chunk_info["chunks"]:
            # 如果是第一次检索，直接使用检索结果
            for k in chunk_info.keys():
                chunk_info[k] = kbinfos[k]
        else:
            # 合并新检索到的信息，避免重复
            # 基于chunk_id去重
            cids = [c["chunk_id"] for c in chunk_info["chunks"]]
            for c in kbinfos["chunks"]:
                if c["chunk_id"] not in cids:
                    chunk_info["chunks"].append(c)
                    
            # 基于doc_id去重文档聚合信息
            dids = [d["doc_id"] for d in chunk_info["doc_aggs"]]
            for d in kbinfos["doc_aggs"]:
                if d["doc_id"] not in dids:
                    chunk_info["doc_aggs"].append(d)

    async def _extract_relevant_info(self, truncated_prev_reasoning, search_query, kbinfos):
        """
        提取和总结相关信息
        
        使用LLM分析检索到的文档，提取与当前搜索查询和之前推理步骤相关的信息。
        这是深度研究中的关键步骤，确保只保留最相关的信息用于后续推理。
        
        Args:
            truncated_prev_reasoning (str): 截断后的之前推理步骤
            search_query (str): 当前搜索查询
            kbinfos (dict): 检索到的文档信息
            
        Yields:
            str: 流式生成的相关信息总结
            
        Returns:
            str: 完整的相关信息总结
        """
        summary_think = ""
        
        # 构建提示词，包含之前的推理步骤、搜索查询和检索到的文档
        prompt = RELEVANT_EXTRACTION_PROMPT.format(
            prev_reasoning=truncated_prev_reasoning,
            search_query=search_query,
            document="\n".join(kb_prompt(kbinfos, 4096))  # 格式化文档内容，限制token数量
        )
        
        # 构建用户消息
        user_msg = [{"role": "user",
                    "content": f'Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.'}]
        
        # 使用LLM分析文档并提取相关信息，流式输出
        async for ans in self.chat_mdl.chat_stream(prompt, user_msg, {"temperature": 0.7}):
            # 清理LLM输出，移除思考过程标记
            ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
            if not ans:
                continue
            summary_think = ans
            yield summary_think
        
        return

    async def thinking(self, chunk_info: dict, question: str):
        """
        深度研究的主要推理方法
        
        这是深度研究的核心方法，通过多轮推理、检索和分析来回答复杂问题。
        整个流程包括：问题分析、查询提取、多源检索、信息整合、相关性分析等步骤。
        
        主要特点：
        1. 多轮迭代推理 - 基于前一轮结果进行下一轮推理
        2. 智能查询生成 - 从推理过程中自动提取搜索查询
        3. 多源信息整合 - 从知识库、网络、知识图谱等获取信息
        4. 重复查询检测 - 避免重复执行相同的搜索查询
        5. 流式输出 - 实时显示推理过程和思考步骤
        
        Args:
            chunk_info (dict): 用于存储检索结果的文档片段信息，会被更新
            question (str): 用户提出的问题
            
        Yields:
            dict: 包含推理过程的响应字典，格式为：
                {
                    "answer": 推理过程文本,
                    "reference": 引用信息（空字典）,
                    "audio_binary": 音频数据（None）
                }
        """
        # 初始化变量
        executed_search_queries = []  # 已执行的搜索查询列表，用于避免重复
        msg_history = [{"role": "user", "content": f'Question:\"{question}\"\n'}]  # 消息历史
        all_reasoning_steps = []  # 所有推理步骤
        think = "<think>"  # 思考过程开始标记
        
        # 主循环：最多执行MAX_SEARCH_LIMIT轮推理
        for step_index in range(MAX_SEARCH_LIMIT + 1):
            # 检查是否达到最大搜索限制
            if step_index == MAX_SEARCH_LIMIT - 1:
                # 样例：达到搜索限制时的输出
                # summary_think = "\n<|begin_search_result|>\nThe maximum search limit is exceeded. You are not allowed to search.\n<|end_search_result|>\n"
                summary_think = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                yield {"answer": think + summary_think + "</think>", "reference": {}, "audio_binary": None}
                all_reasoning_steps.append(summary_think)
                msg_history.append({"role": "assistant", "content": summary_think})
                break

            # 步骤1：生成推理步骤
            # 样例：LLM生成的推理内容
            # query_think = "<think>我需要搜索一些信息来回答这个问题</think>让我搜索一下：<|begin_search_query|>什么是人工智能<|end_search_query|>"
            query_think = ""
            async for ans in self._generate_reasoning(msg_history):
                query_think = ans
                # 样例：yield输出给用户的内容（已移除<think>标签）
                # {"answer": "<think>让我搜索一下：什么是人工智能", "reference": {}, "audio_binary": None}
                yield {"answer": think + self._remove_query_tags(query_think) + "</think>", "reference": {}, "audio_binary": None}

            # 样例：think变量累积内容
            # think = "<think>让我搜索一下：什么是人工智能"
            think += self._remove_query_tags(query_think)
            all_reasoning_steps.append(query_think)
            
            # 步骤2：提取搜索查询
            # 样例：从query_think中提取出["什么是人工智能"]
            queries = self._extract_search_queries(query_think, question, step_index)
            if not queries and step_index > 0:
                # 如果不是第一步且没有查询，结束搜索过程
                break

            # 处理每个搜索查询
            for search_query in queries:
                # 样例：search_query = "什么是人工智能"
                logging.info(f"[THINK]Query: {step_index}. {search_query}")
                msg_history.append({"role": "assistant", "content": search_query})
                # 样例：think += "\n\n> 1. 什么是人工智能\n\n"
                think += f"\n\n> {step_index + 1}. {search_query}\n\n"
                # 样例：yield输出给用户
                # {"answer": "<think>让我搜索一下：什么是人工智能\n\n> 1. 什么是人工智能\n\n", "reference": {}, "audio_binary": None}
                yield {"answer": think + "</think>", "reference": {}, "audio_binary": None}

                # 检查查询是否已经执行过
                if search_query in executed_search_queries:
                    # 样例：重复查询时的提示
                    # summary_think = "\n<|begin_search_result|>\nYou have searched this query. Please refer to previous results.\n<|end_search_result|>\n"
                    summary_think = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                    yield {"answer": think + summary_think + "</think>", "reference": {}, "audio_binary": None}
                    all_reasoning_steps.append(summary_think)
                    msg_history.append({"role": "user", "content": summary_think})
                    think += summary_think
                    continue
                
                executed_search_queries.append(search_query)
                
                # 步骤3：截断之前的推理步骤
                # 样例：truncated_prev_reasoning = "之前的推理步骤内容（截断后）"
                truncated_prev_reasoning = self._truncate_previous_reasoning(all_reasoning_steps)
                
                # 步骤4：检索信息
                # 样例：kbinfos = [{"content": "人工智能是...", "source": "doc1"}, {"content": "AI技术包括...", "source": "doc2"}]
                kbinfos = await self._retrieve_information(search_query)
                
                # 步骤5：更新文档片段信息
                # 样例：chunk_info = {"doc1": [chunk1, chunk2], "doc2": [chunk3]}
                self._update_chunk_info(chunk_info, kbinfos)
                
                # 步骤6：提取相关信息
                think += "\n\n"
                summary_think = ""
                # 样例：summary_think = "**Final Information**\n\n人工智能（AI）是计算机科学的一个分支..."
                async for ans in self._extract_relevant_info(truncated_prev_reasoning, search_query, kbinfos):
                    summary_think = ans
                    # 样例：yield输出给用户（已移除结果标签）
                    # {"answer": "<think>让我搜索一下：什么是人工智能\n\n> 1. 什么是人工智能\n\n\n\n人工智能（AI）是计算机科学的一个分支...", "reference": {}, "audio_binary": None}
                    yield {"answer": think + self._remove_result_tags(summary_think) + "</think>", "reference": {}, "audio_binary": None}

                all_reasoning_steps.append(summary_think)
                # 样例：msg_history添加用户消息
                # {"role": "user", "content": "\n\n<|begin_search_result|>**Final Information**\n\n人工智能（AI）是计算机科学的一个分支...<|end_search_result|>\n\n"}
                msg_history.append(
                    {"role": "user", "content": f"\n\n{BEGIN_SEARCH_RESULT}{summary_think}{END_SEARCH_RESULT}\n\n"})
                think += self._remove_result_tags(summary_think)
                logging.info(f"[THINK]Summary: {step_index}. {summary_think}")

        # 返回完整的思考过程
        yield think + "</think>"

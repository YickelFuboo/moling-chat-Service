import logging
import time
import re
from copy import deepcopy
from typing import Dict, Any, Generator, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemes.qa import SingleQaRequest, ChatRequest, ChatMessage
from app.services.kb_service import KBService
from app.infrastructure.llm.llms import llm_factory, embedding_factory, rerank_factory
from app.rag_core.search_api import RETRIEVALER, KG_RETRIEVALER
from app.rag_core.rag.prompts import kb_prompt, chunks_format, cross_languages, keyword_extraction, message_fit_in
from app.rag_core.llm_service import LLMBundle, LLMType
from app.rag_core.rag.app.tag import label_question
from app.rag_core.utils import ParserType
from app.rag_core.rag.app.resume import forbidden_select_fields4resume
from app.infrastructure.web_search.tavily import Tavily
from app.rag_core.utils import num_tokens_from_string


BAD_CITATION_PATTERNS = [
    re.compile(r"\(\s*ID\s*[: ]*\s*(\d+)\s*\)"),  # (ID: 12) - 圆括号格式
    re.compile(r"\[\s*ID\s*[: ]*\s*(\d+)\s*\]"),  # [ID: 12] - 方括号格式
    re.compile(r"【\s*ID\s*[: ]*\s*(\d+)\s*】"),  # 【ID: 12】 - 中文方括号格式
    re.compile(r"ref\s*(\d+)", flags=re.IGNORECASE),  # ref12、REF 12 - ref格式
]

class QAService:
    """问答服务类"""

    @staticmethod
    async def single_ask(
        session: AsyncSession,
        request: SingleQaRequest,
        user_id: str,
        tenant_id: str,
        is_stream: bool = False
    ):
        """问答服务"""
        try:
            # 获取知识库信息
            kbs = await KBService.get_kb_by_ids(session, request.kb_ids)
            if not kbs:
                raise ValueError("知识库不存在")
            
            # 检查知识库是否使用相同的嵌入模型
            embedding_list = list(set([kb.embd_model_name for kb in kbs]))
            if len(embedding_list) > 1:
                raise ValueError("知识库使用了不同的嵌入模型，无法同时检索")
            
            # 检查是否为知识图谱模式
            is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
            retriever = KG_RETRIEVALER if is_knowledge_graph else RETRIEVALER
            
            # 创建模型
            embd_mdl = LLMBundle(kbs[0].tenant_id, LLMType.EMBEDDING, provider=kbs[0].embd_provider_name, model=kbs[0].embd_model_name)
            chat_mdl = LLMBundle(kbs[0].tenant_id, LLMType.CHAT)
            
            max_tokens = chat_mdl.max_length if hasattr(chat_mdl, 'max_length') else 8192
            tenant_ids = list(set([kb.tenant_id for kb in kbs]))
            kb_ids = [kb.id for kb in kbs]
            
            # 执行检索
            kbinfos = await retriever.retrieval(
                question=request.question,
                embd_mdl=embd_mdl,
                tenant_ids=tenant_ids,
                kb_ids=kb_ids,
                page=1,
                page_size=12,
                similarity_threshold=0.1,
                vector_similarity_weight=0.3,
                aggs=False,
                rank_feature=await label_question(session, request.question, kbs)
            )
            
            # 格式化知识库内容
            knowledges = await kb_prompt(session, kbinfos, max_tokens)
            
            # 构建提示词
            prompt = """
角色：你是一个智能助手，名字叫小智。
任务：基于知识库信息总结并回答用户的问题。
要求和限制：
  - 不要编造信息，特别是数字。
  - 如果知识库中的信息与用户问题无关，请直接说：抱歉，没有找到相关信息。
  - 使用markdown格式回答。
  - 使用用户问题的语言回答。
  - 不要编造信息，特别是数字。

### 知识库信息
%s

以上是知识库中的信息。

""" % "\n".join(knowledges)
            
            msg = [{"role": "user", "content": request.question}]
            
            async def decorate_answer(answer):
                nonlocal kbinfos
                
                # 插入引用
                answer, idx = await retriever.insert_citations(
                    answer, 
                    [ck["content_ltks"] for ck in kbinfos["chunks"]], 
                    [ck["vector"] for ck in kbinfos["chunks"]], 
                    embd_mdl, 
                    tkweight=0.7, 
                    vtweight=0.3
                )
                
                idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
                recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
                if not recall_docs:
                    recall_docs = kbinfos["doc_aggs"]
                kbinfos["doc_aggs"] = recall_docs
                
                refs = deepcopy(kbinfos)
                for c in refs["chunks"]:
                    if c.get("vector"):
                        del c["vector"]
                
                if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
                    answer += " 请在'用户设置 -> 模型提供商 -> API密钥'中设置LLM API密钥"
                
                refs["chunks"] = chunks_format(refs)
                return {"answer": answer, "reference": refs}
            
            # 生成回答
            if is_stream:
                # 流式生成
                answer = ""
                for ans in await chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
                    answer = ans
                    yield {"answer": answer, "reference": {}}
                
                # 最后返回完整结果和引用信息
                final_response = await decorate_answer(answer)
                final_response["prompt"] = prompt
                final_response["created_at"] = time.time()
                yield final_response
            else:
                # 非流式生成
                answer = await chat_mdl.chat(prompt, msg, {"temperature": 0.1})
                final_response = await decorate_answer(answer)
                final_response["prompt"] = prompt
                final_response["created_at"] = time.time()
                yield final_response
                
        except Exception as e:
            logging.error(f"问答服务执行失败: {e}")
            error_response = {
                "answer": f"抱歉，处理您的问题时出现了错误：{str(e)}",
                "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                "prompt": "",
                "created_at": time.time()
            }
            yield error_response

    @staticmethod
    async def chat(
        session: AsyncSession,
        messages: List[ChatMessage],
        user_id: str,
        kb_ids: List[str],
        doc_ids: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        top_n: int = 12,
        similarity_threshold: float = 0.1,
        vector_similarity_weight: float = 0.3,
        top_k: int = 5,
        enable_quote: bool = True,
        enable_multiturn_refine: bool = False,
        target_language: Optional[str] = None,
        enable_keyword_extraction: bool = False,
        use_kg: bool = False,
        tavily_api_key: Optional[str] = None,
        temperature: float = 0.1,
        is_stream: bool = True
    ):
        """多轮对话服务"""
        try:
            # 验证最后一条消息必须是用户消息
            if not messages or messages[-1].role != "user":
                raise ValueError("最后一条消息必须是用户消息")
            
            # 获取知识库信息
            kbs = await KBService.get_kb_by_ids(session, kb_ids)
            if not kbs:
                raise ValueError("知识库不存在")
            
            # 检查知识库是否使用相同的嵌入模型
            embedding_list = list(set([kb.embd_model_name for kb in kbs]))
            if len(embedding_list) > 1:
                raise ValueError("知识库使用了不同的嵌入模型，无法同时检索")
            
            # 创建模型
            embd_mdl = LLMBundle(kbs[0].tenant_id, LLMType.EMBEDDING, provider=kbs[0].embd_provider_name, model=kbs[0].embd_model_name)
            chat_mdl = LLMBundle(kbs[0].tenant_id, LLMType.CHAT)
            rerank_mdl = LLMBundle(kbs[0].tenant_id, LLMType.RERANK, provider=kbs[0].rerank_provider_name, model=kbs[0].rerank_model_name)
                        
            max_tokens = chat_mdl.max_length if hasattr(chat_mdl, 'max_length') else 8192
            tenant_ids = list(set([kb.tenant_id for kb in kbs]))
            kb_ids = [kb.id for kb in kbs]

            # 初始化检索器
            retriever = RETRIEVALER

            # 提取最后3条用户消息
            user_messages = [m.content for m in messages if m.role == "user"]
            questions = user_messages[-3:] if len(user_messages) >= 3 else user_messages
            
            # 处理多轮对话问题优化
            if len(questions) > 1 and enable_multiturn_refine:
                # 简化版：直接使用最后一个问题
                questions = questions[-1:]
            else:
                questions = questions[-1:]
            
            # 1. 尝试使用SQL查询（如果知识库支持）
            field_map = await KBService.get_field_map(session, kb_ids)
            if field_map:
                logging.debug(f"Use SQL to retrieval: {questions[-1]}")
                sql_result = await QAService.use_sql(
                    questions[-1], field_map, tenant_ids[0], chat_mdl, enable_quote
                )
                if sql_result:
                    yield sql_result
                    return

            # 2. 问题转化处理目标语言
            if target_language:
                questions = [cross_languages(tenant_ids[0], LLMType.CHAT, questions[0], [target_language])]
            
            # 3. 处理关键词提取
            if enable_keyword_extraction:
                questions[-1] += keyword_extraction(chat_mdl, questions[-1])
            
            # 初始化知识库信息
            kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
            knowledges = []

            # 是否启动推理过程（预埋）
            thought = None
                        
            # 4. 执行知识库检索
            if embd_mdl:
                kbinfos = await retriever.retrieval(
                    question=" ".join(questions),
                    embd_mdl=embd_mdl,
                    tenant_ids=tenant_ids,
                    kb_ids=kb_ids,
                    page=1,
                    page_size=top_n,
                    similarity_threshold=similarity_threshold,
                    vector_similarity_weight=vector_similarity_weight,
                    doc_ids=doc_ids,
                    top=top_k,
                    aggs=False,
                    rank_feature=await label_question(session, " ".join(questions), kbs)
                )

            # 5. 集成Tavily外部知识源
            if tavily_api_key:
                try:
                    tav = Tavily(tavily_api_key)
                    tav_res = tav.retrieve_chunks(" ".join(questions))
                    if tav_res and tav_res.get("chunks"):
                        kbinfos["chunks"].extend(tav_res["chunks"])
                    if tav_res and tav_res.get("doc_aggs"):
                        kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
                except Exception as e:
                    logging.warning(f"Tavily外部知识源检索失败: {e}")
            
            # 6. 集成知识图谱检索
            if use_kg:
                try:
                    ck = KG_RETRIEVALER.retrieval(
                        " ".join(questions), 
                        tenant_ids, 
                        kb_ids, 
                        embd_mdl, 
                        chat_mdl(tenant_ids[0], LLMType.CHAT)
                    )
                    if ck and ck.get("content_with_weight"):
                        kbinfos["chunks"].insert(0, ck)  # 将知识图谱结果插入到最前面
                except Exception as e:
                    logging.warning(f"知识图谱检索失败: {e}")
                        
            # 格式化知识库内容
            knowledges = await kb_prompt(session, kbinfos, max_tokens)

            if not knowledges:
                yield {
                    "answer": "抱歉，没有找到相关信息。",
                    "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                    "prompt": "",
                    "created_at": time.time()
                }
                return
            
            # 构建系统提示词
            system_prompt_content = system_prompt or """
角色：你是一个智能助手，名字叫小智。
任务：基于知识库信息总结并回答用户的问题。
要求和限制：
  - 不要编造信息，特别是数字。
  - 如果知识库中的信息与用户问题无关，请直接说：抱歉，没有找到相关信息。
  - 使用markdown格式回答。
  - 使用用户问题的语言回答。
  - 不要编造信息，特别是数字。

### 知识库信息
{knowledge}

以上是知识库中的信息。
"""
            
            # 构建消息列表
            msg = [{"role": "system", "content": system_prompt_content.format(knowledge="\n------\n" + "\n\n------\n\n".join(knowledges))}]
            
            # 添加历史消息，清理引用标记
            for m in messages:
                if m.role != "system":
                    content = re.sub(r"##\d+\$\$", "", m.content)
                    msg.append({"role": m.role, "content": content})

            # 调整消息长度以适应token限制
            used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))
            assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
            prompt = msg[0]["content"]
            
            def decorate_answer(answer):
                """装饰和格式化最终答案"""
                nonlocal kbinfos, prompt, questions
                
                refs = []

                # 分离思考过程和最终答案
                ans = answer.split("</think>")
                think = ""
                if len(ans) == 2:
                    think = ans[0] + "</think>"
                    answer = ans[1]
                
                # 处理引用插入
                if knowledges and enable_quote:
                    idx = set([])
                    
                    # 如果答案中没有引用标记且启用了嵌入模型，自动插入引用
                    if embd_mdl and not re.search(r"\[ID:([0-9]+)\]", answer):
                        answer, idx = retriever.insert_citations(
                            answer,
                            [ck["content_ltks"] for ck in kbinfos["chunks"]],
                            [ck["vector"] for ck in kbinfos["chunks"]],
                            embd_mdl,
                            tkweight=1 - vector_similarity_weight,
                            vtweight=vector_similarity_weight,
                        )
                    else:
                        # 如果答案中已有引用标记，提取引用索引
                        for match in re.finditer(r"\[ID:([0-9]+)\]", answer):
                            i = int(match.group(1))
                            if i < len(kbinfos["chunks"]):
                                idx.add(i)

                    # 修复错误的引用格式
                    answer, idx = QAService.repair_bad_citation_formats(answer, kbinfos, idx)
                    
                    # 处理文档聚合信息，只保留被引用的文档
                    idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
                    recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
                    if not recall_docs:
                        recall_docs = kbinfos["doc_aggs"]
                    kbinfos["doc_aggs"] = recall_docs
                    
                    # 准备引用信息，移除向量数据以减小响应大小
                    refs = deepcopy(kbinfos)
                    for c in refs["chunks"]:
                        if c.get("vector"):
                            del c["vector"]
                
                # 处理API密钥错误
                if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
                    answer += " 请在'用户设置 -> 模型提供商 -> API密钥'中设置LLM API密钥"
                
                # 计算token使用情况
                tk_num = num_tokens_from_string(think + answer)
                prompt += "\n\n### Query:\n%s" % " ".join(questions)
                prompt += f"\n\n## Token usage:\n  - Generated tokens(approximately): {tk_num}\n"
                
                return {"answer": answer, "reference": refs, "prompt": prompt}
            
            # 生成回答
            if is_stream:
                # 流式生成
                last_ans = ""
                answer = ""
                thought = False  # 先设置为False，后续可以自己添加
                
                # 流式生成答案
                for ans in await chat_mdl.chat_streamly(msg[0]["content"], msg[1:], {"temperature": temperature}):
                    # 如果存在思考过程，移除思考部分
                    if thought:
                        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
                    answer = ans
                    
                    # 计算增量内容
                    delta_ans = ans[len(last_ans):]
                    
                    # 如果增量内容token数太少，跳过此次输出
                    if num_tokens_from_string(delta_ans) < 16:
                        continue
                        
                    last_ans = answer
                    
                    # 返回增量内容
                    yield {"answer": thought + answer, "reference": {}}
                
                # 处理最后的增量内容
                delta_ans = answer[len(last_ans):]
                if delta_ans:
                    yield {"answer": thought + answer, "reference": {}}
                    
                # 返回最终装饰后的完整答案
                yield decorate_answer(answer)
            else:
                # 非流式输出模式：一次性生成完整答案
                answer = await chat_mdl.chat(msg[0]["content"], msg[1:], {"temperature": temperature})
                
                # 记录对话日志
                user_content = msg[-1].get("content", "[content not available]")
                logging.debug("User: {}|Assistant: {}".format(user_content, answer))
                
                # 装饰答案
                yield decorate_answer(answer)
                
        except Exception as e:
            logging.error(f"多轮对话服务执行失败: {e}")
            error_response = {
                "answer": f"抱歉，处理您的问题时出现了错误：{str(e)}",
                "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                "prompt": "",
                "created_at": time.time()
            }
            yield error_response

    @staticmethod
    async def use_sql(question: str, field_map: dict, tenant_id: str, chat_mdl, quota: bool = True):
        """
        使用SQL查询结构化数据
        
        此函数使用LLM将自然语言问题转换为SQL查询，然后执行查询并返回格式化的结果。
        主要用于处理结构化数据（如简历、表格等）的查询。
        
        Args:
            question (str): 用户的自然语言问题
            field_map (dict): 字段映射，描述数据库表结构
            tenant_id (str): 租户ID
            chat_mdl: 聊天模型实例，用于生成SQL
            quota (bool): 是否启用引用功能，默认True
            
        Returns:
            dict: 包含answer、reference、prompt字段的响应字典，如果失败则返回None
        """
        # 构建系统提示词
        sys_prompt = "You are a Database Administrator. You need to check the fields of the following tables based on the user's list of questions and write the SQL corresponding to the last question."
        
        # 构建用户提示词，包含表结构和问题
        user_prompt = f"""
Table name: {tenant_id};
Table of database fields are as follows:
{chr(10).join([f"{k}: {v}" for k, v in field_map.items()])}

Question are as follows:
{question}
Please write the SQL, only SQL, without any other explanations or text.
"""
        tried_times = 0

        def get_table():
            """
            获取SQL查询结果
            
            使用LLM生成SQL，执行查询并返回结果。
            包含SQL清理和优化逻辑。
            
            Returns:
                tuple: (查询结果表, SQL语句)
            """
            nonlocal sys_prompt, user_prompt, question, tried_times
            
            # 使用LLM生成SQL
            sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.06})
            
            # 清理LLM输出，移除思考过程
            sql = re.sub(r"^.*</think>", "", sql, flags=re.DOTALL)
            logging.debug(f"{question} ==> {user_prompt} get SQL: {sql}")
            
            # SQL清理和标准化
            sql = re.sub(r"[\r\n]+", " ", sql.lower())  # 移除换行符
            sql = re.sub(r".*select ", "select ", sql.lower())  # 确保以SELECT开头
            sql = re.sub(r" +", " ", sql)  # 标准化空格
            sql = re.sub(r"([;；]|```).*", "", sql)  # 移除分号后的内容
            
            # 验证SQL格式
            if sql[: len("select ")] != "select ":
                return None, None
                
            # 优化SQL查询，确保包含必要的字段
            if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
                if sql[: len("select *")] != "select *":
                    # 如果不是SELECT *，添加doc_id和docnm_kwd字段
                    sql = "select doc_id,docnm_kwd," + sql[6:]
                else:
                    # 如果是SELECT *，替换为具体字段列表
                    flds = []
                    for k in field_map.keys():
                        if k in forbidden_select_fields4resume:
                            continue
                        if len(flds) > 11:  # 限制字段数量
                            break
                        flds.append(k)
                    sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

            logging.debug(f"{question} get SQL(refined): {sql}")
            tried_times += 1
            
            # 执行SQL查询
            return RETRIEVALER.sql_retrieval(sql, format="json"), sql

        tbl, sql = get_table()
        if tbl is None:
            return None
        if tbl.get("error") and tried_times <= 2:
            user_prompt = f"""
            Table name: {tenant_id};
            Table of database fields are as follows:
            {chr(10).join([f"{k}: {v}" for k, v in field_map.items()])}

            Question are as follows:
            {question}
            Please write the SQL, only SQL, without any other explanations or text.


            The SQL error you provided last time is as follows:
            {sql}

            Error issued by database as follows:
            {tbl["error"]}

            Please correct the error and write SQL again, only SQL, without any other explanations or text.
            """
            tbl, sql = get_table()
            logging.debug("TRY it again: {}".format(sql))

        logging.debug("GET table: {}".format(tbl))
        if tbl.get("error") or len(tbl["rows"]) == 0:
            return None

        docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "doc_id"])
        doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "docnm_kwd"])
        column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

        # compose Markdown table
        columns = (
            "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"], tbl["columns"][i]["name"])) for i in column_idx]) + ("|Source|" if docid_idx and docid_idx else "|")
        )

        line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

        rows = ["|" + "|".join([str(r[i]).replace("None", " ") for i in column_idx]).replace("None", " ") + "|" for r in tbl["rows"]]
        rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
        if quota:
            rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
        else:
            rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
        rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

        if not docid_idx or not doc_name_idx:
            logging.warning("SQL missing field: " + sql)
            return {"answer": "\n".join([columns, line, rows]), "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}

        docid_idx = list(docid_idx)[0]
        doc_name_idx = list(doc_name_idx)[0]
        doc_aggs = {}
        for r in tbl["rows"]:
            if r[docid_idx] not in doc_aggs:
                doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
            doc_aggs[r[docid_idx]]["count"] += 1
        return {
            "answer": "\n".join([columns, line, rows]),
            "reference": {
                "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
                "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()],
            },
            "prompt": sys_prompt,
        }

    @staticmethod
    def repair_bad_citation_formats(answer: str, kbinfos: dict, idx: set):
        """
        修复错误的引用格式
        
        将LLM生成的各种不标准引用格式统一转换为标准的[ID:X]格式。
        这确保了引用的一致性和可解析性。
        
        Args:
            answer (str): 包含引用的答案文本
            kbinfos (dict): 知识库信息，包含chunks列表
            idx (set): 已识别的引用索引集合，会被更新
            
        Returns:
            tuple: (修复后的答案文本, 更新后的引用索引集合)
        """
        max_index = len(kbinfos["chunks"])

        def safe_add(i):
            """
            安全地添加引用索引
            
            检查索引是否在有效范围内，如果是则添加到索引集合中。
            
            Args:
                i (int): 要添加的索引
                
            Returns:
                bool: 是否成功添加
            """
            if 0 <= i < max_index:
                idx.add(i)
                return True
            return False

        def find_and_replace(pattern, group_index=1, repl=lambda i: f"ID:{i}", flags=0):
            """
            查找并替换匹配的引用格式
            
            Args:
                pattern: 正则表达式模式
                group_index: 捕获组索引
                repl: 替换函数
                flags: 正则表达式标志
            """
            nonlocal answer

            def replacement(match):
                """
                替换匹配的引用格式为标准格式
                
                Args:
                    match: 正则表达式匹配对象
                    
                Returns:
                    str: 替换后的文本
                """
                try:
                    i = int(match.group(group_index))
                    if safe_add(i):
                        return f"[{repl(i)}]"
                except Exception:
                    pass
                return match.group(0)

            answer = re.sub(pattern, replacement, answer, flags=flags)

        # 遍历所有错误格式模式并进行修复
        for pattern in BAD_CITATION_PATTERNS:
            find_and_replace(pattern)

        return answer, idx
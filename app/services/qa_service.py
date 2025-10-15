import logging
import time
from copy import deepcopy
from typing import Dict, Any, Generator
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.qa import SingleQaRequest
from app.services.kb_service import KBService
from app.infrastructure.llm.llms import llm_factory, embedding_factory
from app.rag_core.search_api import RETRIEVALER, KG_RETRIEVALER
from app.rag_core.rag.prompts import kb_prompt, chunks_format
from app.rag_core.rag.app.tag import label_question
from app.rag_core.utils import ParserType


class QAService:
    """问答服务类"""

    @staticmethod
    async def qa(
        session: AsyncSession,
        request: SingleQaRequest,
        user_id: str,
        tenant_id: str
    ) -> Generator[Dict[str, Any], None, None]:
        """问答服务"""
        try:
            # 获取知识库信息
            kbs = await KBService.get_kb_by_ids(session, request.kb_ids)
            if not kbs:
                raise ValueError("知识库不存在")
            
            # 验证用户权限
            for kb in kbs:
                if kb.owner_id != user_id:
                    raise ValueError(f"无权限访问知识库: {kb.name}")
            
            # 检查知识库是否使用相同的嵌入模型
            embedding_list = list(set([kb.embd_model_name for kb in kbs]))
            if len(embedding_list) > 1:
                raise ValueError("知识库使用了不同的嵌入模型，无法同时检索")
            
            # 检查是否为知识图谱模式
            is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
            retriever = KG_RETRIEVALER if is_knowledge_graph else RETRIEVALER
            
            # 创建模型
            embd_mdl = embedding_factory.create_model()
            chat_mdl = llm_factory.create_model()
            
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
                rank_feature=await label_question(request.question, kbs)
            )
            
            # 格式化知识库内容
            knowledges = kb_prompt(kbinfos, max_tokens)
            
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
            
            def decorate_answer(answer):
                nonlocal kbinfos
                
                # 插入引用
                answer, idx = retriever.insert_citations(
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
            if request.is_stream:
                # 流式生成
                answer = ""
                for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
                    answer = ans
                    yield {"answer": answer, "reference": {}}
                
                # 最后返回完整结果和引用信息
                final_response = decorate_answer(answer)
                final_response["prompt"] = prompt
                final_response["audio_binary"] = None
                final_response["created_at"] = time.time()
                yield final_response
            else:
                # 非流式生成
                answer = chat_mdl.chat(prompt, msg, {"temperature": 0.1})
                final_response = decorate_answer(answer)
                final_response["prompt"] = prompt
                final_response["audio_binary"] = None
                final_response["created_at"] = time.time()
                yield final_response
                
        except Exception as e:
            logging.error(f"问答服务执行失败: {e}")
            error_response = {
                "answer": f"抱歉，处理您的问题时出现了错误：{str(e)}",
                "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                "prompt": "",
                "audio_binary": None,
                "created_at": time.time()
            }
            yield error_response


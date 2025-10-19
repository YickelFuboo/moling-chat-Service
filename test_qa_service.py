import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.qa_service import QAService
from app.schemes.qa import SingleQaRequest, ChatRequest, ChatMessage
from app.models.kb import KB
from app.models.document import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQAService:
    """QAService功能测试类"""
    
    @pytest.fixture
    def mock_session(self):
        """模拟数据库会话"""
        return AsyncMock()
    
    @pytest.fixture
    def mock_kb_data(self):
        """模拟知识库数据"""
        kb = KB()
        kb.id = "83c9f0b5d2dc472e81bf6d5da382d2a1"
        kb.name = "积分政策知识库"
        kb.tenant_id = "test_tenant"
        kb.embd_model_name = "text-embedding-ada-002"
        kb.embd_provider_name = "openai"
        kb.rerank_model_name = "bge-reranker-base"
        kb.rerank_provider_name = "huggingface"
        kb.parser_id = "naive"
        return kb
    
    @pytest.fixture
    def mock_document_data(self):
        """模拟文档数据"""
        docs = [
            Document(
                id="eba5baea-dc61-4499-be59-fb599eea7ac5",
                kb_id="83c9f0b5d2dc472e81bf6d5da382d2a1",
                name="北上广深积分入户条件.docx",
                meta_fields={"category": "入户政策", "region": "北上广深"}
            ),
            Document(
                id="3036e8d1-42e9-4fc9-a505-c6b26730a521",
                kb_id="83c9f0b5d2dc472e81bf6d5da382d2a1",
                name="龙岗福田积分入学.docx",
                meta_fields={"category": "入学政策", "region": "龙岗福田"}
            )
        ]
        return docs

    @pytest.mark.asyncio
    async def test_single_ask_basic(self, mock_session, mock_kb_data):
        """测试基础单次问答功能"""
        logger.info("测试用例1: 基础单次问答功能")
        
        # 准备测试数据
        request = SingleQaRequest(
            question="北上广深的积分入学政策是什么？",
            kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"]
        )
        
        # 模拟KBService.get_kb_by_ids
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            # 模拟检索结果
            mock_kbinfos = {
                "total": 2,
                "chunks": [
                    {
                        "content": "北上广深积分入学政策包括基础分和加分项",
                        "content_ltks": "北上广深积分入学政策包括基础分和加分项",
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "vector": [0.1, 0.2, 0.3],
                        "similarity": 0.95
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "doc_name": "北上广深积分入户条件.docx",
                        "count": 1
                    }
                ]
            }
            
            # 模拟retriever.retrieval
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                # 模拟kb_prompt
                with patch('app.services.qa_service.kb_prompt', return_value=["北上广深积分入学政策包括基础分和加分项"]):
                    # 模拟label_question
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        # 模拟chat_mdl.chat
                        with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                            mock_chat_mdl = AsyncMock()
                            mock_chat_mdl.chat.return_value = "根据知识库信息，北上广深的积分入学政策主要包括基础分和加分项两部分。"
                            mock_chat_mdl.max_length = 8192
                            mock_llm_bundle.return_value = mock_chat_mdl
                            
                            # 模拟insert_citations
                            with patch('app.services.qa_service.RETRIEVALER.insert_citations', return_value=("根据知识库信息，北上广深的积分入学政策主要包括基础分和加分项两部分。[ID:0]", [0])):
                                # 执行测试
                                result = None
                                async for response in QAService.single_ask(
                                    session=mock_session,
                                    request=request,
                                    user_id="test_user",
                                    tenant_id="test_tenant",
                                    is_stream=False
                                ):
                                    result = response
                                    break
                                
                                # 验证结果
                                assert result is not None
                                assert "answer" in result
                                assert "reference" in result
                                assert "prompt" in result
                                assert "created_at" in result
                                assert "北上广深" in result["answer"]
                                logger.info(f"测试用例1通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_single_ask_with_keyword_extraction(self, mock_session, mock_kb_data):
        """测试带关键词提取的单次问答"""
        logger.info("测试用例2: 带关键词提取的单次问答")
        
        request = SingleQaRequest(
            question="龙岗区积分入学需要多少积分？",
            kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"]
        )
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            mock_kbinfos = {
                "total": 1,
                "chunks": [
                    {
                        "content": "龙岗区积分入学基础分为100分，加分项最高10分",
                        "content_ltks": "龙岗区积分入学基础分为100分，加分项最高10分",
                        "doc_id": "3036e8d1-42e9-4fc9-a505-c6b26730a521",
                        "vector": [0.2, 0.3, 0.4],
                        "similarity": 0.92
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "3036e8d1-42e9-4fc9-a505-c6b26730a521",
                        "doc_name": "龙岗福田积分入学.docx",
                        "count": 1
                    }
                ]
            }
            
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                with patch('app.services.qa_service.kb_prompt', return_value=["龙岗区积分入学基础分为100分，加分项最高10分"]):
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        with patch('app.services.qa_service.keyword_extraction', return_value=" 关键词: 龙岗区, 积分, 入学"):
                            with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                                mock_chat_mdl = AsyncMock()
                                mock_chat_mdl.chat.return_value = "龙岗区积分入学需要基础分100分，加上加分项最高可达110分。"
                                mock_chat_mdl.max_length = 8192
                                mock_llm_bundle.return_value = mock_chat_mdl
                                
                                with patch('app.services.qa_service.RETRIEVALER.insert_citations', return_value=("龙岗区积分入学需要基础分100分，加上加分项最高可达110分。[ID:0]", [0])):
                                    result = None
                                    async for response in QAService.single_ask(
                                        session=mock_session,
                                        request=request,
                                        user_id="test_user",
                                        tenant_id="test_tenant",
                                        is_stream=False
                                    ):
                                        result = response
                                        break
                                    
                                    assert result is not None
                                    assert "龙岗区" in result["answer"]
                                    assert "100分" in result["answer"]
                                    logger.info(f"测试用例2通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_chat_basic(self, mock_session, mock_kb_data):
        """测试基础多轮对话功能"""
        logger.info("测试用例3: 基础多轮对话功能")
        
        messages = [
            ChatMessage(role="user", content="如何计算自己的积分有多少？")
        ]
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            mock_kbinfos = {
                "total": 2,
                "chunks": [
                    {
                        "content": "积分计算包括基础分和加分项，基础分根据户籍和住房情况确定",
                        "content_ltks": "积分计算包括基础分和加分项，基础分根据户籍和住房情况确定",
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "vector": [0.1, 0.2, 0.3],
                        "similarity": 0.88
                    },
                    {
                        "content": "加分项包括居住时间、独生子女等，最高不超过10分",
                        "content_ltks": "加分项包括居住时间、独生子女等，最高不超过10分",
                        "doc_id": "3036e8d1-42e9-4fc9-a505-c6b26730a521",
                        "vector": [0.2, 0.3, 0.4],
                        "similarity": 0.85
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "doc_name": "北上广深积分入户条件.docx",
                        "count": 1
                    },
                    {
                        "doc_id": "3036e8d1-42e9-4fc9-a505-c6b26730a521",
                        "doc_name": "龙岗福田积分入学.docx",
                        "count": 1
                    }
                ]
            }
            
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                with patch('app.services.qa_service.kb_prompt', return_value=["积分计算包括基础分和加分项", "加分项包括居住时间、独生子女等"]):
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                            mock_chat_mdl = AsyncMock()
                            mock_chat_mdl.chat.return_value = "积分计算包括基础分和加分项两部分。基础分根据您的户籍和住房情况确定，加分项包括居住时间、独生子女等，最高不超过10分。"
                            mock_chat_mdl.max_length = 8192
                            mock_llm_bundle.return_value = mock_chat_mdl
                            
                            with patch('app.services.qa_service.RETRIEVALER.insert_citations', return_value=("积分计算包括基础分和加分项两部分。[ID:0][ID:1]", [0, 1])):
                                result = None
                                async for response in QAService.chat(
                                    session=mock_session,
                                    messages=messages,
                                    user_id="test_user",
                                    kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"],
                                    is_stream=False
                                ):
                                    result = response
                                    break
                                
                                assert result is not None
                                assert "积分计算" in result["answer"]
                                assert "基础分" in result["answer"]
                                assert "加分项" in result["answer"]
                                logger.info(f"测试用例3通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_chat_with_target_language(self, mock_session, mock_kb_data):
        """测试带目标语言转换的多轮对话"""
        logger.info("测试用例4: 带目标语言转换的多轮对话")
        
        messages = [
            ChatMessage(role="user", content="深圳和上海积分计算差异是什么？")
        ]
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            mock_kbinfos = {
                "total": 2,
                "chunks": [
                    {
                        "content": "深圳积分计算主要看户籍、住房、社保等因素",
                        "content_ltks": "深圳积分计算主要看户籍、住房、社保等因素",
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "vector": [0.1, 0.2, 0.3],
                        "similarity": 0.90
                    },
                    {
                        "content": "上海积分计算更注重学历、技能和投资等因素",
                        "content_ltks": "上海积分计算更注重学历、技能和投资等因素",
                        "doc_id": "3036e8d1-42e9-4fc9-a505-c6b26730a521",
                        "vector": [0.2, 0.3, 0.4],
                        "similarity": 0.87
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "doc_name": "北上广深积分入户条件.docx",
                        "count": 1
                    },
                    {
                        "doc_id": "3036e8d1-42e9-4fc9-a505-c6b26730a521",
                        "doc_name": "龙岗福田积分入学.docx",
                        "count": 1
                    }
                ]
            }
            
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                with patch('app.services.qa_service.kb_prompt', return_value=["深圳积分计算主要看户籍、住房、社保等因素", "上海积分计算更注重学历、技能和投资等因素"]):
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        with patch('app.services.qa_service.cross_languages', return_value="What are the differences between Shenzhen and Shanghai's point calculation systems?"):
                            with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                                mock_chat_mdl = AsyncMock()
                                mock_chat_mdl.chat.return_value = "The main differences between Shenzhen and Shanghai's point calculation systems are: Shenzhen focuses more on household registration, housing, and social security, while Shanghai emphasizes education, skills, and investment factors."
                                mock_chat_mdl.max_length = 8192
                                mock_llm_bundle.return_value = mock_chat_mdl
                                
                                with patch('app.services.qa_service.RETRIEVALER.insert_citations', return_value=("The main differences between Shenzhen and Shanghai's point calculation systems are: Shenzhen focuses more on household registration, housing, and social security, while Shanghai emphasizes education, skills, and investment factors.[ID:0][ID:1]", [0, 1])):
                                    result = None
                                    async for response in QAService.chat(
                                        session=mock_session,
                                        messages=messages,
                                        user_id="test_user",
                                        kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"],
                                        target_language="en",
                                        is_stream=False
                                    ):
                                        result = response
                                        break
                                    
                                    assert result is not None
                                    assert "Shenzhen" in result["answer"]
                                    assert "Shanghai" in result["answer"]
                                    assert "differences" in result["answer"]
                                    logger.info(f"测试用例4通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_chat_with_tavily_search(self, mock_session, mock_kb_data):
        """测试带Tavily外部搜索的多轮对话"""
        logger.info("测试用例5: 带Tavily外部搜索的多轮对话")
        
        messages = [
            ChatMessage(role="user", content="最新的积分政策有什么变化？")
        ]
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            mock_kbinfos = {
                "total": 1,
                "chunks": [
                    {
                        "content": "2024年积分政策有所调整，基础分标准提高",
                        "content_ltks": "2024年积分政策有所调整，基础分标准提高",
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "vector": [0.1, 0.2, 0.3],
                        "similarity": 0.85
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "doc_name": "北上广深积分入户条件.docx",
                        "count": 1
                    }
                ]
            }
            
            # 模拟Tavily搜索结果
            mock_tavily_result = {
                "chunks": [
                    {
                        "content": "2024年最新政策显示，积分标准有所上调",
                        "content_ltks": "2024年最新政策显示，积分标准有所上调",
                        "doc_id": "tavily_001",
                        "vector": [0.3, 0.4, 0.5],
                        "similarity": 0.88
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "tavily_001",
                        "doc_name": "最新政策文件",
                        "count": 1
                    }
                ]
            }
            
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                with patch('app.services.qa_service.kb_prompt', return_value=["2024年积分政策有所调整，基础分标准提高", "2024年最新政策显示，积分标准有所上调"]):
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        with patch('app.services.qa_service.Tavily') as mock_tavily:
                            mock_tavily_instance = AsyncMock()
                            mock_tavily_instance.retrieve_chunks.return_value = mock_tavily_result
                            mock_tavily.return_value = mock_tavily_instance
                            
                            with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                                mock_chat_mdl = AsyncMock()
                                mock_chat_mdl.chat.return_value = "根据最新信息，2024年积分政策确实有所调整，基础分标准有所提高，建议关注官方最新发布。"
                                mock_chat_mdl.max_length = 8192
                                mock_llm_bundle.return_value = mock_chat_mdl
                                
                                with patch('app.services.qa_service.RETRIEVALER.insert_citations', return_value=("根据最新信息，2024年积分政策确实有所调整，基础分标准有所提高，建议关注官方最新发布。[ID:0][ID:1]", [0, 1])):
                                    result = None
                                    async for response in QAService.chat(
                                        session=mock_session,
                                        messages=messages,
                                        user_id="test_user",
                                        kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"],
                                        tavily_api_key="test_api_key",
                                        is_stream=False
                                    ):
                                        result = response
                                        break
                                    
                                    assert result is not None
                                    assert "2024年" in result["answer"]
                                    assert "政策" in result["answer"]
                                    assert "调整" in result["answer"]
                                    logger.info(f"测试用例5通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_chat_with_kg_search(self, mock_session, mock_kb_data):
        """测试带知识图谱搜索的多轮对话"""
        logger.info("测试用例6: 带知识图谱搜索的多轮对话")
        
        messages = [
            ChatMessage(role="user", content="积分政策相关的法律法规有哪些？")
        ]
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            mock_kbinfos = {
                "total": 1,
                "chunks": [
                    {
                        "content": "积分政策依据《居住证暂行条例》等法律法规制定",
                        "content_ltks": "积分政策依据《居住证暂行条例》等法律法规制定",
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "vector": [0.1, 0.2, 0.3],
                        "similarity": 0.90
                    }
                ],
                "doc_aggs": [
                    {
                        "doc_id": "eba5baea-dc61-4499-be59-fb599eea7ac5",
                        "doc_name": "北上广深积分入户条件.docx",
                        "count": 1
                    }
                ]
            }
            
            # 模拟知识图谱搜索结果
            mock_kg_result = {
                "content_with_weight": "相关法律法规包括《居住证暂行条例》、《积分入户管理办法》等",
                "doc_id": "kg_001",
                "similarity": 0.95
            }
            
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                with patch('app.services.qa_service.kb_prompt', return_value=["积分政策依据《居住证暂行条例》等法律法规制定", "相关法律法规包括《居住证暂行条例》、《积分入户管理办法》等"]):
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        with patch('app.services.qa_service.KG_RETRIEVALER.retrieval', return_value=mock_kg_result):
                            with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                                mock_chat_mdl = AsyncMock()
                                mock_chat_mdl.chat.return_value = "积分政策相关的法律法规主要包括《居住证暂行条例》、《积分入户管理办法》等，这些法规为积分制度的实施提供了法律依据。"
                                mock_chat_mdl.max_length = 8192
                                mock_llm_bundle.return_value = mock_chat_mdl
                                
                                with patch('app.services.qa_service.RETRIEVALER.insert_citations', return_value=("积分政策相关的法律法规主要包括《居住证暂行条例》、《积分入户管理办法》等，这些法规为积分制度的实施提供了法律依据。[ID:0][ID:1]", [0, 1])):
                                    result = None
                                    async for response in QAService.chat(
                                        session=mock_session,
                                        messages=messages,
                                        user_id="test_user",
                                        kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"],
                                        use_kg=True,
                                        is_stream=False
                                    ):
                                        result = response
                                        break
                                    
                                    assert result is not None
                                    assert "法律法规" in result["answer"]
                                    assert "居住证暂行条例" in result["answer"]
                                    assert "积分入户管理办法" in result["answer"]
                                    logger.info(f"测试用例6通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_error_handling_no_kb(self, mock_session):
        """测试知识库不存在的错误处理"""
        logger.info("测试用例7: 知识库不存在的错误处理")
        
        request = SingleQaRequest(
            question="测试问题",
            kb_ids=["non_existent_kb"]
        )
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[]):
            result = None
            async for response in QAService.single_ask(
                session=mock_session,
                request=request,
                user_id="test_user",
                tenant_id="test_tenant",
                is_stream=False
            ):
                result = response
                break
            
            assert result is not None
            assert "抱歉" in result["answer"]
            assert "错误" in result["answer"]
            logger.info(f"测试用例7通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_error_handling_no_knowledge(self, mock_session, mock_kb_data):
        """测试没有找到相关知识的错误处理"""
        logger.info("测试用例8: 没有找到相关知识的错误处理")
        
        messages = [
            ChatMessage(role="user", content="完全不相关的问题")
        ]
        
        with patch('app.services.qa_service.KBService.get_kb_by_ids', return_value=[mock_kb_data]):
            mock_kbinfos = {
                "total": 0,
                "chunks": [],
                "doc_aggs": []
            }
            
            with patch('app.services.qa_service.RETRIEVALER.retrieval', return_value=mock_kbinfos):
                with patch('app.services.qa_service.kb_prompt', return_value=[]):
                    with patch('app.services.qa_service.label_question', return_value=[]):
                        result = None
                        async for response in QAService.chat(
                            session=mock_session,
                            messages=messages,
                            user_id="test_user",
                            kb_ids=["83c9f0b5d2dc472e81bf6d5da382d2a1"],
                            is_stream=False
                        ):
                            result = response
                            break
                        
                        assert result is not None
                        assert "抱歉" in result["answer"]
                        assert "没有找到相关信息" in result["answer"]
                        logger.info(f"测试用例8通过: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_use_sql_functionality(self, mock_session):
        """测试SQL查询功能"""
        logger.info("测试用例9: SQL查询功能")
        
        # 模拟字段映射
        field_map = {
            "doc_id": "文档ID",
            "docnm_kwd": "文档名称",
            "content": "内容",
            "category": "类别",
            "region": "地区"
        }
        
        # 模拟SQL查询结果
        mock_sql_result = {
            "columns": [
                {"name": "doc_id"},
                {"name": "docnm_kwd"},
                {"name": "category"},
                {"name": "region"}
            ],
            "rows": [
                ["eba5baea-dc61-4499-be59-fb599eea7ac5", "北上广深积分入户条件.docx", "入户政策", "北上广深"],
                ["3036e8d1-42e9-4fc9-a505-c6b26730a521", "龙岗福田积分入学.docx", "入学政策", "龙岗福田"]
            ]
        }
        
        with patch('app.services.qa_service.RETRIEVALER.sql_retrieval', return_value=mock_sql_result):
            with patch('app.services.qa_service.LLMBundle') as mock_llm_bundle:
                mock_chat_mdl = AsyncMock()
                mock_chat_mdl.chat.return_value = "SELECT doc_id, docnm_kwd, category, region FROM test_table WHERE category LIKE '%积分%'"
                mock_llm_bundle.return_value = mock_chat_mdl
                
                result = await QAService.use_sql(
                    question="查询所有积分相关的政策文档",
                    field_map=field_map,
                    tenant_id="test_tenant",
                    chat_mdl=mock_chat_mdl,
                    quota=True
                )
                
                assert result is not None
                assert "answer" in result
                assert "reference" in result
                assert "prompt" in result
                assert "入户政策" in result["answer"]
                assert "入学政策" in result["answer"]
                logger.info(f"测试用例9通过: SQL查询功能正常")

    @pytest.mark.asyncio
    async def test_repair_bad_citation_formats(self):
        """测试修复错误引用格式功能"""
        logger.info("测试用例10: 修复错误引用格式功能")
        
        answer = "这是答案内容 (ID: 0) 和 [ID: 1] 以及【ID: 2】的引用"
        kbinfos = {
            "chunks": [
                {"content": "内容1"},
                {"content": "内容2"},
                {"content": "内容3"}
            ]
        }
        idx = set()
        
        result_answer, result_idx = QAService.repair_bad_citation_formats(answer, kbinfos, idx)
        
        assert "[ID:0]" in result_answer
        assert "[ID:1]" in result_answer
        assert "[ID:2]" in result_answer
        assert 0 in result_idx
        assert 1 in result_idx
        assert 2 in result_idx
        logger.info(f"测试用例10通过: 引用格式修复正常")


async def run_tests():
    """运行所有测试"""
    logger.info("开始运行QAService功能测试...")
    
    test_instance = TestQAService()
    
    # 运行所有测试用例
    test_cases = [
        test_instance.test_single_ask_basic,
        test_instance.test_single_ask_with_keyword_extraction,
        test_instance.test_chat_basic,
        test_instance.test_chat_with_target_language,
        test_instance.test_chat_with_tavily_search,
        test_instance.test_chat_with_kg_search,
        test_instance.test_error_handling_no_kb,
        test_instance.test_error_handling_no_knowledge,
        test_instance.test_use_sql_functionality,
        test_instance.test_repair_bad_citation_formats
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"运行测试用例 {i}/{len(test_cases)}")
            await test_case()
            passed += 1
            logger.info(f"✅ 测试用例 {i} 通过")
        except Exception as e:
            failed += 1
            logger.error(f"❌ 测试用例 {i} 失败: {str(e)}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试完成! 通过: {passed}, 失败: {failed}")
    logger.info(f"成功率: {passed/(passed+failed)*100:.1f}%")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_tests())

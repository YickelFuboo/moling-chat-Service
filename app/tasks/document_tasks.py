import asyncio
import logging
from celery import shared_task
from app.models.document import Document
from app.services.common.doc_parser_service import DocParserService

@shared_task(bind=True, name="parse_document_task")
def parse_document_task(document_id: str, user_id: str, delete_old: bool = False):
    """
    文档解析任务 - Celery任务（主任务）
    
    Args:
        document_id: 文档ID
        user_id: 用户ID
        delete_old: 是否删除旧数据
    """
    try:
        # 添加详细的调试信息
        logging.info("=" * 50)
        logging.info(f"🚀 [CELERY] 开始执行文档解析任务")
        logging.info(f"📋 [CELERY] 任务参数: document_id={document_id}, user_id={user_id}, delete_old={delete_old}")
        logging.info("=" * 50)
        
        if not document_id:
            raise ValueError("文档ID不能为空")

        logging.info(f"✅ [CELERY] 参数验证通过，开始处理文档: {document_id}")
        
        # 调试断点 - 取消注释下面的行来启用调试
        # import pdb; pdb.set_trace()  # 基础调试器
        # import ipdb; ipdb.set_trace()  # 增强调试器
        
        # 定义异步任务
        async def async_task():
            # 在异步上下文中获取文档对象
            async def get_document():
                logging.info(f"🔍 [CELERY] 开始从数据库获取文档: {document_id}")
                from app.infrastructure.database import get_db
                async for session in get_db():
                    document = await session.get(Document, document_id)
                    if not document:
                        raise ValueError(f"文档不存在: {document_id}")
                    logging.info(f"✅ [CELERY] 成功获取文档: {document.name}")
                    return document
            
            # 获取文档对象
            document = await get_document()
            
            logging.info(f"🛠️ [CELERY] 创建文档解析服务...")
            # 创建文档解析服务
            docparser = DocParserService(document, user_id, delete_old)

            logging.info(f"⚡ [CELERY] 开始执行文档解析...")
            # 运行异步任务
            result = await docparser.parse_document()
            logging.info(f"🎉 [CELERY] 文档解析任务完成: {document_id}")
            logging.info(f"📊 [CELERY] 解析结果: {result}")
            return result
        
        # 直接使用asyncio.run()运行异步任务
        logging.info(f"🔄 [CELERY] 使用asyncio.run()执行异步任务...")
        result = asyncio.run(async_task())
        return result
            
    except Exception as e:
        logging.error(f"❌ [CELERY] 文档解析任务失败: {document_id}")
        logging.error(f"💥 [CELERY] 错误详情: {str(e)}")
        logging.error(f"📋 [CELERY] 错误类型: {type(e).__name__}")
        import traceback
        logging.error(f"🔍 [CELERY] 错误堆栈: {traceback.format_exc()}")
        raise

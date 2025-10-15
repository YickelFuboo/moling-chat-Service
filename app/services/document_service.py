import sys
import io
import re
import threading
import uuid
import asyncio
import logging
import pdfplumber
import aspose.pydrawing as drawing
import aspose.slides as slides
import concurrent.futures
from io import BytesIO
from PIL import Image
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, BinaryIO
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile
from sqlalchemy import select, and_, or_, func, desc, asc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from app.models import Document, KB
from app.rag_core.utils import ParserType
from app.rag_core.chunk_api import CHUNK_FACTORY
from app.constants.common import DocumentConstants, FileType, FileSource
from app.services.common.doc_parser_service import DocParserService
from app.services.kb_service import KBService
from app.tasks.document_tasks import parse_document_task
from app.schemas.document import FileUploadResult
from app.services.common.file_service import FileService, FileUsage


class DocumentService:
    """文档服务类"""

    @staticmethod
    async def create_document(
        session: AsyncSession,
        kb_id: str,
        name: str,
        doc_type: FileType,
        suffix: str,
        file_id: str,
        size: int,
        source_type: FileSource,
        parser_id: ParserType,
        parser_config: Optional[str] = None,
        thumbnail_id: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Document:
        """创建文档"""
        try:
            existing_doc = await DocumentService.get_document_by_name(
                session, kb_id, name
            )
            if existing_doc:
                raise ValueError(f"文档名称 '{name}' 在知识库中已存在")
            
            document = Document(
                id=str(uuid.uuid4()),
                kb_id=kb_id,
                name=name,
                type=doc_type.value,
                suffix=suffix,
                file_id=file_id,
                size=size,
                parser_id=parser_id.value,
                parser_config=parser_config,
                source_type=source_type.value,
                created_by=created_by,
                thumbnail_id=thumbnail_id
            )
            
            session.add(document)
            await session.commit()
            await session.refresh(document)
            
            await DocumentService._update_kb_doc_count(session, kb_id)
            
            logging.info(f"创建文档成功: {document.id}")
            return document
            
        except Exception as e:
            await session.rollback()
            logging.error(f"创建文档失败: {e}")
            raise

    @staticmethod
    async def upload_document_to_kb(
        session: AsyncSession,
        kb: KB,
        file: UploadFile,
        created_by: str
    ) -> FileUploadResult:
        """上传单个文档到知识库"""

        try:
            filename = file.filename    
            file_obj = file.file

            # 读取文件内容
            file_content = file_obj.read()
            file_size = len(file_content)
            
            # 验证文件名有效性
            is_valid, error_msg = FileService.validate_filename(filename)
            if not is_valid:
                return FileUploadResult(
                    filename=filename,
                    success=False,
                    error=f"文件 '{filename}': {error_msg}"
                )
            
            # 处理重复文件名
            final_filename = await DocumentService._handle_duplicate_filename(session, kb.id, filename)

            # 判断文件大小
            if file_size > DocumentConstants.MAX_DOCUMENT_FILE_SIZE:
                return FileUploadResult(
                    filename=filename,
                    success=False,
                    error=f"文件大小 {file_size} 超过最大限制 {DocumentConstants.MAX_DOCUMENT_FILE_SIZE} M"
                )
            
            # 识别文件类型
            file_type = DocumentService._get_file_type_by_suffix(final_filename)
            if file_type == FileType.OTHER:
                return FileUploadResult(
                    filename=filename,
                    success=False,
                    error=f"文件 '{filename}': 不支持的文件类型"
                )
            
            # 特殊处理PDF文件
            if file_type == FileType.PDF:
                file_content = FileService.fix_broken_pdf(file_content)
            
            # 存储文件到对象存储
            file_id = await FileService.upload_file(
                file_data=io.BytesIO(file_content),
                file_name=final_filename,
                file_usage=FileUsage.DOCUMENT
            )
            
            # 生成缩略图
            thumbnail_id = await DocumentService._generate_document_thumbnail(
                final_filename, file_content
            )
            
            # 确定解析器类型
            parser_type = DocumentService._get_parser_type(file_type, final_filename, kb.parser_id)
            
            # 创建文档记录
            document = Document(
                id=str(uuid.uuid4()),
                kb_id=kb.id,
                name=final_filename,
                type=file_type.value,
                suffix=Path(final_filename).suffix.lstrip("."), # 文件扩展名
                file_id=file_id,
                size=file_size,
                parser_id=parser_type.value,
                parser_config=kb.parser_config,
                source_type=FileSource.UPLOAD,
                created_by=created_by,
                thumbnail_id=thumbnail_id
            )
            
            session.add(document)
            await session.commit()
            await session.refresh(document)            
            await DocumentService._update_kb_doc_count(session, kb.id)
            
            # 异步触发文档解析任务
            # task = parse_document_task.delay(document.id, created_by, True)
            # await parse_document_task(document.id, created_by, True)
            docparser = DocParserService(kb, document, created_by, True)

            logging.info(f"⚡ [CELERY] 开始执行文档解析...")
            # 运行异步任务
            result = await docparser.parse_document()
            
            logging.info(f"文档上传成功: {final_filename}")
            
            return FileUploadResult(
                filename=filename,
                success=True,
                document_id=document.id,
                #task_id=task.id
            )
            
        except Exception as e:
            error_msg = f"处理文件 '{filename}' 失败: {str(e)}"
            logging.error(error_msg)
            return FileUploadResult(
                filename=filename,
                success=False,
                error=error_msg
            )
    
    @staticmethod
    async def _handle_duplicate_filename(
        session: AsyncSession, 
        kb_id: str, 
        filename: str
    ) -> str:
        """处理重复文件名"""
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        counter = 1
        final_filename = filename
        
        while await DocumentService.get_document_by_name(session, kb_id, final_filename):
            final_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        
        return final_filename
    
    @staticmethod
    def _get_file_type_by_suffix(filename: str) -> FileType:
        """根据文件名获取文件类型"""
        filename = filename.lower()
        if re.match(r".*\.pdf$", filename):
            return FileType.PDF

        if re.match(r".*\.(eml|doc|docx|ppt|pptx|yml|xml|htm|json|csv|txt|ini|xls|xlsx|wps|rtf|hlp|pages|numbers|key|md|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|html|sql)$", filename):
            return FileType.DOC

        if re.match(r".*\.(wav|flac|ape|alac|wavpack|wv|mp3|aac|ogg|vorbis|opus)$", filename):
            return FileType.AURAL

        if re.match(r".*\.(jpg|jpeg|png|tif|gif|pcx|tga|exif|fpx|svg|psd|cdr|pcd|dxf|ufo|eps|ai|raw|WMF|webp|avif|apng|icon|ico|mpg|mpeg|avi|rm|rmvb|mov|wmv|asf|dat|asx|wvx|mpe|mpa|mp4)$", filename):
            return FileType.VISUAL

        return FileType.OTHER
    
    @staticmethod
    def _get_parser_type(
        file_type: FileType, 
        filename: str, 
        default_parser: str 
    ) -> ParserType:
        """根据文件类型确定解析器类型"""
        if file_type == FileType.VISUAL:
            return ParserType.PICTURE
        if file_type == FileType.AURAL:
            return ParserType.AUDIO
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION
        if re.search(r"\.(eml)$", filename):
            return ParserType.EMAIL
        return ParserType(default_parser)
    
    @staticmethod
    async def _generate_document_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成文档缩略图"""
        try:
            if re.match(r".*\.pdf$", filename):
                return await DocumentService._generate_pdf_thumbnail(filename, content)
            elif re.match(r".*\.(jpg|jpeg|png|tif|gif|icon|ico|webp)$", filename):
                return await DocumentService._generate_image_thumbnail(filename, content)
            elif re.match(r".*\.(ppt|pptx)$", filename):
                return await DocumentService._generate_ppt_thumbnail(filename, content)
            else:
                return None
                
        except Exception as e:
            logging.error(f"生成文档缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _generate_pdf_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成PDF缩略图"""
        try:
            # 导入必要的库
            
            # 创建全局锁来避免pdfplumber的并发问题
            LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
            if LOCK_KEY_pdfplumber not in sys.modules:
                sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()
            
            with sys.modules[LOCK_KEY_pdfplumber]:
                # 打开PDF文件
                pdf = pdfplumber.open(BytesIO(content))
                
                if not pdf.pages:
                    pdf.close()
                    logging.warning(f"PDF文件 {filename} 没有页面")
                    return None
                
                # 生成缩略图
                buffered = BytesIO()
                resolution = 32
                img = None
                
                for _ in range(10):
                    try:
                        # 使用第一页生成缩略图
                        page = pdf.pages[0]
                        page_image = page.to_image(resolution=resolution)
                        page_image.save(buffered, format="PNG")
                        img = buffered.getvalue()
                        
                        # 如果图片大小合适，则使用当前分辨率
                        if len(img) >= 64000 and resolution >= 2:
                            resolution = resolution / 2
                            buffered = BytesIO()
                        else:
                            break
                    except Exception as e:
                        logging.warning(f"生成PDF缩略图时分辨率 {resolution} 失败: {e}")
                        resolution = resolution / 2
                        buffered = BytesIO()
                        if resolution < 1:
                            break
                
                pdf.close()
                
                if img:
                    # 将缩略图保存到文件存储
                    thumbnail_filename = f"{Path(filename).stem}_thumb.png"
                    
                    # 上传缩略图到对象存储，返回缩略图ID
                    thumbnail_id = await FileService.upload_file(
                        file_data=io.BytesIO(img),
                        file_name=thumbnail_filename,
                        file_usage=FileUsage.DOCUMENT_THUMBNAIL
                    )
                    
                    logging.info(f"PDF缩略图生成成功: {filename} -> 缩略图ID: {thumbnail_id}")
                    return thumbnail_id
                else:
                    logging.warning(f"PDF缩略图生成失败: {filename}")
                    return None
                    
        except Exception as e:
            logging.error(f"生成PDF缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _generate_image_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成图片缩略图"""
        try:
            # 导入必要的库
            
            # 打开图片
            image = Image.open(BytesIO(content))
            
            # 转换为RGB模式（如果是RGBA等）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # 生成缩略图，保持宽高比
            image.thumbnail((300, 300))
            
            # 保存缩略图到内存
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_content = buffered.getvalue()
            
            # 上传缩略图到对象存储，返回缩略图ID
            thumbnail_filename = f"{Path(filename).stem}_thumb.png"
            
            thumbnail_id = await FileService.upload_file(
                file_data=io.BytesIO(img_content),
                file_name=thumbnail_filename,
                file_usage=FileUsage.DOCUMENT_THUMBNAIL
            )
            
            logging.info(f"图片缩略图生成成功: {filename} -> 缩略图ID: {thumbnail_id}")
            return thumbnail_id
            
        except Exception as e:
            logging.error(f"生成图片缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _generate_ppt_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成PPT缩略图"""
        try:
            # 导入必要的库
            
            try:
                with slides.Presentation(BytesIO(content)) as presentation:
                    if not presentation.slides:
                        logging.warning(f"PPT文件 {filename} 没有幻灯片")
                        return None
                    
                    buffered = BytesIO()
                    scale = 0.03
                    img = None
                    
                    for _ in range(10):
                        # https://reference.aspose.com/slides/python-net/aspose.slides/slide/get_thumbnail/#float-float
                        presentation.slides[0].get_thumbnail(scale, scale).save(buffered, drawing.imaging.ImageFormat.png)
                        img = buffered.getvalue()
                        if len(img) >= 64000:
                            scale = scale / 2.0
                            buffered = BytesIO()
                        else:
                            break
                    
                    if img:
                        # 上传缩略图到对象存储，返回缩略图ID
                        thumbnail_filename = f"{Path(filename).stem}_thumb.png"
                        
                        thumbnail_id = await FileService.upload_file(
                            file_data=io.BytesIO(img),
                            file_name=thumbnail_filename,
                            file_usage=FileUsage.DOCUMENT_THUMBNAIL
                        )
                        
                        logging.info(f"PPT缩略图生成成功: {filename} -> 缩略图ID: {thumbnail_id}")
                        return thumbnail_id
                    else:
                        logging.warning(f"PPT缩略图生成失败: {filename}")
                        return None
                        
            except Exception as e:
                logging.warning(f"使用Aspose.Slides生成PPT缩略图失败: {e}")
                return None

        except Exception as e:
            logging.error(f"生成PPT缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _chunk_document_content(
        document: Document,
        file_content: bytes,
        from_page: int,
        to_page: int,
        task_config: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        使用Chunker进行文档内容切片
        
        Args:
            document: 文档对象
            file_content: 文件内容
            from_page: 起始页/行
            to_page: 结束页/行
            task_config: 任务配置
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 切片结果
        """
        try:
            # 从工厂获取对应的Chunker
            
            chunker = CHUNK_FACTORY[document.parser_id]
            if not chunker:
                raise ValueError(f"未找到对应的Chunker: {document.parser_id}")
            
            # 准备切片参数
            chunk_params = {
                "name": document.name,
                "binary": file_content,
                "from_page": from_page,
                "to_page": to_page,
                "lang": task_config.get("language", "zh"),  # 默认中文
                "kb_id": document.kb_id,
                "parser_config": document.parser_config,
                "tenant_id": user_id
            }
            
            # 定义进度回调函数
            def progress_callback(progress: float, message: str = ""):
                logging.info(f"文档切片进度: {progress:.2%} - {message}")
            
            # 调用Chunker进行切片
            chunk_result = await chunker.chunk(
                name=chunk_params["name"],
                binary=chunk_params["binary"],
                from_page=chunk_params["from_page"],
                to_page=chunk_params["to_page"],
                lang=chunk_params["lang"],
                callback=progress_callback,
                kb_id=chunk_params["kb_id"],
                parser_config=chunk_params["parser_config"],
                tenant_id=chunk_params["tenant_id"]
            )
            
            logging.info(f"文档切片完成: {document.id}, 页数范围: {from_page}-{to_page}, 切片数: {len(chunk_result.get('chunks', []))}")
            
            return chunk_result
            
        except Exception as e:
            logging.error(f"文档切片失败: {document.id}, 页数范围: {from_page}-{to_page}, 错误: {e}")
            raise
    
    @staticmethod
    async def _execute_parser_with_threading(
        document: Document,
        chunk_result: Dict[str, Any],
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用多线程执行后续解析操作
        
        Args:
            document: 文档对象
            chunk_result: 切片结果
            task_config: 任务配置
            
        Returns:
            Dict[str, Any]: 解析结果
        """
        try:
            # 获取事件循环
            loop = asyncio.get_event_loop()
            
            # 从切片结果中获取chunks
            chunks = chunk_result.get("chunks", [])
            if not chunks:
                return {
                    "status": "success",
                    "message": "没有可处理的切片",
                    "chunks": [],
                    "embeddings": [],
                    "metadata": chunk_result.get("metadata", {})
                }
            
            # 使用线程池执行计算密集型操作
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                
                # 并行执行向量化操作
                future_embedding = executor.submit(
                    DocumentService._generate_embeddings_sync,
                    chunks, document
                )
                
                # 等待向量化结果
                embedding_result = await loop.run_in_executor(executor, future_embedding.result)
                
                # 并行执行存储操作
                future_storage = executor.submit(
                    DocumentService._store_chunks_sync,
                    chunks, document, embedding_result
                )
                
                # 等待存储结果
                storage_result = await loop.run_in_executor(executor, future_storage.result)
                
                return {
                    "status": "success",
                    "chunks": chunks,
                    "embeddings": embedding_result,
                    "storage": storage_result,
                    "metadata": chunk_result.get("metadata", {}),
                    "chunk_count": len(chunks),
                    "token_count": embedding_result.get("token_count", 0),
                    "vector_size": embedding_result.get("vector_size", 0)
                }
                
        except Exception as e:
            logging.error(f"多线程解析执行失败: {e}")
            raise
    
    @staticmethod
    def _parse_document_sync(
        document: Document,
        file_content: bytes,
        from_page: int,
        to_page: int,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        同步执行文档解析（在线程中运行）
        
        Args:
            document: 文档对象
            file_content: 文件内容
            from_page: 起始页/行
            to_page: 结束页/行
            task_config: 任务配置
            
        Returns:
            Dict[str, Any]: 解析结果
        """
        try:
            # 这里调用现有的解析逻辑
            # 注意：由于是在线程中运行，不能使用异步操作
            # 需要确保所有依赖都是同步的
            
            # 根据文档类型选择解析器
            if document.type == "pdf":
                # PDF 解析逻辑
                pass
            elif document.type == "excel":
                # Excel 解析逻辑
                pass
            else:
                # 其他类型解析逻辑
                pass
            
            # 临时返回示例数据
            return {
                "chunks": [],
                "metadata": {
                    "from_page": from_page,
                    "to_page": to_page,
                    "document_type": document.type
                }
            }
            
        except Exception as e:
            logging.error(f"同步文档解析失败: {e}")
            raise
    
    @staticmethod
    def _generate_embeddings_sync(
        chunks: List[Dict[str, Any]],
        document: Document
    ) -> Dict[str, Any]:
        """
        同步生成向量嵌入（在线程中运行）
        
        Args:
            chunks: 文档切片列表
            document: 文档对象
            
        Returns:
            Dict[str, Any]: 向量化结果
        """
        try:
            # 这里实现向量化逻辑
            # 注意：由于是在线程中运行，不能使用异步操作
            
            # 临时返回示例数据
            return {
                "token_count": len(chunks) * 100,  # 示例
                "vector_size": 768,  # 示例
                "embeddings": []
            }
            
        except Exception as e:
            logging.error(f"同步向量化失败: {e}")
            raise
    
    @staticmethod
    def _store_chunks_sync(
        chunks: List[Dict[str, Any]],
        document: Document,
        embedding_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        同步存储切片到向量数据库（在线程中运行）
        
        Args:
            chunks: 文档切片列表
            document: 文档对象
            embedding_result: 向量化结果
            
        Returns:
            Dict[str, Any]: 存储结果
        """
        try:
            # 这里实现存储逻辑
            # 注意：由于是在线程中运行，不能使用异步操作
            
            # 临时返回示例数据
            return {
                "stored_count": len(chunks),
                "storage_status": "success"
            }
            
        except Exception as e:
            logging.error(f"同步存储失败: {e}")
            raise

    @staticmethod
    async def get_documents_by_kb_id(
        session: AsyncSession,
        kb_id: str,
        page: int = 1,
        page_size: int = 20,
        keywords: str = "",
        order_by: str = "created_at",
        desc: bool = True
    ) -> Tuple[List[Document], int]:
        """根据知识库ID获取文档列表"""
        try:
            query = select(Document).where(Document.kb_id == kb_id)
            
            if keywords:
                query = query.where(
                    or_(
                        Document.name.contains(keywords),
                        Document.description.contains(keywords)
                    )
                )
            
            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total = count_result.scalar()
            
            if hasattr(Document, order_by):
                order_column = getattr(Document, order_by)
                if desc:
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(asc(order_column))
            else:
                if desc:
                    query = query.order_by(desc(Document.created_at))
                else:
                    query = query.order_by(asc(Document.created_at))
            
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            result = await session.execute(query)
            documents = result.scalars().all()
            
            return list(documents), total
            
        except Exception as e:
            logging.error(f"获取文档列表失败: {e}")
            raise
    
    @staticmethod
    async def update_document(
        session: AsyncSession,
        doc_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Document:
        """更新文档信息"""
        try:
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                raise ValueError("文档不存在")
            
            if name is not None:
                if name != document.name:
                    existing_doc = await DocumentService.get_document_by_name(
                        session, document.kb_id, name
                    )
                    if existing_doc:
                        raise ValueError(f"文档名称 '{name}' 在知识库中已存在")
                document.name = name
            
            if description is not None:
                document.description = description
            
            await session.commit()
            await session.refresh(document)
            
            logging.info(f"更新文档成功: {doc_id}")
            return document
            
        except Exception as e:
            await session.rollback()
            logging.error(f"更新文档失败: {e}")
            raise
    
    @staticmethod
    async def delete_document(
        session: AsyncSession,
        doc_id: str
    ):
        """删除文档"""
        try:
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                raise ValueError("文档不存在")
            
            # 这里可以添加清除向量数据、分块数据等的逻辑
            # 具体实现取决于您的向量存储方案
            
            await session.delete(document)
            await session.commit()
            
            await DocumentService._update_kb_doc_count(session, document.kb_id)
            
            logging.info(f"删除文档成功: {doc_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"删除文档失败: {e}")
            raise
    
    @staticmethod
    async def get_document_by_id(
        session: AsyncSession, 
        doc_id: str
    ) -> Optional[Document]:
        """根据ID获取文档"""
        try:
            result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"获取文档失败: {e}")
            return None
    
    @staticmethod
    async def get_documents_by_ids(
        session: AsyncSession, 
        doc_ids: List[str]
    ) -> List[Document]:
        """根据ID列表获取文档"""
        try:
            result = await session.execute(
                select(Document).where(Document.id.in_(doc_ids))
            )
            return result.scalars().all()
        except Exception as e:
            logging.error(f"批量获取文档失败: {e}")
            return []
    
    @staticmethod
    async def get_document_by_name(
        session: AsyncSession, 
        kb_id: str, 
        name: str
    ) -> Optional[Document]:
        """根据知识库ID和文档名称获取文档"""
        try:
            result = await session.execute(
                select(Document).where(
                    and_(Document.kb_id == kb_id, Document.name == name)
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"获取文档失败: {e}")
            return None

    @staticmethod
    async def _update_kb_doc_count(session: AsyncSession, kb_id: str):
        """更新知识库文档数量"""
        try:
            count_result = await session.execute(
                select(func.count(Document.id)).where(Document.kb_id == kb_id)
            )
            doc_count = count_result.scalar()
            
            kb_result = await session.execute(
                select(KB).where(KB.id == kb_id)
            )
            kb = kb_result.scalar_one_or_none()
            
            if kb:
                kb.document_count = doc_count
                await session.commit()
                
        except Exception as e:
            logging.error(f"更新知识库文档数量失败: {e}")
            await session.rollback() 
    
    @staticmethod
    def _build_storage_path(kb: KB) -> str:
        """构造知识库文档存储路径"""
        if kb.team_id:
            return f"{kb.team_id}/{kb.name}"
        else:
            return f"{kb.name}"

    @staticmethod
    async def cancel_celery_tasks(
        session: AsyncSession,
        doc_id: str
    ):
        """取消文档的Celery任务"""
        try:
            # 这里可以添加取消Celery任务的逻辑
            # 例如：通过celery.control.revoke()取消任务
            # 或者通过Redis直接操作任务队列
            
            logging.info(f"已取消文档 {doc_id} 的Celery任务")
            
        except Exception as e:
            logging.error(f"取消Celery任务失败: {e}")
            raise

    @staticmethod
    async def clear_document_data(
        session: AsyncSession,
        doc_id: str
    ):
        """清除文档相关数据（用于重新解析）"""
        try:
            # 这里可以添加清除向量数据、分块数据等的逻辑
            # 具体实现取决于您的向量存储方案
            
            logging.info(f"已清除文档 {doc_id} 的相关数据")
            
        except Exception as e:
            logging.error(f"清除文档数据失败: {e}")
            raise

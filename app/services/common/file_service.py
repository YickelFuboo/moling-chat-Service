import pdfplumber
from io import BytesIO
import shutil
import subprocess
import tempfile
import os
import asyncio
from datetime import datetime
from typing import Optional, BinaryIO, List, Tuple, Dict, Any
import logging
import uuid
import io
import re
from pathlib import Path
from PIL import Image
from enum import StrEnum
from app.config.settings import APP_NAME
from app.infrastructure.storage import STORAGE_CONN
from app.constants.common import DocumentConstants, KBConstants

class FileConfig:
    """文件配置类"""
    
    def __init__(
        self,
        allowed_extensions: Optional[list] = None,
        max_size_mb: Optional[int] = None,
        bucket: str = "",
        content_type_prefix: str = "",
        process_image: bool = False,
        max_dimensions: Optional[tuple] = None
    ):
        self.allowed_extensions = allowed_extensions
        self.max_size_mb = max_size_mb
        self.bucket = bucket
        self.content_type_prefix = content_type_prefix
        self.process_image = process_image
        self.max_dimensions = max_dimensions

class FileMetadata:
    """文件元数据类"""
    
    def __init__(
        self,
        original_filename: str,
        file_size: int,
        content_type: str,
        upload_time: Optional[str] = None,
        **kwargs
    ):
        self.original_filename = original_filename
        self.file_size = file_size
        self.content_type = content_type
        self.upload_time = upload_time
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "upload_time": self.upload_time,
            **self.extra
        }

class FileUsage(StrEnum):
    """文件类型枚举"""
    KB_IMAGE = "kb_image"                    # 知识库图像
    DOCUMENT = "document"                    # 文档文件
    DOCUMENT_THUMBNAIL = "document_thumbnail"  # 文档缩略图
    CHUNK_IMAGE = "chunk_image"              # 切片图像
    OTHER = "other"                          # 其他文件

FILE_USAGE_CONFIG =  {
    FileUsage.KB_IMAGE.value: FileConfig(
        allowed_extensions=KBConstants.SUPPORTED_IMAGE_TYPES,
        max_size_mb=KBConstants.MAX_IMAGE_SIZE,
        bucket=f"{APP_NAME.replace('_', '-')}-kb-images",
        content_type_prefix="image/",
        process_image=True,
        max_dimensions=(300, 300)
    ),
    FileUsage.DOCUMENT.value: FileConfig(
        allowed_extensions=None,
        max_size_mb=None,
        bucket=f"{APP_NAME.replace('_', '-')}-documents",
        content_type_prefix="application/",
        process_image=False
    ),
    FileUsage.DOCUMENT_THUMBNAIL.value: FileConfig(
        allowed_extensions=None,
        max_size_mb=None,
        bucket=f"{APP_NAME.replace('_', '-')}-documents-thumbnails",
        content_type_prefix="image/",
        process_image=True,
        max_dimensions=(300, 300)
    ),
    FileUsage.CHUNK_IMAGE.value: FileConfig(
        allowed_extensions=None,
        max_size_mb=None,
        bucket=f"{APP_NAME.replace('_', '-')}-chunks-images",
        content_type_prefix="image/",
        process_image=False
    )
}

class FileService:
    """文件存储服务 - 提供文件上传、下载、处理等业务功能"""

    @staticmethod
    async def upload_file(
        file_data: BinaryIO,
        file_name: str,
        file_usage: FileUsage,
        **kwargs
    ) -> str:
        """统一文件上传方法"""
        try:
            if file_usage.value not in FILE_USAGE_CONFIG:
                raise ValueError(f"不支持的文件类型: {file_usage.value}")
            
            config = FILE_USAGE_CONFIG[file_usage.value]
            
            # 验证文件扩展名（如果配置了允许的扩展名）
            if config.allowed_extensions:
                file_ext = os.path.splitext(file_name)[1].lower()
                if file_ext not in config.allowed_extensions:
                    raise ValueError(f"不支持的文件类型: {file_ext}")
            
            # 验证文件大小
            file_data.seek(0, 2)
            file_size = file_data.tell()
            file_data.seek(0)
            
            # 根据文件类型验证大小限制
            if config.max_size_mb and file_size > config.max_size_mb * 1024 * 1024:
                raise ValueError(f"文件大小超过限制: {file_size} bytes")
            
            # 处理图片文件
            if config.process_image and FileService._is_image_file(os.path.splitext(file_name)[1].lower()):
                file_data = FileService._process_image(file_data, config.max_dimensions)
            
            # 准备元数据
            file_ext = os.path.splitext(file_name)[1].lower()
            metadata = {
                "original_filename": file_name,
                "upload_time": datetime.utcnow().isoformat(),
                "file_size": file_size,
                "content_type": f"{config.content_type_prefix}{file_ext[1:]}",
                **kwargs
            }
            
            # 保存文件
            file_id = file_name + str(uuid.uuid4()).replace("-", "")
            await STORAGE_CONN.put(
                file_index = file_id,
                file_data = file_data,
                bucket_name = config.bucket,
                content_type = metadata["content_type"],
                metadata = metadata
            )
            
            logging.info(f"文件上传成功: {file_name}")
            return file_id
            
        except Exception as e:
            logging.error(f"文件上传失败: {e}")
            raise

    @staticmethod
    async def get_file_content(
        file_id: str,
        file_usage: FileUsage
    ) -> bytes:
        """统一获取文件内容方法"""  
        try:
            if file_usage.value not in FILE_USAGE_CONFIG:
                raise ValueError(f"不支持的文件类型: {file_usage.value}")
            
            config = FILE_USAGE_CONFIG[file_usage.value]
            
            # 在线程池中执行同步的storage操作
            file_data = await STORAGE_CONN.get(
                file_index = file_id, 
                bucket_name = config.bucket
            )

            if not file_data:
                raise ValueError(f"文件不存在或下载失败: {file_id}")
            
            return file_data.read()
            
        except Exception as e:
            logging.error(f"获取文件内容失败: {e}")
            raise

    @staticmethod
    async def delete_file(
        file_id: str,
        file_usage: FileUsage
    ) -> bool:
        """统一删除文件方法"""
        try:
            if file_usage.value not in FILE_USAGE_CONFIG:
                raise ValueError(f"不支持的文件类型: {file_usage.value}")
            
            config = FILE_USAGE_CONFIG[file_usage.value]
            
            # 删除文件操作
            success = await STORAGE_CONN.delete(
                file_index = file_id,
                bucket_name = config.bucket
            )

            if success:
                logging.info(f"文件删除成功: {file_id}")
            else:
                logging.warning(f"文件删除失败: {file_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"删除文件失败: {e}")
            return False

    @staticmethod
    async def get_file_url(
        file_id: str,
        file_usage: FileUsage,
        expires_in: Optional[int] = None
    ) -> Optional[str]:
        """获取文件URL（仅支持头像类型）"""
        try:
            if file_usage.value not in FILE_USAGE_CONFIG:
                raise ValueError(f"不支持的文件类型: {file_usage.value}")
                
            config = FILE_USAGE_CONFIG[file_usage.value]

            url = await STORAGE_CONN.get_url(
                file_id=file_id,
                bucket_name=config.bucket,
                expires_in=expires_in
            )
            return url
            
        except Exception as e:
            logging.error(f"获取文件URL失败: {e}")
            return None

    @staticmethod
    async def get_file_metadata(
        file_id: str,
        file_usage: FileUsage
    ) -> Optional[dict]:
        """获取文件元数据"""
        try:
            if file_usage.value not in FILE_USAGE_CONFIG:
                raise ValueError(f"不支持的文件类型: {file_usage.value}")
                
            config = FILE_USAGE_CONFIG[file_usage.value]
            
            # 在线程池中执行元数据获取操作
            metadata = await STORAGE_CONN.get_file_metadata(
                file_id=file_id,
                bucket_name=config.bucket
            )
            
            return metadata
            
        except Exception as e:
            logging.error(f"获取文件元数据失败: {e}")
            return None

    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """验证文件名"""
        if not filename or filename.strip() == "":
            return False, "文件名不能为空"
        
        if len(filename.encode("utf-8")) > DocumentConstants.FILE_NAME_LEN_LIMIT:
            return False, f"文件名长度超过限制 ({DocumentConstants.FILE_NAME_LEN_LIMIT} 字节)"
        
        invalid_chars = r'[<>:"/\\|?*]'
        if re.search(invalid_chars, filename):
            return False, "文件名包含非法字符"
        
        return True, ""

    @staticmethod
    def get_file_info(content: bytes, filename: str) -> dict:
        """获取文件信息"""
        return {
            "size": len(content),
            "extension": Path(filename).suffix.lower().lstrip("."),
            "filename": filename
        }

    @staticmethod
    def _is_image_file(file_ext: str) -> bool:
        """判断文件是否为图片类型"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg'}
        return file_ext.lower() in image_extensions

    @staticmethod
    def _process_image(file_data: BinaryIO, max_dimensions: tuple) -> BinaryIO:
        """处理图片文件"""
        try:
            image = Image.open(file_data)
            image.thumbnail(max_dimensions, Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            image.save(output, format=image.format or 'JPEG', quality=85)
            output.seek(0)
            
            return output
        except Exception as e:
            logging.error(f"图片处理失败: {e}")
            return file_data
    
    @staticmethod
    def fix_broken_pdf(content: bytes) -> bytes:
        """修复可能损坏的PDF文件"""
        def try_open(blob):
            try:
                with pdfplumber.open(BytesIO(blob)) as pdf:
                    if pdf.pages:
                        return True
            except Exception:
                return False
            return False

        # 尝试直接打开PDF
        if try_open(content):
            return content

        # 如果无法打开，尝试使用Ghostscript修复
        try:
            repaired = FileService._repair_pdf_with_ghostscript(content)
            if try_open(repaired):
                return repaired
        except Exception as e:
            logging.warning(f"PDF修复失败: {e}")

        # 如果修复失败，返回原内容
        return content
    
    @staticmethod
    def _repair_pdf_with_ghostscript(input_bytes: bytes) -> bytes:
        """使用Ghostscript修复PDF"""
        
        if shutil.which("gs") is None:
            return input_bytes

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_in, tempfile.NamedTemporaryFile(suffix=".pdf") as temp_out:
            temp_in.write(input_bytes)
            temp_in.flush()

            cmd = [
                "gs",
                "-o",
                temp_out.name,
                "-sDEVICE=pdfwrite",
                "-dPDFSETTINGS=/prepress",
                temp_in.name,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    return input_bytes
            except Exception:
                return input_bytes

            temp_out.seek(0)
            repaired_bytes = temp_out.read()

        return repaired_bytes
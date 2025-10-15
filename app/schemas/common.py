from pydantic import BaseModel
from typing import Optional

class SuccessResponse(BaseModel):
    """通用成功响应模型"""
    message: str = "操作成功"
    details: Optional[str] = None 
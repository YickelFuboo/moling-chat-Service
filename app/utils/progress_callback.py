import logging
from typing import Optional


class ProgressCallback:
    """文档解析进度通知类"""
    
    def progress_callback(self, progress: Optional[float] = None, msg: Optional[str] = None) -> None:
        """
        进度回调方法
        
        Args:
            progress: 进度值 (0.0-1.0)
            msg: 进度消息
        """
        if progress is not None and msg is not None:
            logging.info(f"文档解析进度: {progress*100:.1f}% - {msg}")
        elif msg is not None:
            logging.info(f"文档解析: {msg}")
        else:
            logging.info("文档解析进度更新")

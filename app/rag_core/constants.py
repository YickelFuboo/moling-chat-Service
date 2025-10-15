import os
import asyncio
import logging
from app.config.settings import settings

#CHAT_LIMITER = trio.CapacityLimiter(int(settings.max_concurrent_chats))
#CHUNK_LIMITER = trio.CapacityLimiter(int(settings.max_concurrent_chunk_builders))
#MINIO_LIMITER = trio.CapacityLimiter(int(settings.max_concurrent_minio))
#KG_LIMITER = trio.CapacityLimiter(2)
# 使用asyncio的Semaphore替代trio的CapacityLimiter
CHAT_LIMITER = asyncio.Semaphore(int(settings.max_concurrent_chats))
CHUNK_LIMITER = asyncio.Semaphore(int(settings.max_concurrent_chunk_builders))
MINIO_LIMITER = asyncio.Semaphore(int(settings.max_concurrent_minio))
KG_LIMITER = asyncio.Semaphore(2)

# 下载模型缓存目录，rag当前目录下的/res
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "res")

# RAG模式胚子
RAG_LIGHTEN_MODE = settings.lighten_mode

# 如下内容来资源原RAG项目的rag/settings.py
DOC_MAXIMUM_SIZE = int(os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024))
DOC_BULK_SIZE = int(os.environ.get("DOC_BULK_SIZE", 4))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 16))

TAG_FLD = "tag_feas"
PAGERANK_FLD = "pagerank_fea"


PARALLEL_DEVICES = 0
try:
    import torch.cuda
    PARALLEL_DEVICES = torch.cuda.device_count()
    logging.info(f"found {PARALLEL_DEVICES} gpus")
except Exception:
    logging.info("can't import package 'torch'")

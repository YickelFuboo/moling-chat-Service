from .utils import ParserType
from .rag.app import naive, paper, presentation, manual, qa, table, book, resume, picture, one, audio, email, tag, laws


# 如下内容来资源原项目的rag/srv/task_executor.py
CHUNK_FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,  # 法律文档使用laws解析器
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: naive,  # 知识图谱使用naive解析器
    ParserType.TAG.value: tag
}
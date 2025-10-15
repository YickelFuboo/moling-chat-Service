# 1. RAG App 模块 - 文档解析切片器

## 1.1 概述

`rag/app` 模块是RAGFlow系统的文档解析切片器集合，负责将各种格式的文档解析并切片成适合检索的小块。

## 1.2 核心功能

- **文档解析**：支持多种文档格式的解析
- **智能切片**：根据文档结构和内容进行智能分块
- **统一接口**：所有解析器都提供 `chunk()` 函数作为对外入口

## 1.3 解析器列表

| 解析器 | 支持格式 | 主要用途 |
|--------|----------|----------|
| `naive.py` | PDF, Word, Excel, HTML, Markdown | 通用文档处理 |
| `qa.py` | Excel, PDF, Word | 问答对提取 |
| `table.py` | Excel, Word表格 | 表格数据处理 |
| `paper.py` | PDF, Word | 学术论文处理 |
| `manual.py` | PDF, Word | 手册文档处理 |
| `book.py` | PDF, Word | 书籍文档处理 |
| `resume.py` | PDF, Word | 简历文档处理 |
| `presentation.py` | PowerPoint, PDF | 演示文稿处理 |
| `picture.py` | 图片文件 | 图片内容描述 |
| `audio.py` | 音频文件 | 音频转文字 |
| `email.py` | 邮件文件 | 邮件内容提取 |
| `laws.py` | PDF, Word | 法律文档处理 |
| `tag.py` | 各种文档 | 标签生成处理 |
| `one.py` | 各种文档 | 统一处理接口 |

## 1.4 使用方法

### 1.4.1 直接调用

```python
from rag.app import naive, qa, manual

# 使用通用解析器
result = naive.chunk("document.pdf")

# 使用问答解析器
result = qa.chunk("qa_document.xlsx")

# 使用手册解析器
result = manual.chunk("user_manual.docx")
```

### 1.4.2 通过任务执行器

```python
# 在task_executor.py中，系统会根据parser_id自动选择对应的解析器
FACTORY = {
    "naive": naive,
    "qa": qa,
    "manual": manual,
    # ... 其他解析器
}

chunker = FACTORY[task["parser_id"]]
result = chunker.chunk(...)
```

## 1.5 统一接口

所有解析器都提供相同的 `chunk()` 函数签名：

```python
def chunk(filename, binary=None, from_page=0, to_page=100000, 
          lang="Chinese", callback=None, **kwargs):
    """
    文档切片函数
    
    参数:
        filename: 文件名
        binary: 完整的待分片文档二进制内容（可选，从存储中读取的原始文件数据）
        from_page: 起始页码
        to_page: 结束页码
        lang: 语言（中文/英文）
        callback: 进度回调函数
        **kwargs: 其他参数
    
    返回:
        解析后的文档分块列表
    """
    pass
```

### 1.5.1 参数详细说明

| 参数名 | 数据类型 | 必需 | 默认值 | 参数用途 | 涉及解析器 |
|--------|----------|------|--------|----------|------------|
| **filename** | `string` | ✅ | - | 文件名，用于文档标识和标题生成 | 所有解析器 |
| **binary** | `bytes` | ❌ | `None` | 完整的待分片文档二进制内容，从存储中读取的原始文件数据 | 所有解析器 |
| **from_page** | `int` | ❌ | `0` | 起始页码，用于分页处理 | naive, manual, book, paper, laws, presentation |
| **to_page** | `int` | ❌ | `100000` | 结束页码，用于分页处理 | naive, manual, book, paper, laws, presentation |
| **lang** | `string` | ❌ | `"Chinese"` | 语言设置（中文/英文），影响分词和LLM调用 | 所有解析器 |
| **callback** | `function` | ❌ | `None` | 进度回调函数，用于报告处理进度 | 所有解析器 |
| **tenant_id** | `string` | ❌ | - | 租户ID，用于调用LLM模型 | audio, picture, naive, presentation |
| **kwargs** | `dict` | ❌ | `{}` | 其他参数，包含以下关键key： | 见下方详细说明 |

### 1.5.2 kwargs关键参数说明

| 参数key | 数据类型 | 默认值 | 参数用途 | 使用解析器 |
|---------|----------|--------|----------|------------|
| **parser_config** | `dict` | `{"chunk_token_num": 128, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"}` | 解析器配置字典，包含分块和布局识别设置 | naive, email |
| **layout_recognize** | `string` | `"DeepDOC"` | 布局识别方式（"DeepDOC"或"Plain Text"），优先级高于parser_config中的设置 | naive, manual, book, paper, laws, presentation |
| **section_only** | `bool` | `False` | 是否只返回section，用于naive解析器 | naive |
| **html4excel** | `bool` | `False` | 是否以HTML格式处理Excel，用于naive解析器 | naive |
| **tenant_id** | `string` | - | 租户ID，用于调用LLM模型（当作为kwargs传入时） | naive, presentation |
| **kb_id** | `string` | - | 知识库ID，用于更新知识库的parser_config | table |

**注意**: 
- 个人观点：实际上模块上传递tenant_id和kb_id不合适，把本模块内容业务耦合了。建议
   （1）tenant_id对象可替换为直接传入LLM实例
   （2）table解析器的kb-id没想好如何替换

## 1.6 返回格式

所有解析器返回的分块列表格式如下：

```python
[
    {
        "content": "这是切片后的片段内容",
        "content_with_weight": "这是切片后的片段内容",
        "docnm_kwd": "user_manual.pdf",
        "title_tks": ["用户", "手册", "pdf"],
        "title_sm_tks": ["用户", "手册", "pdf"],
        "doc_type_kwd": "image",
        "image": "PIL.Image对象",
        "page_num_int": [1, 2],
        "position_int": [(1, 100, 200, 50, 100), (2, 150, 250, 75, 125)],
        "top_int": [50, 75],
        "content_ltks": ["内容", "分词", "结果"],
        "content_sm_ltks": ["内容", "分词", "细粒度"]
    },
    # ... 更多分块
]
```

## 1.7 完整字段说明表

| 字段名 | 数据类型 | 格式示例 | 说明 | 数据来源 | 有效解析器 |
|--------|----------|----------|------|----------|------------|
| **content** | `string` | `"这是切片后的片段内容"` | 分块的主要内容文本，经过智能切片处理 | chunk函数内部生成：文档解析和切片后的文本内容 | 所有解析器 |
| **content_with_weight** | `string` | `"这是切片后的片段内容"` | 包含权重信息的内容文本，通常与content相同 | chunk函数内部生成：为权重计算预留的内容副本 | 所有解析器 |
| **docnm_kwd** | `string` | `"user_manual.pdf"` | 文档文件名（不含路径） | 从chunk函数filename参数获取 | 所有解析器 |
| **title_tks** | `list[string]` | `["用户", "手册", "pdf"]` | 文档名称的分词结果，使用rag_tokenizer.tokenize() | chunk函数内部生成：基于filename的分词结果 | 所有解析器 |
| **title_sm_tks** | `list[string]` | `["用户", "手册", "pdf"]` | 文档名称的细粒度分词结果，用于更精确的语义匹配 | chunk函数内部生成：基于title_tks的细粒度分词 | 所有解析器 |
| **doc_type_kwd** | `string` | `"image"`, `"pdf"`, `"docx"` | 文档类型标识，用于分类和检索策略选择 | chunk函数内部生成：根据文档内容或类型判断 | 所有解析器 |
| **image** | `PIL.Image` | `PIL.Image对象` | 图片对象，仅图片解析器返回 | chunk函数内部生成：根据文档内容判断是否包含图片 | picture, presentation, manual(docx), naive(markdown), table |
| **page_num_int** | `list[int]` | `[1, 2, 3]` | 页码列表，表示分块内容来自哪些页面 | chunk函数内部生成：根据文档页面结构生成 | presentation, manual, book, paper, laws, naive, table |
| **position_int** | `list[tuple]` | `[(1, 100, 200, 50, 100), (2, 150, 250, 75, 125)]` | 位置信息，包含页面、坐标等详细信息 | chunk函数内部生成：根据文档布局信息生成 | presentation, manual, book, paper, laws, naive, table |
| **top_int** | `list[int]` | `[50, 75]` | 顶部位置列表，用于定位分块在页面中的位置 | chunk函数内部生成：根据文档位置信息生成 | presentation, manual, book, paper, laws, naive, table |
| **content_ltks** | `list[string]` | `["内容", "分词", "结果"]` | 内容的分词结果，用于文本检索和匹配 | chunk函数内部生成：使用rag_tokenizer对content进行分词 | 所有解析器 |
| **content_sm_ltks** | `list[string]` | `["内容", "分词", "细粒度"]` | 内容的细粒度分词结果，用于精确语义匹配 | chunk函数内部生成：使用rag_tokenizer对content进行细粒度分词 | 所有解析器 |
| **authors_tks** | `list[string]` | `["作者", "姓名"]` | 作者名称的分词结果 | chunk函数内部生成：从论文元数据中提取作者信息 | paper |
| **authors_sm_tks** | `list[string]` | `["作者", "姓名"]` | 作者名称的细粒度分词结果 | chunk函数内部生成：基于authors_tks的细粒度分词 | paper |
| **important_kwd** | `list[string]` | `["abstract", "总结"]` | 重要关键词列表 | chunk函数内部生成：根据文档类型和内容提取关键词 | paper, resume |
| **important_tks** | `string` | `"abstract 总结"` | 重要关键词的文本形式 | chunk函数内部生成：将important_kwd转换为文本 | paper, resume |
| **name_kwd** | `string` | `"张三"` | 姓名关键词 | chunk函数内部生成：从简历内容中提取姓名信息 | resume |
| **gender_kwd** | `string` | `"男"` | 性别关键词 | chunk函数内部生成：从简历内容中提取性别信息 | resume |
| **position_name_tks** | `list[string]` | `["软件", "工程师"]` | 职位名称分词 | chunk函数内部生成：从简历内容中提取职位信息并分词 | resume |
| **age_int** | `int` | `28` | 年龄数值 | chunk函数内部生成：从简历内容中提取年龄信息 | resume |

## 1.8 字段用途说明

### 1.8.1 title_sm_tks 的作用
- **细粒度分词**：比 `title_tks` 更精确的分词结果
- **语义匹配**：用于更精确的文档标题语义搜索
- **检索优化**：提高文档检索的准确性和相关性

### 1.8.2 doc_type_kwd 的作用
- **文档分类**：标识文档类型（PDF、Word、图片等）
- **解析策略**：系统根据类型选择不同的解析和处理策略
- **存储优化**：不同类型的文档可能使用不同的存储方式
- **检索过滤**：用户可以根据文档类型进行过滤搜索



## 1.9 特点

- **模块化设计**：每个解析器独立，职责明确
- **格式支持广泛**：支持PDF、Word、Excel、图片、音频等多种格式
- **智能切片**：根据文档类型和内容结构进行优化切片
- **统一接口**：所有解析器使用相同的调用方式
- **可扩展性**：易于添加新的文档格式支持

## 1.10 注意事项

- 这些是**静态文件解析器**，不涉及动态内容
- 主要处理本地或存储的文档文件
- 解析结果会保存到向量数据库中用于后续检索
- 不同解析器针对不同文档类型进行了优化
- 字节字段主要用于内存中的文档处理，避免频繁的磁盘I/O操作

## 1.11 task_executor后处理添加的字段

在rag/app模块的chunk函数返回基础字段后，task_executor会进一步处理并添加以下字段：

| 字段名 | 数据类型 | 格式示例 | 说明 | 数据来源 | 添加时机 |
|--------|----------|----------|------|----------|----------|
| **kb_id** | `string` | `"kb_12345"` | 所属知识库ID，用于数据隔离和组织 | 从task["kb_id"]参数获取 | 构建基础doc对象时 |
| **doc_id** | `string` | `"doc_67890"` | 源文档ID，用于追踪文档来源 | 从task["doc_id"]参数获取 | 构建基础doc对象时 |
| **chunk_id** | `string` | `"chunk_fghij"` | 分块的唯一标识，用于去重和引用 | 通过xxhash对content_with_weight和doc_id生成 | 处理每个chunk时 |
| **create_time** | `string` | `"2024-01-15 10:30:45"` | 分块创建时间，格式为字符串 | 生成分块时记录的时间 | 处理每个chunk时 |
| **create_timestamp_flt** | `float` | `1705305045.123` | 分块创建时间戳，浮点数格式 | 生成分块时记录的时间戳 | 处理每个chunk时 |
| **img_id** | `string` | `"kb_12345-img_abcde"` | 图片在存储中的唯一标识 | 上传image对象到存储后生成的ID | 处理包含图片的chunk时 |
| **important_kwd** | `list[string]` | `["关键词1", "关键词2"]` | 自动提取的重要关键词列表 | 通过LLM模型对content_with_weight进行关键词提取 | 配置auto_keywords时 |
| **important_tks** | `list[string]` | `["关键词1", "关键词2"]` | 重要关键词的分词结果 | 对important_kwd进行分词处理 | 配置auto_keywords时 |
| **pagerank** | `int` | `100` | 页面排名权重值 | 从task["pagerank"]参数获取 | 构建基础doc对象时 | 
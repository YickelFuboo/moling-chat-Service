# RAG/NLP 模块技术文档

## 概述

`rag/nlp` 模块是 RAGFlow 项目中的自然语言处理核心模块，为整个检索增强生成系统提供强大的 NLP 基础能力。该模块主要负责文本预处理、分词、搜索、查询处理、术语权重计算、同义词处理等核心功能。

## 模块架构

```
rag/nlp/
├── __init__.py           # 模块入口，提供主要工具函数
├── rag_tokenizer.py      # RAG分词器，支持中英文混合分词
├── search.py             # 搜索功能，支持向量和全文搜索
├── query.py              # 查询处理，优化用户查询
├── term_weight.py        # 术语权重计算
├── synonym.py            # 同义词处理和扩展
└── surname.py            # 姓氏处理
``` 

## 核心功能模块

### 1. 文本分词与处理 (rag_tokenizer.py)

#### RagTokenizer 类
提供中英文混合文本的分词功能，支持多种文本预处理操作。

**主要功能：**
- 中文繁体转简体
- 全角转半角转换
- 英文词干提取和词形还原
- 自定义词典加载和管理

**核心方法：**
```python
# 基础分词
tokens = tokenizer.tokenize(text)

# 细粒度分词
fine_tokens = tokenizer.fine_grained_tokenize(tokens)

# 繁体转简体
simplified = tokenizer.tradi2simp(text)

# 全角转半角
normalized = tokenizer.strQ2B(text)
```

**内部结构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                        RagTokenizer                             │
├─────────────────────────────────────────────────────────────────┤
│ • __init__() - 初始化分词器，加载词典                          │
│   ├── loadDict_() - 加载词典文件                               │
│   └── loadUserDict() - 加载用户自定义词典                      │
│ • tokenize() - 基础分词                                        │
│   ├── 被 fine_grained_tokenize() 调用                         │
│   └── 被外部模块广泛调用                                       │
│ • fine_grained_tokenize() - 细粒度分词                         │
│   └── 被外部模块调用                                           │
│ • tradi2simp() - 繁体转简体                                    │
│   └── 被 query.py 调用                                         │
│ • strQ2B() - 全角转半角                                        │
│   └── 被 query.py 调用                                         │
│ • tag() - 词性标注                                             │
│   └── 被 term_weight.py 调用                                   │
│ • freq() - 词频统计                                            │
│   └── 被 term_weight.py 调用                                   │
└─────────────────────────────────────────────────────────────────┘
```

**被调用关系：**

| 调用模块 | 调用接口 | 使用场景 | 调用频率 |
|---------|---------|----------|----------|
| rag/app/*.py (所有应用模块) | tokenize(), fine_grained_tokenize() | 文档标题和内容分词 | 高频 |
| rag/nlp/term_weight.py | tokenize(), fine_grained_tokenize(), tag(), freq() | 词汇权重计算 | 高频 |
| rag/nlp/query.py | tokenize(), fine_grained_tokenize(), tradi2simp(), strQ2B() | 查询预处理 | 高频 |
| rag/nlp/search.py | tokenize(), fine_grained_tokenize() | 搜索关键词分词 | 高频 |
| rag/svr/task_executor.py | tokenize() | 关键词和问题分词 | 高频 |
| rag/utils/*.py (存储连接器) | tokenize(), fine_grained_tokenize() | 查询分词处理 | 中频 |
| graphrag/*.py (图RAG模块) | tokenize() | 索引构建分词 | 中频 |
| deepdoc/*.py (文档解析) | tokenize() | 文档内容分词 | 中频 |
| api/*.py (API服务) | tokenize() | 接口分词处理 | 中频 |

**依赖关系：**
- datrie (字典树)
- hanziconv (中文繁简转换)
- nltk (英文词干提取和词形还原)

### 2. 搜索功能 (search.py)

#### Dealer 类
提供向量搜索和全文搜索的混合搜索能力，支持复杂的搜索场景。

**主要功能：**
- 向量相似度搜索
- 全文关键词搜索
- 混合搜索（向量+关键词）
- 搜索结果分页、高亮、聚合
- 多种过滤条件和排序方式

**核心方法：**
```python
# 执行搜索
result = dealer.search(
    req=search_request,
    idx_names=index_names,
    kb_ids=knowledge_base_ids,
    emb_mdl=embedding_model,
    highlight=True
)

# 获取向量表示
vector_expr = dealer.get_vector(text, embedding_model, topk=10)
```

**搜索方式控制：**

搜索方法的选择是通过参数自动控制的，无需手动指定：

1. **纯关键词搜索**：当 `emb_mdl=None` 时
   ```python
   # 只进行关键词搜索
   result = dealer.search(
       req={"question": "查询内容"},
       idx_names=["index_name"],
       kb_ids=["kb_id"],
       emb_mdl=None  # 不提供嵌入模型
   )
   ```

2. **混合搜索（默认）**：当 `emb_mdl` 提供时
   ```python
   # 自动进行向量+关键词混合搜索
   result = dealer.search(
       req={"question": "查询内容"},
       idx_names=["index_name"],
       kb_ids=["kb_id"],
       emb_mdl=embedding_model,  # 提供嵌入模型
       highlight=True
   )
   ```

3. **搜索参数控制**：
   ```python
   search_request = {
       "question": "用户查询",
       "similarity": 0.1,        # 向量相似度阈值
       "min_match": 0.3,         # 关键词匹配阈值
       "page": 1,                # 分页
       "size": 10,               # 每页大小
       "topk": 100               # 最大返回结果数
   }
   ```

**搜索策略说明：**
- **自动选择**：系统根据是否提供 `emb_mdl` 自动选择搜索方式
- **权重分配**：混合搜索中，关键词搜索权重为 0.05，向量搜索权重为 0.95
- **智能回退**：如果混合搜索无结果，会自动降低阈值重试
- **结果融合**：使用 `FusionExpr` 将不同搜索方式的结果进行加权融合

**内部结构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                            Dealer                               │
├─────────────────────────────────────────────────────────────────┤
│ • __init__() - 初始化搜索处理器                                 │
│   ├── 初始化 FulltextQueryer 实例                               │
│   └── 初始化数据存储连接                                        │
│ • search() - 执行搜索                                           │
│   ├── 被外部模块广泛调用                                        │
│   ├── 内部调用 get_filters() 获取过滤条件                       │
│   ├── 内部调用 get_vector() 获取向量表示                       │
│   └── 内部调用 query.question() 处理查询                        │
│ • get_vector() - 获取向量表示                                   │
│   └── 被 search() 调用                                          │
│ • get_filters() - 获取过滤条件                                   │
│   └── 被 search() 调用                                          │
│ • insert_citations() - 插入引用                                 │
│   └── 被外部模块调用                                            │
└─────────────────────────────────────────────────────────────────┘
```

**被调用关系：**

| 调用模块 | 调用接口 | 使用场景 | 调用频率 |
|---------|---------|----------|----------|
| rag/svr/task_executor.py | search() | 文档检索任务 | 高频 |
| graphrag/general/index.py | search() | 图RAG索引构建 | 中频 |
| graphrag/utils.py | search() | 图RAG工具搜索 | 中频 |
| api/apps/sdk/doc.py | search() | SDK文档搜索 | 高频 |
| api/apps/chunk_app.py | search() | 分块应用搜索 | 中频 |
| api/db/services/doc_service.py | search() | 文档服务搜索 | 中频 |

**依赖关系：**
- rag/nlp/rag_tokenizer.py → tokenize(), fine_grained_tokenize()
- rag/nlp/query.py → question()
- rag/utils/doc_store_conn.py → DocStoreConnection, MatchDenseExpr, FusionExpr

### 3. 查询处理 (query.py)

#### FulltextQueryer 类
处理用户查询的预处理和优化，提升搜索效果。

**主要功能：**
- 中英文混合查询处理
- 查询扩展和同义词处理
- 问题类型识别
- 关键词提取和优化

**核心方法：**
```python
# 处理问题查询
processed_query = queryer.question(text, table="qa")

# 处理关键词查询
keywords = queryer.keyword(text, table="kw")

# 添加中英文间空格
formatted_text = queryer.add_space_between_eng_zh(text)
```

**内部结构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                      FulltextQueryer                            │
├─────────────────────────────────────────────────────────────────┤
│ • __init__() - 初始化查询器                                     │
│   ├── 初始化 term_weight.Dealer 实例                            │
│   └── 初始化 synonym.Dealer 实例                                │
│ • question() - 处理问题查询                                     │
│   ├── 被 search.py 和 task_executor.py 调用                     │
│   ├── 内部调用 add_space_between_eng_zh() 处理中英文空格        │
│   ├── 内部调用 tradi2simp() 和 strQ2B() 进行文本预处理          │
│   ├── 内部调用 rmWWW() 移除疑问词                               │
│   ├── 内部调用 tokenize() 进行分词                              │
│   └── 内部调用 term_weight.weights() 计算权重                   │
│ • keyword() - 处理关键词查询                                    │
│   ├── 被 search.py 调用                                         │
│   ├── 内部调用 add_space_between_eng_zh() 处理中英文空格        │
│   ├── 内部调用 tradi2simp() 和 strQ2B() 进行文本预处理          │
│   └── 内部调用 tokenize() 进行分词                              │
│ • add_space_between_eng_zh() - 中英文空格处理                   │
│   └── 被 question() 和 keyword() 调用                           │
│ • rmWWW() - 移除疑问词                                          │
│   └── 被 question() 调用                                        │
│ • subSpecialChar() - 特殊字符处理                               │
│   └── 被外部模块调用                                            │
│ • isChinese() - 中文检测                                        │
│   └── 被外部模块调用                                            │
└─────────────────────────────────────────────────────────────────┘
```

**被调用关系：**

| 调用模块 | 调用接口 | 使用场景 | 调用频率 |
|---------|---------|----------|----------|
| rag/nlp/search.py | question(), keyword() | 搜索查询预处理 | 高频 |
| rag/svr/task_executor.py | question() | 任务查询处理 | 中频 |

**依赖关系：**
- rag/nlp/rag_tokenizer.py → tokenize(), fine_grained_tokenize(), tradi2simp(), strQ2B()
- rag/nlp/term_weight.py → weights()
- rag/nlp/synonym.py → lookup()

### 4. 术语权重 (term_weight.py)

#### Dealer 类
计算词汇的重要性和权重，为搜索排序提供依据。

**主要功能：**
- 词汇重要性评估
- 命名实体识别
- 词频统计
- 停用词过滤

**核心方法：**
```python
# 计算词汇权重
weights = weight_dealer.weights(tokens, preprocess=True)

# 文本预处理
processed = weight_dealer.pretoken(text, num=False, stpwd=True)
```

**内部结构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                           Dealer                                │
├─────────────────────────────────────────────────────────────────┤
│ • __init__() - 加载NER词典和词频词典                           │
│   ├── 加载 ner.json 文件                                        │
│   └── 加载 term.freq 文件                                       │
│ • weights() - 计算词汇权重                                     │
│   ├── 被 query.py 和 task_executor.py 调用                      │
│   ├── 内部调用 pretoken() 进行文本预处理                        │
│   ├── 内部调用 tokenize() 进行分词                              │
│   └── 内部调用 tag() 和 freq() 获取词性和词频                   │
│ • pretoken() - 文本预处理                                      │
│   └── 被 weights() 调用                                         │
│ • load_dict() - 加载词典                                       │
│   └── 被 __init__() 调用                                        │
└─────────────────────────────────────────────────────────────────┘
```

**被调用关系：**

| 调用模块 | 调用接口 | 使用场景 | 调用频率 |
|---------|---------|----------|----------|
| rag/nlp/query.py | weights() | 查询词汇权重计算 | 高频 |
| rag/svr/task_executor.py | weights() | 文档词汇权重计算 | 中频 |

**依赖关系：**
- rag/nlp/rag_tokenizer.py → tokenize(), fine_grained_tokenize(), tag(), freq()
- api/utils/file_utils.py → 文件路径处理

### 5. 同义词处理 (synonym.py)

#### Dealer 类
提供同义词查找和扩展功能，提升搜索召回率。

**主要功能：**
- 中英文同义词词典
- WordNet 英文同义词集成
- Redis 实时同义词更新
- 动态同义词加载

**核心方法：**
```python
# 查找同义词
synonyms = synonym_dealer.lookup(token, topn=8)

# 加载同义词词典
synonym_dealer.load()
```

**内部结构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                           Dealer                                │
├─────────────────────────────────────────────────────────────────┤
│ • __init__() - 加载同义词词典                                   │
│   ├── 加载 synonym.json 文件                                    │
│   ├── 初始化 Redis 连接（可选）                                 │
│   └── 调用 load() 加载词典                                      │
│ • lookup() - 查找同义词                                         │
│   ├── 被 query.py 调用                                          │
│   └── 内部调用 load() 检查词典更新                              │
│ • load() - 动态加载词典                                         │
│   ├── 被 __init__() 和 lookup() 调用                            │
│   └── 内部检查 Redis 更新                                       │
└─────────────────────────────────────────────────────────────────┘
```

**被调用关系：**

| 调用模块 | 调用接口 | 使用场景 | 调用频率 |
|---------|---------|----------|----------|
| rag/nlp/query.py | lookup() | 查询同义词扩展 | 中频 |

**依赖关系：**
- nltk.corpus.wordnet (英文同义词)
- redis (可选，实时更新)
- api/utils/file_utils.py → 文件路径处理

### 6. 姓氏处理 (surname.py)

#### 功能说明
专门处理中文姓氏相关的NLP任务。

**内部结构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                           姓氏处理模块                           │
├─────────────────────────────────────────────────────────────────┤
│ • 中文姓氏识别和处理                                            │
│   └── 被 deepdoc/parser/resume/step_two.py 调用                 │
│ • 姓氏相关的文本分析                                            │
│   └── 被简历解析模块调用                                        │
└─────────────────────────────────────────────────────────────────┘
```

**被调用关系：**

| 调用模块 | 调用接口 | 使用场景 | 调用频率 |
|---------|---------|----------|----------|
| deepdoc/parser/resume/step_two.py | 姓氏处理功能 | 简历姓名识别 | 低频 |

**依赖关系：**
- 基础文本处理功能 
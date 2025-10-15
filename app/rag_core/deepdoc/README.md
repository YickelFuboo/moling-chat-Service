# DeepDoc 模块调用关系说明

DeepDoc 是 RAGFlow 项目中的核心文档处理模块，提供文档解析和视觉识别功能。本文档详细说明了 DeepDoc 模块如何被项目中的其他模块调用。

## 模块结构

DeepDoc 模块主要包含两个子模块：
- **parser**: 文档解析器，支持多种文档格式
- **vision**: 视觉识别模块，包括OCR、布局识别、表格结构识别等

## 调用关系总览

### 1. RAG 应用模块调用

**说明**: RAG应用模块主要通过两种方式使用DeepDoc：
1. **继承方式**: 继承DeepDoc的解析器类，重写或扩展功能
2. **API调用方式**: 直接导入DeepDoc的类或函数，作为工具使用

| 源文件 | 调用文件 | 导入的模块/类 | 调用方式 | 主要用途 |
|---------|---------|---------------|----------|----------|
| `deepdoc/parser/utils.py` | `rag/app/book.py` | `deepdoc.parser.utils.get_text` | API调用 | 文本文件内容读取 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/book.py` | `deepdoc.parser.PdfParser` | **继承** | PDF文档解析和分块 |
| `deepdoc/parser/docx_parser.py` | `rag/app/book.py` | `deepdoc.parser.DocxParser` | API调用 | DOCX文档解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/book.py` | `deepdoc.parser.PlainParser` | API调用 | 纯文本PDF解析 |
| `deepdoc/parser/html_parser.py` | `rag/app/book.py` | `deepdoc.parser.HtmlParser` | API调用 | HTML文档解析 |
| `deepdoc/parser/docx_parser.py` | `rag/app/naive.py` | `deepdoc.parser.DocxParser` | **继承** | DOCX文档解析处理 |
| `deepdoc/parser/excel_parser.py` | `rag/app/naive.py` | `deepdoc.parser.ExcelParser` | **继承** | Excel表格解析处理 |
| `deepdoc/parser/html_parser.py` | `rag/app/naive.py` | `deepdoc.parser.HtmlParser` | **继承** | HTML文档解析处理 |
| `deepdoc/parser/json_parser.py` | `rag/app/naive.py` | `deepdoc.parser.JsonParser` | **继承** | JSON文档解析处理 |
| `deepdoc/parser/markdown_parser.py` | `rag/app/naive.py` | `deepdoc.parser.MarkdownParser` | **继承** | Markdown文档解析处理 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/naive.py` | `deepdoc.parser.PdfParser` | **继承** | PDF文档解析处理 |
| `deepdoc/parser/txt_parser.py` | `rag/app/naive.py` | `deepdoc.parser.TxtParser` | **继承** | 纯文本解析处理 |
| `deepdoc/parser/figure_parser.py` | `rag/app/naive.py` | `deepdoc.parser.figure_parser.VisionFigureParser` | API调用 | 图片和图表解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/naive.py` | `deepdoc.parser.pdf_parser.PlainParser, VisionParser` | API调用 | PDF解析器变体 |
| `deepdoc/vision/ocr.py` | `rag/app/picture.py` | `deepdoc.vision.OCR` | API调用 | 图片OCR识别和视觉LLM处理 |
| `deepdoc/parser/utils.py` | `rag/app/qa.py` | `deepdoc.parser.utils.get_text` | API调用 | 文本文件内容读取 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/qa.py` | `deepdoc.parser.PdfParser` | **继承** | PDF文档解析处理 |
| `deepdoc/parser/excel_parser.py` | `rag/app/qa.py` | `deepdoc.parser.ExcelParser` | **继承** | Excel表格解析处理 |
| `deepdoc/parser/docx_parser.py` | `rag/app/qa.py` | `deepdoc.parser.DocxParser` | **继承** | DOCX文档解析处理 |
| `deepdoc/parser/resume/__init__.py` | `rag/app/resume.py` | `deepdoc.parser.resume.refactor`<br>`deepdoc.parser.resume.step_one, step_two` | API调用 | 简历解析和重构 |
| `deepdoc/parser/utils.py` | `rag/app/tag.py` | `deepdoc.parser.utils.get_text` | API调用 | 文档标签提取 |
| `deepdoc/parser/utils.py` | `rag/app/table.py` | `deepdoc.parser.utils.get_text` | API调用 | 文本文件内容读取 |
| `deepdoc/parser/excel_parser.py` | `rag/app/table.py` | `deepdoc.parser.ExcelParser` | API调用 | 表格数据提取 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/presentation.py` | `deepdoc.parser.pdf_parser.VisionParser` | API调用 | PDF视觉解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/presentation.py` | `deepdoc.parser.PdfParser` | API调用 | PDF文档解析 |
| `deepdoc/parser/ppt_parser.py` | `rag/app/presentation.py` | `deepdoc.parser.PptParser` | API调用 | PowerPoint演示文稿解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/presentation.py` | `deepdoc.parser.PlainParser` | API调用 | 纯文本PDF解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/paper.py` | `deepdoc.parser.PdfParser` | API调用 | PDF学术论文解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/paper.py` | `deepdoc.parser.PlainParser` | API调用 | 纯文本PDF解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/manual.py` | `deepdoc.parser.PdfParser` | API调用 | PDF手册解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/manual.py` | `deepdoc.parser.PlainParser` | API调用 | 纯文本PDF解析 |
| `deepdoc/parser/docx_parser.py` | `rag/app/manual.py` | `deepdoc.parser.DocxParser` | API调用 | DOCX手册解析 |
| `deepdoc/parser/utils.py` | `rag/app/one.py` | `deepdoc.parser.utils.get_text` | API调用 | 文本文件内容读取 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/one.py` | `deepdoc.parser.PdfParser` | API调用 | PDF文档解析 |
| `deepdoc/parser/excel_parser.py` | `rag/app/one.py` | `deepdoc.parser.ExcelParser` | API调用 | Excel表格解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/one.py` | `deepdoc.parser.PlainParser` | API调用 | 纯文本PDF解析 |
| `deepdoc/parser/html_parser.py` | `rag/app/one.py` | `deepdoc.parser.HtmlParser` | API调用 | HTML文档解析 |
| `deepdoc/parser/utils.py` | `rag/app/laws.py` | `deepdoc.parser.utils.get_text` | API调用 | 文本文件内容读取 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/laws.py` | `deepdoc.parser.PdfParser` | API调用 | PDF法律文档解析 |
| `deepdoc/parser/docx_parser.py` | `rag/app/laws.py` | `deepdoc.parser.DocxParser` | API调用 | DOCX法律文档解析 |
| `deepdoc/parser/pdf_parser.py` | `rag/app/laws.py` | `deepdoc.parser.PlainParser` | API调用 | 纯文本PDF法律文档解析 |
| `deepdoc/parser/html_parser.py` | `rag/app/laws.py` | `deepdoc.parser.HtmlParser` | API调用 | HTML法律文档解析 |
| `deepdoc/parser/html_parser.py` | `rag/app/email.py` | `deepdoc.parser.HtmlParser` | API调用 | HTML邮件内容解析 |
| `deepdoc/parser/txt_parser.py` | `rag/app/email.py` | `deepdoc.parser.TxtParser` | API调用 | 纯文本邮件内容解析 |

### 2. API 服务模块调用

| 源文件 | 调用文件 | 导入的模块/类 | 调用方式 | 主要用途 |
|---------|---------|---------------|----------|----------|
| `deepdoc/parser/html_parser.py` | `api/apps/document_app.py` | `deepdoc.parser.html_parser.RAGFlowHtmlParser` | API调用 | HTML文档解析服务 |
| `deepdoc/parser/pdf_parser.py` | `api/db/services/task_service.py` | `deepdoc.parser.PdfParser` | API调用 | 任务服务中的PDF解析 |
| `deepdoc/parser/excel_parser.py` | `api/db/services/task_service.py` | `deepdoc.parser.excel_parser.RAGFlowExcelParser` | API调用 | 任务服务中的Excel解析 |

### 3. Agent 组件调用

| 源文件 | 调用文件 | 导入的模块/类 | 调用方式 | 主要用途 |
|---------|---------|---------------|----------|----------|
| `deepdoc/parser/html_parser.py` | `agent/component/invoke.py` | `deepdoc.parser.HtmlParser` | API调用 | Agent系统中的HTML解析 |

### 4. DeepDoc 内部模块调用

| 源文件 | 调用文件 | 导入的模块/类 | 调用方式 | 主要用途 |
|---------|---------|---------------|----------|----------|
| `deepdoc/vision/recognizer.py` | `deepdoc/vision/layout_recognizer.py` | `deepdoc.vision.Recognizer` | 内部模块调用 | 布局识别器实现 |
| `deepdoc/vision/operators.py` | `deepdoc/vision/layout_recognizer.py` | `deepdoc.vision.operators.nms` | 内部模块调用 | 非极大值抑制算法 |
| `deepdoc/vision/dla_cli.py` | `deepdoc/vision/layout_recognizer.py` | `deepdoc.vision.dla_cli.DLAClient` | 内部模块调用 | DLA客户端 |
| `deepdoc/vision/seeit.py` | `deepdoc/vision/t_recognizer.py` | `deepdoc.vision.seeit.draw_box` | 内部模块调用 | 绘制边界框 |
| `deepdoc/vision/layout_recognizer.py` | `deepdoc/vision/t_recognizer.py` | `deepdoc.vision.LayoutRecognizer` | 内部模块调用 | 布局识别器 |
| `deepdoc/vision/table_structure_recognizer.py` | `deepdoc/vision/t_recognizer.py` | `deepdoc.vision.TableStructureRecognizer` | 内部模块调用 | 表格结构识别器 |
| `deepdoc/vision/ocr.py` | `deepdoc/vision/t_recognizer.py` | `deepdoc.vision.OCR` | 内部模块调用 | OCR功能 |
| `deepdoc/vision/__init__.py` | `deepdoc/vision/t_recognizer.py` | `deepdoc.vision.init_in_out` | 内部模块调用 | 输入输出初始化 |
| `deepdoc/vision/seeit.py` | `deepdoc/vision/t_ocr.py` | `deepdoc.vision.seeit.draw_box` | 内部模块调用 | 绘制边界框 |
| `deepdoc/vision/ocr.py` | `deepdoc/vision/t_ocr.py` | `deepdoc.vision.OCR` | 内部模块调用 | OCR功能 |
| `deepdoc/vision/__init__.py` | `deepdoc/vision/t_ocr.py` | `deepdoc.vision.init_in_out` | 内部模块调用 | 输入输出初始化 |
| `deepdoc/parser/utils.py` | `deepdoc/parser/txt_parser.py` | `deepdoc.parser.utils.get_text` | 内部模块调用 | 文本解析器实现 |
| `deepdoc/parser/resume/entities.py` | `deepdoc/parser/resume/step_one.py` | `deepdoc.parser.resume.entities.degrees, regions, industries` | 内部模块调用 | 简历解析第一步 |
| `deepdoc/parser/resume/entities.py` | `deepdoc/parser/resume/step_two.py` | `deepdoc.parser.resume.entities.degrees, schools, corporations` | 内部模块调用 | 简历解析第二步 |
| `deepdoc/vision/ocr.py` | `deepdoc/parser/pdf_parser.py` | `deepdoc.vision.OCR` | 内部模块调用 | PDF解析器中的OCR处理 |
| `deepdoc/vision/layout_recognizer.py` | `deepdoc/parser/pdf_parser.py` | `deepdoc.vision.LayoutRecognizer` | 内部模块调用 | PDF解析器中的布局识别 |
| `deepdoc/vision/recognizer.py` | `deepdoc/parser/pdf_parser.py` | `deepdoc.vision.Recognizer` | 内部模块调用 | PDF解析器中的通用识别 |
| `deepdoc/vision/table_structure_recognizer.py` | `deepdoc/parser/pdf_parser.py` | `deepdoc.vision.TableStructureRecognizer` | 内部模块调用 | PDF解析器中的表格结构识别 |

## 主要调用模式

### 1. 继承方式 (Inheritance)
RAG应用模块通过继承DeepDoc的解析器类来扩展功能：
```python
# 继承PdfParser，重写__call__方法
class Pdf(PdfParser):
    def __call__(self, filename, binary=None, ...):
        # 自定义OCR和布局分析流程
        self.__images__(filename, zoomin, from_page, to_page, callback)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        # ... 其他自定义逻辑
        pass

# 继承DocxParser，添加图片处理功能
class Docx(DocxParser):
    def get_picture(self, document, paragraph):
        # 自定义图片提取逻辑
        pass
```

**使用场景**: 需要重写或扩展解析器的核心功能，如添加自定义的OCR流程、图片处理等

### 2. 组合方式 (Composition)
直接导入DeepDoc的类或函数，作为工具使用：
```python
# 直接导入使用
from deepdoc.parser import PdfParser, DocxParser, ExcelParser
from deepdoc.vision import OCR

# 实例化调用
pdf_parser = PdfParser()
docx_parser = DocxParser()
excel_parser = ExcelParser()
ocr = OCR()

# 调用方法
sections, tables = pdf_parser(filename, binary, from_page, to_page)
text = ocr(image_array)
```

**使用场景**: 直接使用解析器的标准功能，不需要修改核心逻辑

### 3. 工具函数调用
导入DeepDoc提供的工具函数：
```python
from deepdoc.parser.utils import get_text

# 调用工具函数
text = get_text(filename, binary)
```

**使用场景**: 使用通用的工具函数，如文本编码检测、文件读取等

### 4. 内部模块调用
DeepDoc模块内部各子模块之间的相互调用：
```python
# 在pdf_parser.py中调用vision模块
from deepdoc.vision import OCR, LayoutRecognizer, Recognizer, TableStructureRecognizer

# 在vision模块内部调用
from deepdoc.vision.seeit import draw_box
from deepdoc.vision.operators import nms
```

**使用场景**: 模块内部功能整合，如PDF解析器需要OCR和布局识别功能

## 核心功能模块

### Parser 模块
- **PdfParser**: PDF文档解析，支持OCR和布局识别
- **DocxParser**: Word文档解析
- **ExcelParser**: Excel表格解析
- **HtmlParser**: HTML文档解析
- **TxtParser**: 纯文本解析
- **PptParser**: PowerPoint演示文稿解析
- **JsonParser**: JSON文档解析
- **MarkdownParser**: Markdown文档解析

### Vision 模块
- **OCR**: 光学字符识别
- **LayoutRecognizer**: 文档布局识别
- **TableStructureRecognizer**: 表格结构识别
- **Recognizer**: 通用识别器

## 使用场景

DeepDoc 模块主要用于：
1. **文档预处理**: 将各种格式的文档转换为结构化文本
2. **内容提取**: 提取文档中的文本、表格、图片等信息
3. **智能分析**: 通过视觉识别技术分析文档布局和结构
4. **数据标准化**: 将不同格式的文档统一处理为可用的数据格式

该模块是 RAGFlow 项目中文档处理流程的核心，为后续的检索增强生成（RAG）提供了高质量的文档解析基础。
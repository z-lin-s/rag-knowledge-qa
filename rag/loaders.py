"""文档加载器：TXT / PDF。

设计：返回自研 Document 列表（page_content + metadata），不引入 langchain。
保留原 rag_app.py load_document 的后缀分派逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import LoaderError
from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Document:
    """单个文档片段的载体。

    page_content: 纯文本内容
    metadata:     来源信息（文件路径、页码等）
    """
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_txt(file_path: str | Path) -> list[Document]:
    """加载 UTF-8 文本文件，整体作为一个 Document。"""
    p = Path(file_path)
    if not p.exists():
        raise LoaderError(f"文件不存在: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 兜底用 GBK（Windows 中文环境常见）
        text = p.read_text(encoding="gbk", errors="ignore")
        log.warning("文件 %s 非 UTF-8，按 GBK 读取", p)
    return [Document(page_content=text, metadata={"source": str(p)})]


def load_pdf(file_path: str | Path) -> list[Document]:
    """用 PyMuPDF 解析 PDF，每页一个 Document。

    PyMuPDF 对复杂版式（多栏、表格、图文混排）有较好的文本提取能力。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise LoaderError("未安装 pymupdf，无法解析 PDF。请 pip install pymupdf") from e

    p = Path(file_path)
    if not p.exists():
        raise LoaderError(f"文件不存在: {p}")
    docs: list[Document] = []
    try:
        doc = fitz.open(p)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(p), "page": page_idx + 1},
                    )
                )
        doc.close()
    except Exception as e:
        raise LoaderError(f"PDF 解析失败 {p}: {e}") from e
    log.info("加载 PDF %s: %d 页有效文本", p.name, len(docs))
    return docs


def load_file(file_path: str | Path) -> list[Document]:
    """按后缀分派加载器。保留原 rag_app.py 的命名与逻辑。"""
    fp = str(file_path)
    log.debug("加载文档: %s", fp)
    if fp.endswith(".txt"):
        return load_txt(fp)
    if fp.endswith(".pdf"):
        return load_pdf(fp)
    raise LoaderError(f"不支持的文件类型: {fp}（仅支持 .txt / .pdf）")

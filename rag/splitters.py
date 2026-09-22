"""文本分块器。

实现：
  - RecursiveSplitter：递归字符分块（保留原 rag_app.py 的递归分割语义，
    自实现一份，分隔符对中文友好）。
  - SemanticSplitter：基于 embedding 相似度的语义分块，按句切分后
    贪心聚合相邻高相似度句子，遇到语义断崖切分。
  - HeaderAwareSplitter：识别 Markdown #/## 和 HTML <h1>~<h6> 作为
    段落边界，段落超长再用递归切分，标题行前缀到每个子片段。
"""
from __future__ import annotations

import math
import re
from typing import Protocol

from .clients import embed_texts
from .exceptions import LLMError, SplitterError
from .loaders import Document
from .logging_utils import get_logger

log = get_logger(__name__)


class Splitter(Protocol):
    """分块器接口。"""

    def split(self, docs: list[Document]) -> list[Document]:
        ...


class RecursiveSplitter:
    """递归字符分块。

    原理：依次尝试一组分隔符（从段落 → 句号 → 逗号 → 空格），用更细的
    分隔符不断二分，直到片段 ≤ chunk_size。重叠 chunk_overlap 保证上下文完整。
    """
    # 中英文通用的分隔符层级，从粗到细
    SEPARATORS: list[str] = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise SplitterError("chunk_size 必须 > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise SplitterError("chunk_overlap 必须 ∈ [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: list[Document]) -> list[Document]:
        """对每个 Document 递归切分，继承 source metadata。"""
        out: list[Document] = []
        for doc in docs:
            chunks = self._split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta["chunk_idx"] = idx
                out.append(Document(page_content=chunk, metadata=meta))
        log.info("分块完成: %d 文档 → %d 片段", len(docs), len(out))
        return out

    def _split_text(self, text: str) -> list[str]:
        """对单段文本做递归分割，返回 chunk 列表。"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        # 找到第一个能在 chunk_size 内切分的分隔符
        for sep in self.SEPARATORS:
            if sep == "":
                # 兜底：硬切
                return self._hard_split(text)
            idx = text.rfind(sep, 0, self.chunk_size)
            if idx > 0:
                head = text[: idx + len(sep)]
                tail = text[max(0, idx + len(sep) - self.chunk_overlap):]
                return [head] + self._split_text(tail)
        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        """无分隔符可用时，按 chunk_size 硬切，带 overlap。"""
        out: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        i = 0
        while i < len(text):
            out.append(text[i : i + self.chunk_size])
            i += step
        return out


class SemanticSplitter:
    """语义分块：按句切分 → 算句子 embedding → 相邻相似度 → 贪心聚合。

    原理：相邻句子语义相似度高时属于同一主题，合并；相似度断崖处切分。
    适合长文档主题划分，避免硬切破坏语义完整性。

    成本：构建索引时每个句子一次 embedding 调用（批量），网络往返 +1。
    """

    # 句子边界：句号/问号/叹号/中英文分号/换行
    _SENTENCE_RE = re.compile(r"(?<=[。！？；!?])\s*|\n+")

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        similarity_threshold: float = 0.6,
    ) -> None:
        if chunk_size <= 0:
            raise SplitterError("chunk_size 必须 > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise SplitterError("chunk_overlap 必须 ∈ [0, chunk_size)")
        if not 0.0 < similarity_threshold < 1.0:
            raise SplitterError("similarity_threshold 必须 ∈ (0, 1)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        # 兜底硬切用递归分块器
        self._fallback = RecursiveSplitter(chunk_size, chunk_overlap)

    def split(self, docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in docs:
            chunks = self._split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta["chunk_idx"] = idx
                meta["splitter"] = "semantic"
                out.append(Document(page_content=chunk, metadata=meta))
        log.info("语义分块完成: %d 文档 → %d 片段", len(docs), len(out))
        return out

    def _split_text(self, text: str) -> list[str]:
        # 1. 按句切分，过滤空串
        sentences = [s.strip() for s in self._SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return []
        # 2. 短文本直接返回，不必算 embedding
        if len(text) <= self.chunk_size:
            return [text]
        # 3. 批量算 embedding
        try:
            embs = embed_texts(sentences)
        except LLMError as e:
            raise SplitterError(f"语义分块需要 embedding 调用: {e}") from e
        if len(embs) != len(sentences):
            raise SplitterError("embedding 数量与句子数不一致")
        # 4. 相邻句子余弦相似度
        sims = [self._cos(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
        # 5. 贪心聚合：相似度 ≥ 阈值且长度未超限 → 累加；否则切分
        chunks: list[str] = []
        buf: list[str] = [sentences[0]]
        for i, sim in enumerate(sims):
            candidate = "\n".join(buf + [sentences[i + 1]])
            if sim >= self.similarity_threshold and len(candidate) <= self.chunk_size:
                buf.append(sentences[i + 1])
            else:
                self._flush(buf, chunks)
                buf = [sentences[i + 1]]
        self._flush(buf, chunks)
        return chunks

    def _flush(self, buf: list[str], sink: list[str]) -> None:
        """把累积的句子落盘：超长用递归硬切，否则直接拼接。"""
        if not buf:
            return
        text = "\n".join(buf)
        if len(text) > self.chunk_size:
            sink.extend(self._fallback._split_text(text))
        else:
            sink.append(text)

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


class HeaderAwareSplitter:
    """标题感知分块。

    识别 Markdown 标题（# ~ ######）和 HTML 标题（<h1> ~ <h6>）作为
    段落边界。每个 section 内若超过 chunk_size，再调用 RecursiveSplitter
    细分；标题行作为前缀拼到该 section 的每个子片段前，保留结构上下文。
    """

    _MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    _HTML_HEADER_RE = re.compile(
        r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL
    )

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise SplitterError("chunk_size 必须 > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise SplitterError("chunk_overlap 必须 ∈ [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._inner = RecursiveSplitter(chunk_size, chunk_overlap)

    def split(self, docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in docs:
            sections = self._split_by_headers(doc.page_content)
            idx = 0
            for header, body in sections:
                body = body.strip()
                if not body:
                    continue
                if len(body) > self.chunk_size:
                    sub_chunks = self._inner._split_text(body)
                else:
                    sub_chunks = [body]
                for sc in sub_chunks:
                    text = f"{header}\n{sc}" if header else sc
                    meta = dict(doc.metadata)
                    meta["chunk_idx"] = idx
                    meta["splitter"] = "header"
                    out.append(Document(page_content=text, metadata=meta))
                    idx += 1
        log.info("标题感知分块完成: %d 文档 → %d 片段", len(docs), len(out))
        return out

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """按标题切分。返回 [(header_line, body), ...]。

        HTML 标题先归一化为 Markdown 等价形式再统一处理；无标题返回 [("", text)]。
        """
        # 1. HTML 标题 → markdown 等价
        def _html_to_md(m: re.Match) -> str:
            level = len(m.group(1))
            content = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            return f"{'#' * level} {content}"

        normalized = self._HTML_HEADER_RE.sub(_html_to_md, text)

        # 2. 找所有 markdown 标题位置
        matches = list(self._MD_HEADER_RE.finditer(normalized))
        if not matches:
            return [("", normalized)]

        sections: list[tuple[str, str]] = []
        # 标题前的内容（如有）作为无标题段
        if matches[0].start() > 0:
            pre = normalized[: matches[0].start()].strip()
            if pre:
                sections.append(("", pre))
        for i, m in enumerate(matches):
            header_line = m.group(0)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
            body = normalized[start:end].strip()
            sections.append((header_line, body))
        return sections


def get_splitter(strategy: str = "recursive", **kwargs) -> Splitter:
    """工厂：按策略名返回分块器。

    :param strategy: "recursive" | "semantic" | "header"
    """
    if strategy == "recursive":
        return RecursiveSplitter(**kwargs)
    if strategy == "semantic":
        return SemanticSplitter(**kwargs)
    if strategy == "header":
        return HeaderAwareSplitter(**kwargs)
    raise SplitterError(f"未知分块策略: {strategy}（支持 recursive/semantic/header）")

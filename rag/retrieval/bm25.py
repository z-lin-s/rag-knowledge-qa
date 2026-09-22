"""BM25 稀疏检索。

依赖 rank-bm25 库。中文按字符切分（无需分词依赖，对短文本效果好）。
设计为可增量维护的内存索引：add / remove / search。
"""
from __future__ import annotations

from typing import Any

from ..exceptions import RetrievalError
from ..logging_utils import get_logger
from ..loaders import Document

log = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """中文按字、英文按空格 + 字符的混合切分。

    不引入 jieba 等分词依赖；对 RAG 知识库短片段，字粒度 BM25 召回足够。
    """
    # 英文先按空格切单词，再对每个单词降为字符会过细；
    # 这里采用：英文单词保留，中文字符逐字。简单实现：按字符切，过滤空白。
    return [c for c in text if not c.isspace()]


class BM25Retriever:
    """基于 rank_bm25.BM25Okapi 的稀疏检索器。

    维护 id → Document 的映射，支持增量 add / remove。
    """

    def __init__(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RetrievalError("未安装 rank-bm25，请 pip install rank-bm25") from e
        self._bm25_cls = BM25Okapi
        self._docs: dict[str, Document] = {}
        self._tokens: dict[str, list[str]] = {}
        self._bm25: BM25Okapi | None = None
        self._dirty = True

    def add(self, docs: list[Document], ids: list[str]) -> None:
        """增量加入文档（重复 id 覆盖）。"""
        for doc, doc_id in zip(docs, ids):
            self._docs[doc_id] = doc
            self._tokens[doc_id] = _tokenize(doc.page_content)
        self._dirty = True
        log.debug("BM25 加入 %d 条，总计 %d", len(docs), len(self._docs))

    def remove(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)
            self._tokens.pop(doc_id, None)
        self._dirty = True

    def _ensure_index(self) -> None:
        """懒构建 BM25 索引：仅在检索前且脏标记为 True 时重建。"""
        if not self._dirty and self._bm25 is not None:
            return
        corpus = list(self._tokens.values())
        if corpus:
            self._bm25 = self._bm25_cls(corpus)
        else:
            self._bm25 = None
        self._dirty = False
        log.debug("BM25 索引重建: %d 文档", len(corpus))

    def search(self, query: str, top_k: int = 20) -> list[Document]:
        """检索返回 top_k 个 Document，metadata 带 bm25_score。"""
        self._ensure_index()
        if self._bm25 is None or not self._docs:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as e:
            raise RetrievalError(f"BM25 检索失败: {e}") from e
        # 按 score 降序取 top_k，并关联 doc_id 与原文
        id_list = list(self._docs.keys())
        ranked = sorted(
            zip(id_list, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        out: list[Document] = []
        for doc_id, score in ranked:
            if score <= 0:
                continue  # 跳过完全不匹配的文档
            doc = self._docs[doc_id]
            meta = dict(doc.metadata)
            meta["id"] = doc_id
            meta["bm25_score"] = float(score)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out

    def count(self) -> int:
        return len(self._docs)

    def all_ids(self) -> list[str]:
        return list(self._docs.keys())

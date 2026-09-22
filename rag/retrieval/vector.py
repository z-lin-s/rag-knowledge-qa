"""向量稠密检索。

封装 VectorStore + Embedding 客户端，对外提供文本检索接口。
"""
from __future__ import annotations

from ..clients import embed_texts
from ..exceptions import RetrievalError
from ..logging_utils import get_logger
from ..loaders import Document
from ..store import VectorStore

log = get_logger(__name__)


class VectorRetriever:
    """向量检索器：依赖 VectorStore 实现和 embed_texts 客户端。"""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def search(self, query: str, top_k: int = 20) -> list[Document]:
        """将 query 向量化后在向量库检索 top_k 个文档。"""
        try:
            vecs = embed_texts([query])
            if not vecs:
                return []
            query_vec = vecs[0]
        except Exception as e:
            raise RetrievalError(f"向量化查询失败: {e}") from e
        return self._store.search(query_vec, top_k)

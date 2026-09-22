"""混合检索：向量 dense + BM25 sparse + RRF 结果融合。

RRF（Reciprocal Rank Fusion）：score(d) = Σ 1 / (k + rank_i(d))
无需归一化两路分数，只用排名融合，对两路召回量级不敏感。
"""
from __future__ import annotations

from ..exceptions import RetrievalError
from ..logging_utils import get_logger
from ..loaders import Document
from .bm25 import BM25Retriever
from .vector import VectorRetriever

log = get_logger(__name__)


class HybridRetriever:
    """混合检索器：组合向量检索 + BM25 检索，RRF 融合。"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        self._vec = vector_retriever
        self._bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        per_retriever_top_k: int = 20,
        final_top_k: int = 10,
    ) -> list[Document]:
        """混合检索。

        :param per_retriever_top_k: 每路粗召回数量
        :param final_top_k: RRF 融合后保留多少条
        """
        # 两路并发检索；这里串行实现（嵌入调用本身已是批量）。
        try:
            vec_docs = self._vec.search(query, top_k=per_retriever_top_k)
        except Exception as e:
            log.warning("向量检索失败，降级仅用 BM25: %s", e)
            vec_docs = []
        try:
            bm25_docs = self._bm25.search(query, top_k=per_retriever_top_k)
        except Exception as e:
            log.warning("BM25 检索失败，降级仅用向量: %s", e)
            bm25_docs = []

        if not vec_docs and not bm25_docs:
            return []

        # RRF: 用 (内容前 64 字符 + source) 作为去重 key
        # 同一段落可能被两路都召回，按内容指纹去重后再融合排名
        fused: dict[str, float] = {}
        doc_index: dict[str, Document] = {}

        for rank, doc in enumerate(vec_docs):
            key = self._key(doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in doc_index:
                doc_index[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = self._key(doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in doc_index:
                doc_index[key] = doc

        # 按融合分降序取 final_top_k
        ranked_keys = sorted(fused.items(), key=lambda x: x[1], reverse=True)[
            :final_top_k
        ]
        out: list[Document] = []
        for key, score in ranked_keys:
            doc = doc_index[key]
            meta = dict(doc.metadata)
            meta["rrf_score"] = float(score)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        log.info(
            "混合检索: 向量 %d + BM25 %d → 融合后 %d 条",
            len(vec_docs),
            len(bm25_docs),
            len(out),
        )
        return out

    @staticmethod
    def _key(doc: Document) -> str:
        """去重 key：内容前 64 字符 + source 路径，避免同一片段不同 metadata 算两次。"""
        return f"{doc.metadata.get('source', '')}::{doc.page_content[:64]}"

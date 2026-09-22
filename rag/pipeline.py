"""RAG 主链路编排（业务唯一入口）。

职责：
  - build_index: 从文件加载、分块、向量化、写入向量库（首次建库）
  - ingest_file: 增量入库单个文件
  - delete: 按 id 删除片段
  - ask: 混合检索 → Rerank → 拼 prompt → 调 LLM
  - 查询增强：rewrite_query（改写为检索友好形式）+ decompose_query（拆子问题）

api / webui / cli 都只做薄壳层，调用本模块。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clients import call_llm, embed_texts, stream_llm
from .config import get_settings
from .exceptions import RagError, RetrievalError, StoreError
from .loaders import Document, load_file
from .logging_utils import get_logger
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import rerank_docs
from .retrieval.vector import VectorRetriever
from .splitters import RecursiveSplitter, get_splitter
from .store import ChromaStore, VectorStore

log = get_logger(__name__)


# 保留原 rag_app.py 的 prompt 模板（语义不变，仅做格式清理）
PROMPT_TEMPLATE = """你是知识库问答助手，请严格根据下面提供的上下文回答用户问题。
如果上下文没有相关信息，直接回答不知道，不要编造内容。

【上下文】
{context}

【用户问题】
{question}
"""

_REWRITE_PROMPT = """你是检索查询改写助手。把用户问题改写为更适合向量检索的形式：
- 去掉口语化、寒暄、对话助词
- 把问句改成包含关键词的陈述或查询短语
- 保留原意，不要新增未提及的事实
- 输出一行纯文本，不要前缀和解释

【用户问题】
{question}
"""

_DECOMPOSE_PROMPT = """你是复杂问题拆解助手。把用户问题拆成可独立检索的子问题。

判断规则：
- 简单单一主题问题 → 直接返回原问题
- 含多个并列子问题、对比、复合条件 → 拆为多个独立子问题

仅返回 JSON 数组（字符串列表），不要任何解释：["子问题1", "子问题2", ...]

【用户问题】
{question}
"""


class RagPipeline:
    """RAG 主流程编排器。单例语义：同一进程内多个入口共享一个实例。"""

    def __init__(
        self,
        store: VectorStore | None = None,
        splitter: RecursiveSplitter | None = None,
    ) -> None:
        s = get_settings()
        # 向量库：默认 Chroma
        self.store = store or ChromaStore(persist_directory=str(s.persist_dir))
        # 分块器：默认递归
        self.splitter = splitter or RecursiveSplitter(
            chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
        )
        # 检索器
        self.vec_retriever = VectorRetriever(self.store)
        self.bm25_retriever = BM25Retriever()
        self.hybrid = HybridRetriever(
            self.vec_retriever,
            self.bm25_retriever,
            rrf_k=s.rrf_k,
        )
        # 若向量库已有数据，把现有片段加载进 BM25 索引（保持两路一致）
        self._sync_bm25_from_store()

    # ====================== 索引管理 ======================
    def build_index(self, doc_path: str | Path | None = None) -> int:
        """从文件建库：加载→分块→向量化→写入。返回写入条数。"""
        s = get_settings()
        path = Path(doc_path) if doc_path else s.doc_file
        log.info("开始建库: %s", path)
        docs = load_file(path)
        chunks = self.splitter.split(docs)
        if not chunks:
            log.warning("分块结果为空，未写入向量库")
            return 0
        # 批量向量化（chroma 写入需要 embeddings）
        texts = [c.page_content for c in chunks]
        embeddings = embed_texts(texts)
        ids = self.store.add(chunks, embeddings)
        # 同步进 BM25
        self.bm25_retriever.add(chunks, ids)
        log.info("建库完成: %d 条片段", len(ids))
        return len(ids)

    def ingest_file(self, file_path: str | Path) -> int:
        """增量入库：单个文件加载→分块→向量化→写入。返回写入条数。"""
        path = Path(file_path)
        log.info("增量入库: %s", path)
        docs = load_file(path)
        chunks = self.splitter.split(docs)
        if not chunks:
            return 0
        embeddings = embed_texts([c.page_content for c in chunks])
        ids = self.store.add(chunks, embeddings)
        self.bm25_retriever.add(chunks, ids)
        log.info("增量入库完成: %d 条片段", len(ids))
        return len(ids)

    def delete(self, ids: list[str]) -> None:
        """按 id 删除片段（向量库 + BM25 同步）。"""
        self.store.delete(ids)
        self.bm25_retriever.remove(ids)
        log.info("已删除 %d 条片段", len(ids))

    def _sync_bm25_from_store(self) -> None:
        """从向量库现有数据重建 BM25 索引（启动时调用一次）。

        Chroma 持久化的数据需要重新加载到 BM25 内存索引才能用。
        本轮简化：通过 get 接口拉取全部已有文档。
        """
        try:
            count = self.store.count()
        except Exception as e:
            log.warning("BM25 同步失败（无法读取向量库 count）: %s", e)
            return
        if count == 0:
            return
        # chroma 没有"列出全部"的直接接口，这里用 query 一个全零向量近似不可行；
        # 简化：本轮 BM25 仅索引增量入库的文档，向量库启动时已有的旧数据不重建。
        # 真正的多后端同步需要 VectorStore 暴露 list_all()，下轮补。
        log.info(
            "BM25 同步: 向量库已有 %d 条，本轮 BM25 仅索引新增文档；"
            "若需 BM25 覆盖旧数据，请删除 chroma_db/ 重建。",
            count,
        )

    # ====================== 查询增强 ======================
    def rewrite_query(self, question: str) -> str:
        """LLM 把用户问题改写为检索友好形式。

        适用：口语化提问、问句改关键词。失败/异常时返回原问题，不阻断主流程。
        """
        try:
            raw = call_llm(
                messages=[{
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(question=question),
                }],
                max_tokens=256,
            )
            rewritten = raw.strip()
            # 兜底：返回空或解释性长文时回退原问题
            if not rewritten or len(rewritten) > 5 * len(question) + 50:
                log.warning("改写结果异常，回退原问题: %s", rewritten[:80])
                return question
            log.info("查询改写: %r -> %r", question, rewritten)
            return rewritten
        except RagError as e:
            log.warning("查询改写失败，回退原问题: %s", e)
            return question

    def decompose_query(self, question: str) -> list[str]:
        """LLM 把复杂问题拆为子问题列表。简单问题返回 [question]。

        失败时返回 [question]，不阻断主流程。
        """
        try:
            raw = call_llm(
                messages=[{
                    "role": "user",
                    "content": _DECOMPOSE_PROMPT.format(question=question),
                }],
                max_tokens=256,
            )
            cleaned = _strip_code_fence(raw)
            data = json.loads(cleaned)
            if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
                log.info("查询拆解: %d 个子问题", len(data))
                return data
        except (RagError, json.JSONDecodeError) as e:
            log.warning("查询拆解失败，回退原问题: %s", e)
        return [question]

    # ====================== 主问答链路 ======================
    def _retrieve(
        self, question: str, enhance: bool
    ) -> tuple[list[str], list[Document], list[Document]]:
        """检索阶段（ask / ask_stream 共用，逻辑必须保持唯一一份）。

        :return: (子查询列表, 粗召回候选, Rerank 后片段)
        """
        s = get_settings()
        # 步骤 0（可选）：查询增强
        sub_queries = [question]
        if enhance:
            rewritten = self.rewrite_query(question)
            sub_queries = self.decompose_query(rewritten)
            log.info("查询增强启用，子查询数=%d", len(sub_queries))

        # 步骤 1：混合召回（多子查询时合并去重）
        all_candidates: list[Document] = []
        seen: set[str] = set()
        for sq in sub_queries:
            try:
                cand = self.hybrid.search(
                    sq,
                    per_retriever_top_k=s.retrieve_top_k,
                    final_top_k=s.rerank_candidates,
                )
            except Exception as e:
                log.warning("混合检索失败 (subq=%s): %s", sq, e)
                continue
            for d in cand:
                # 去重 key：优先 metadata.id，回退到内容前 64 字
                key = d.metadata.get("id") or d.page_content[:64]
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(d)

        # 步骤 2：Rerank 精排
        try:
            reranked = rerank_docs(question, all_candidates, top_n=s.rerank_top_n)
        except RetrievalError as e:
            log.warning("Rerank 失败，降级用召回前 %d 条: %s", s.rerank_top_n, e)
            reranked = all_candidates[: s.rerank_top_n]

        return sub_queries, all_candidates, reranked

    @staticmethod
    def _build_prompt(question: str, reranked: list[Document]) -> str:
        """用精排片段拼最终 prompt（保留原格式）。"""
        context_text = "\n\n".join([d.page_content for d in reranked])
        return PROMPT_TEMPLATE.format(context=context_text, question=question)

    def ask(self, question: str, *, enhance: bool = False) -> dict[str, Any]:
        """完整 RAG 主流程：检索→Rerank→生成（非流式，行为保持向后兼容）。

        :param enhance: True 时启用查询增强（rewrite + decompose + 多子查询召回合并）；
            False 走原逻辑（单次混合检索）。默认 False 保持向后兼容。
        :return: dict 包含 answer、contexts、num_candidates、（enhance 时含 sub_queries）
        :raises RagError: 任一关键环节失败
        """
        sub_queries, all_candidates, reranked = self._retrieve(question, enhance)
        prompt = self._build_prompt(question, reranked)
        try:
            answer = call_llm(
                messages=[{"role": "user", "content": prompt}],
            )
        except RagError as e:
            log.error("LLM 生成失败: %s", e)
            raise

        return {
            "answer": answer,
            "contexts": [{"content": d.page_content, "meta": d.metadata} for d in reranked],
            "num_candidates": len(all_candidates),
            **({"sub_queries": sub_queries} if enhance else {}),
        }

    def ask_stream(self, question: str, *, enhance: bool = False):
        """完整 RAG 主流程的流式版本。检索阶段与 ask 完全一致，仅生成阶段流式。

        以事件协议 yield dict，三端（CLI / SSE / Gradio）各自渲染：
          {"type": "meta",  "contexts": [...], "num_candidates": int}  首事件，一次
          {"type": "delta", "content": "增量文本"}                      0..N 次
          {"type": "done",  "answer": "完整答案"}                       末事件，一次
          {"type": "error", "detail": "错误信息"}                        生成失败时替代 done
        检索阶段失败会直接抛 RagError（与 ask 一致，此时还未 yield 任何事件）。
        """
        sub_queries, all_candidates, reranked = self._retrieve(question, enhance)
        contexts = [{"content": d.page_content, "meta": d.metadata} for d in reranked]
        yield {
            "type": "meta",
            "contexts": contexts,
            "num_candidates": len(all_candidates),
            **({"sub_queries": sub_queries} if enhance else {}),
        }

        prompt = self._build_prompt(question, reranked)
        chunks: list[str] = []
        try:
            for delta in stream_llm(messages=[{"role": "user", "content": prompt}]):
                chunks.append(delta)
                yield {"type": "delta", "content": delta}
        except RagError as e:
            log.error("LLM 流式生成失败: %s", e)
            yield {"type": "error", "detail": str(e)}
            return

        yield {"type": "done", "answer": "".join(chunks)}


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 代码围栏（与 evaluation 模块同名函数同语义）。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ====================== 单例访问 ======================
_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """获取 RAG 流程单例（api/webui/cli 共用，避免重复初始化）。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """重置单例（测试或运行时切换配置用）。"""
    global _pipeline
    _pipeline = None

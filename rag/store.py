"""向量库抽象 + 多后端实现（Chroma / Milvus / PGVector）+ 增量入库 / 删除。

设计：
  - VectorStore 是抽象基类，定义 add/delete/search/count 接口；
  - ChromaStore：开发默认后端，本地持久化；
  - MilvusStore：生产级分布式向量库，需独立 milvus 服务（pip install pymilvus）；
  - PgvectorStore：复用 Postgres + pgvector 扩展，需独立 PG 服务
    （pip install pgvector psycopg2-binary）。
  - 三者均显式传入 embeddings（由 rag.clients.embed_texts 统一生成）。
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .exceptions import StoreError
from .loaders import Document
from .logging_utils import get_logger

log = get_logger(__name__)


class VectorStore(ABC):
    """向量库统一接口。"""

    @abstractmethod
    def add(self, docs: list[Document], embeddings: list[list[float]]) -> list[str]:
        """批量写入，返回生成的 doc id 列表。"""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 id 删除。"""

    @abstractmethod
    def search(self, query_vec: list[float], top_k: int) -> list[Document]:
        """向量相似度检索，返回 top_k 个 Document（带 distance metadata）。"""

    @abstractmethod
    def count(self) -> int:
        """当前库中向量数。"""

    def exists(self) -> bool:
        """库是否已初始化（默认实现：count > 0）。"""
        return self.count() > 0


class ChromaStore(VectorStore):
    """Chroma 向量库实现（开发环境默认后端）。

    使用 PersistentClient，数据持久化到本地 persist_directory。
    """

    def __init__(
        self,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = "rag_docs",
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise StoreError("未安装 chromadb，请 pip install chromadb") from e
        self._persist_directory = str(persist_directory)
        self._collection_name = collection_name
        try:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise StoreError(f"Chroma 初始化失败: {e}") from e
        log.info(
            "ChromaStore 已加载: dir=%s, collection=%s, 现有 %d 条",
            self._persist_directory,
            collection_name,
            self._collection.count(),
        )

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> list[str]:
        if len(docs) != len(embeddings):
            raise StoreError("docs 与 embeddings 数量不一致")
        if not docs:
            return []
        ids = [str(uuid.uuid4()) for _ in docs]
        metadatas = [d.metadata for d in docs]
        texts = [d.page_content for d in docs]
        try:
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            log.info("ChromaStore 写入 %d 条，现有 %d", len(ids), self._collection.count())
            return ids
        except Exception as e:
            raise StoreError(f"Chroma 写入失败: {e}") from e

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
            log.info("ChromaStore 删除 %d 条", len(ids))
        except Exception as e:
            raise StoreError(f"Chroma 删除失败: {e}") from e

    def search(self, query_vec: list[float], top_k: int) -> list[Document]:
        try:
            res = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise StoreError(f"Chroma 检索失败: {e}") from e
        docs: list[Document] = []
        # query 返回结构: {ids:[[...]], documents:[[...]], metadatas:[[...]], distances:[[...]]}
        ids = res.get("ids", [[]])[0]
        texts = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, text in enumerate(texts):
            meta = dict(metas[i]) if i < len(metas) else {}
            meta["id"] = ids[i] if i < len(ids) else ""
            meta["distance"] = dists[i] if i < len(dists) else 0.0
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as e:
            raise StoreError(f"Chroma count 失败: {e}") from e


class MilvusStore(VectorStore):
    """Milvus 向量库实现。

    生产级分布式向量库。需独立 milvus 服务（docker compose 起 / Zilliz 云）。
    使用 MilvusClient 简化 API；collection 用动态字段，metadata 走 dynamic field。

    :param uri: milvus 连接串，如 http://localhost:19530
    :param collection_name: 集合名
    :param dim: 向量维度（首次 add 时自动探测；建表需已知）
    :param token: 可选鉴权 token
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "rag_docs",
        dim: int | None = None,
        token: str | None = None,
    ) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise StoreError(
                "未安装 pymilvus，请 pip install pymilvus 或 uv pip install pymilvus"
            ) from e
        self._uri = uri
        self._collection_name = collection_name
        self._dim = dim
        try:
            self._client = MilvusClient(uri=uri, token=token)
        except Exception as e:
            raise StoreError(f"Milvus 连接失败 {uri}: {e}") from e
        # 若指定了 dim 则预建集合
        if dim is not None and not self._client.has_collection(collection_name):
            self._create_collection(dim)
        log.info(
            "MilvusStore 已加载: uri=%s, collection=%s, dim=%s",
            uri, collection_name, dim,
        )

    def _create_collection(self, dim: int) -> None:
        """创建集合：id (auto int64) + vector + text + meta (动态字段)。"""
        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=dim,  # MilvusClient 自动 schema：id + vector
                auto_id=True,
                enable_dynamic_field=True,
            )
        except Exception as e:
            raise StoreError(f"Milvus 建表失败: {e}") from e

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> list[str]:
        if len(docs) != len(embeddings):
            raise StoreError("docs 与 embeddings 数量不一致")
        if not docs:
            return []
        # 探测维度并按需建表
        if self._dim is None:
            self._dim = len(embeddings[0])
            if not self._client.has_collection(self._collection_name):
                self._create_collection(self._dim)
        # 构造插入数据：vector + text + metadata 各字段平铺（dynamic field）
        rows: list[dict[str, Any]] = []
        ids: list[str] = []
        for doc, vec in zip(docs, embeddings):
            # 用 UUID 字符串做业务 id；Milvus auto_id 主键由其生成
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            row: dict[str, Any] = {
                "vector": vec,
                "text": doc.page_content,
                "doc_id": doc_id,  # 业务 id（区别于 milvus 主键）
            }
            # metadata 平铺为顶层字段（dynamic field 限制：值需为标量）
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    row[k] = v
                else:
                    # 复杂值序列化为 JSON 字符串
                    row[k] = json.dumps(v, ensure_ascii=False)
            rows.append(row)
        try:
            self._client.insert(
                collection_name=self._collection_name,
                data=rows,
            )
            log.info("MilvusStore 写入 %d 条", len(ids))
            return ids
        except Exception as e:
            raise StoreError(f"Milvus 写入失败: {e}") from e

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            # 按 doc_id 业务字段过滤删除（doc_id 为字符串，需加引号）
            id_list = ",".join(f'"{i}"' for i in ids)
            self._client.delete(
                collection_name=self._collection_name,
                filter=f"doc_id in [{id_list}]",
            )
            log.info("MilvusStore 删除 %d 条", len(ids))
        except Exception as e:
            raise StoreError(f"Milvus 删除失败: {e}") from e

    def search(self, query_vec: list[float], top_k: int) -> list[Document]:
        try:
            res = self._client.search(
                collection_name=self._collection_name,
                data=[query_vec],
                limit=top_k,
                output_fields=["text", "doc_id", "source", "page", "chunk_idx"],
            )
        except Exception as e:
            raise StoreError(f"Milvus 检索失败: {e}") from e
        # res 为 list[list[dict]]，每个 dict 含 id/distance/entity
        docs: list[Document] = []
        if not res or not res[0]:
            return docs
        for hit in res[0]:
            entity = hit.get("entity", {}) or {}
            meta = {k: v for k, v in entity.items() if k != "text"}
            meta["id"] = str(hit.get("id", entity.get("doc_id", "")))
            meta["distance"] = float(hit.get("distance", 0.0))
            text = entity.get("text", "")
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def count(self) -> int:
        try:
            if not self._client.has_collection(self._collection_name):
                return 0
            stats = self._client.get_collection_stats(self._collection_name)
            return int(stats.get("row_count", 0))
        except Exception as e:
            raise StoreError(f"Milvus count 失败: {e}") from e


class PgvectorStore(VectorStore):
    """PGVector 向量库实现。

    复用 Postgres + pgvector 扩展。需独立 PG 服务且已安装 pgvector 扩展。

    :param dsn: psycopg2 连接串，如 postgresql://user:pass@localhost:5432/dbname
    :param table_name: 表名
    :param dim: 向量维度（首次 add 时若表未创建则按此建表）
    """

    def __init__(
        self,
        dsn: str = "postgresql://postgres:postgres@localhost:5432/rag",
        table_name: str = "rag_docs",
        dim: int | None = None,
    ) -> None:
        try:
            import psycopg2
        except ImportError as e:
            raise StoreError(
                "未安装 psycopg2，请 pip install psycopg2-binary 或 uv pip install psycopg2-binary"
            ) from e
        self._dsn = dsn
        self._table = table_name
        self._dim = dim
        try:
            self._conn = psycopg2.connect(dsn)
            self._conn.autocommit = True
        except Exception as e:
            raise StoreError(f"PG 连接失败: {e}") from e
        self._ensure_extension()
        if dim is not None:
            self._ensure_table(dim)
        log.info(
            "PgvectorStore 已加载: table=%s, dim=%s",
            table_name, dim,
        )

    def _ensure_extension(self) -> None:
        """安装 pgvector 扩展（DBA 已装好 ext 即可，无需 superuser）。"""
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            raise StoreError(
                f"pgvector 扩展安装失败，请确认 PG 已装 pgvector: {e}"
            ) from e

    def _ensure_table(self, dim: int) -> None:
        """建表：id UUID PK / content TEXT / embedding VECTOR(dim) / metadata JSONB。"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id UUID PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR({dim}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_emb_idx "
                    f"ON {self._table} USING ivfflat (embedding vector_cosine_ops);"
                )
        except Exception as e:
            raise StoreError(f"PG 建表失败: {e}") from e

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> list[str]:
        if len(docs) != len(embeddings):
            raise StoreError("docs 与 embeddings 数量不一致")
        if not docs:
            return []
        if self._dim is None:
            self._dim = len(embeddings[0])
            self._ensure_table(self._dim)
        ids = [str(uuid.uuid4()) for _ in docs]
        try:
            with self._conn.cursor() as cur:
                for doc_id, doc, vec in zip(ids, docs, embeddings):
                    vec_str = "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"
                    meta_json = json.dumps(doc.metadata, ensure_ascii=False, default=str)
                    cur.execute(
                        f"INSERT INTO {self._table} (id, content, embedding, metadata) "
                        f"VALUES (%s, %s, %s::vector, %s::jsonb);",
                        (doc_id, doc.page_content, vec_str, meta_json),
                    )
            log.info("PgvectorStore 写入 %d 条", len(ids))
            return ids
        except Exception as e:
            raise StoreError(f"PG 写入失败: {e}") from e

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE id = ANY(%s);",
                    (ids,),
                )
            log.info("PgvectorStore 删除 %d 条", len(ids))
        except Exception as e:
            raise StoreError(f"PG 删除失败: {e}") from e

    def search(self, query_vec: list[float], top_k: int) -> list[Document]:
        try:
            vec_str = "[" + ",".join(f"{float(x):.8f}" for x in query_vec) + "]"
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, content, metadata, "
                    f"1 - (embedding <=> %s::vector) AS score "
                    f"FROM {self._table} "
                    f"ORDER BY embedding <=> %s::vector "
                    f"LIMIT %s;",
                    (vec_str, vec_str, top_k),
                )
                rows = cur.fetchall()
        except Exception as e:
            raise StoreError(f"PG 检索失败: {e}") from e
        docs: list[Document] = []
        for row in rows:
            doc_id, content, meta_json, score = row
            try:
                meta = json.loads(meta_json) if isinstance(meta_json, str) else dict(meta_json or {})
            except (json.JSONDecodeError, TypeError):
                meta = {}
            meta["id"] = str(doc_id)
            # <=> 是余弦距离（pgvector），1 - 距离 = 相似度；distance 字段统一记为 1 - score
            meta["distance"] = float(1.0 - score) if score is not None else 0.0
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def count(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table};")
                return int(cur.fetchone()[0])
        except Exception as e:
            raise StoreError(f"PG count 失败: {e}") from e


def get_store(backend: str = "chroma", **kwargs: Any) -> VectorStore:
    """工厂：按后端名返回向量库。

    :param backend: "chroma" | "milvus" | "pgvector"
    """
    if backend == "chroma":
        return ChromaStore(**kwargs)
    if backend == "milvus":
        return MilvusStore(**kwargs)
    if backend == "pgvector":
        return PgvectorStore(**kwargs)
    raise StoreError(f"未知向量库后端: {backend}（支持 chroma/milvus/pgvector）")

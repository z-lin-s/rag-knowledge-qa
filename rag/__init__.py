"""rag —— 增强型 RAG 私有知识库问答系统主包。

子模块：
  - config         统一配置
  - exceptions     异常层级
  - logging_utils  日志工具
  - clients        LLM / Embedding 客户端（惰性加载 + 重试）
  - loaders        TXT / PDF 文档加载
  - splitters      文本分块（递归 / 语义 / 标题感知，工厂可切换）
  - retrieval      BM25 / 向量 / 混合 RRF / 智谱 Rerank
  - store          向量库抽象 + Chroma / Milvus / PGVector 多后端
  - pipeline       RAG 主链路编排（业务唯一入口，含查询增强）
  - evaluation     RAG 评测：recall@k / precision@k / faithfulness
  - agent          Agentic-RAG：LLM 路由判断是否检索 + 执行
  - api            FastAPI 后端
  - webui          Gradio 本地演示
"""

__version__ = "0.1.0"

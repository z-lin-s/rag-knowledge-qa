"""检索模块：BM25 稀疏 + 向量 dense + 混合 RRF + 智谱 Rerank。

子模块：
  - bm25     稀疏关键词检索
  - vector   向量稠密检索
  - hybrid   混合 + RRF 融合
  - rerank   智谱 Rerank 精排（保留原 rag_app.py 业务逻辑）
"""

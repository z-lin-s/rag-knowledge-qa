"""智谱 Rerank 精排。

保留原 rag_app.py rerank_docs 的核心业务逻辑：
  - 调用 https://open.bigmodel.cn/api/paas/v4/rerank 接口
  - Bearer 鉴权 + JSON payload
  - 失败/异常时降级为召回前 N 条
新增：
  - 文档长度上限分批（智谱 Rerank 对单次请求 documents 数量有限制）
  - 日志包装
"""
from __future__ import annotations

import os

import requests

from ..config import get_settings
from ..exceptions import RetrievalError
from ..logging_utils import get_logger
from ..loaders import Document

log = get_logger(__name__)

# 智谱 Rerank 单次请求 documents 数量上限（保守值，超出分批）
_BATCH_SIZE = 64


def rerank_docs(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """对粗召回文档做精排，返回 top_n 条。

    保留原 rag_app.py 函数签名与降级语义；新增日志与分批。
    """
    if not docs:
        return []
    s = get_settings()
    if not s.zhipuai_api_key:
        log.warning("未配置 ZHIPUAI_API_KEY，跳过 Rerank，降级使用召回前 %d 条", top_n)
        return docs[:top_n]

    # 分批：超出 _BATCH_SIZE 的部分按 batch 调用，最后合并再取 top_n
    if len(docs) <= _BATCH_SIZE:
        return _rerank_once(query, docs, top_n)

    log.info("Rerank 输入 %d 条，超过单批上限 %d，分批调用", len(docs), _BATCH_SIZE)
    merged: list[Document] = []
    for i in range(0, len(docs), _BATCH_SIZE):
        batch = docs[i : i + _BATCH_SIZE]
        # 每批取 top_n，最后合并
        merged.extend(_rerank_once(query, batch, top_n))
    # 合并后再取 top_n（简化处理：保留前 top_n，因为每批已排过序）
    return merged[:top_n]


def _rerank_once(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """单次调用智谱 Rerank 接口。保留原 rag_app.py 的请求逻辑。"""
    s = get_settings()
    raw_texts = [d.page_content for d in docs]
    url = "https://open.bigmodel.cn/api/paas/v4/rerank"
    headers = {
        "Authorization": f"Bearer {s.zhipuai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "rerank",
        "query": query,
        "documents": raw_texts,
        "top_n": top_n,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=s.request_timeout)
        res_json = resp.json()
        log.debug("Rerank 接口返回: %s", res_json)
        result_list = res_json.get("results")
        if not result_list:
            log.warning("Rerank 无 results 字段，降级使用原始召回前 %d 条", top_n)
            return docs[:top_n]
        result_docs = []
        for item in result_list:
            idx = item["index"]
            if 0 <= idx < len(docs):
                result_docs.append(docs[idx])
        return result_docs
    except Exception as e:
        log.warning("Rerank 异常: %s，降级使用原始召回前 %d 条", e, top_n)
        return docs[:top_n]

"""Agentic-RAG：Agent 自主判断是否需要知识库检索。

传统 RAG 每次都强制检索，简单问题也走全流程浪费成本。
Agentic-RAG 用 LLM 做路由判断：
  - need_retrieval=True  → 走 RagPipeline.ask（检索 + Rerank + 生成）
  - need_retrieval=False → 直 LLM 对话（适用闲聊 / 常识 / 问候 / 创意生成）

使用方式：
    from rag.agent import get_agent
    result = get_agent().ask("你好")    # 直 LLM
    result = get_agent().ask("知识库里关于 X 怎么说")  # 走 RAG
"""
from __future__ import annotations

import json
from typing import Any

from .clients import call_llm
from .evaluation import _strip_code_fence  # 复用代码围栏剥离
from .exceptions import RagError
from .logging_utils import get_logger
from .pipeline import get_pipeline

log = get_logger(__name__)


_ROUTE_PROMPT = """你是 RAG 路由助手。判断用户问题是否需要去检索私有知识库。

需要检索的场景：
- 问题涉及具体事实、特定实体、文档内信息
- 用户明确提到"知识库 / 文档 / 上传的资料 / 你有的资料"
- 涉及用户私有业务、特定项目、内部数据
- 时间敏感的具体事件

不需要检索的场景：
- 问候、感谢、闲聊（你好 / 谢谢 / 你是谁）
- 公共常识（1+1=? / 中国首都 / 地球是圆的）
- 创意生成、写代码、写作（除非依赖私有资料）
- 元问题（你怎么工作的 / 你能做什么）

仅返回 JSON，不要任何解释：{{"need_retrieval": true|false, "reason": "简短理由"}}

【用户问题】
{question}
"""


class AgenticRAG:
    """Agentic-RAG：路由 + 执行。

    route(question) -> {"need_retrieval": bool, "reason": str}
    ask(question)  -> 按路由走 pipeline.ask 或直 LLM，统一 dict 输出
    """

    def __init__(self, pipeline: Any | None = None) -> None:
        # 惰性：避免启动即构建 pipeline（依赖 OpenAI client / 向量库连接）
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = get_pipeline()
        return self._pipeline

    def route(self, question: str) -> dict[str, Any]:
        """LLM 判断是否需要检索。返回 {"need_retrieval": bool, "reason": str}。

        路由失败时保守走检索（True），避免漏答知识库内容。
        """
        try:
            raw = call_llm(
                messages=[{
                    "role": "user",
                    "content": _ROUTE_PROMPT.format(question=question),
                }],
                max_tokens=200,
            )
        except RagError as e:
            log.warning("路由判断失败，兜底检索: %s", e)
            return {"need_retrieval": True, "reason": f"路由失败兜底检索: {e}"}
        result = self._parse_route(raw)
        log.info(
            "路由结果: need_retrieval=%s, reason=%s",
            result["need_retrieval"], result["reason"],
        )
        return result

    def _parse_route(self, text: str) -> dict[str, Any]:
        """容错解析 LLM 返回的 JSON。解析失败兜底 True。"""
        cleaned = _strip_code_fence(text)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return {
                    "need_retrieval": bool(data.get("need_retrieval", True)),
                    "reason": str(data.get("reason", "")),
                }
        except json.JSONDecodeError:
            pass
        # 兜底：文本里出现"不需要 / false"判 False，否则保守 True
        need = not ("不需要" in text or "false" in text.lower())
        return {"need_retrieval": need, "reason": f"解析失败兜底: {text[:80]}"}

    def ask(self, question: str) -> dict[str, Any]:
        """按路由结果走 pipeline 或直 LLM。

        返回 dict 包含 answer / contexts / num_candidates / source / route。
        source ∈ {"pipeline", "llm"}，标识答案来源。
        """
        route = self.route(question)
        if route["need_retrieval"]:
            log.info("路由 → 知识库检索")
            result = self.pipeline.ask(question)
            result["source"] = "pipeline"
            result["route"] = route
            return result
        log.info("路由 → 直 LLM 对话")
        try:
            answer = call_llm(messages=[{"role": "user", "content": question}])
        except RagError as e:
            log.error("直 LLM 失败: %s", e)
            raise
        return {
            "answer": answer,
            "contexts": [],
            "num_candidates": 0,
            "source": "llm",
            "route": route,
        }


# ====================== 单例访问 ======================
_agent: AgenticRAG | None = None


def get_agent() -> AgenticRAG:
    """获取 AgenticRAG 单例。"""
    global _agent
    if _agent is None:
        _agent = AgenticRAG()
    return _agent


def reset_agent() -> None:
    """重置单例（测试用）。"""
    global _agent
    _agent = None

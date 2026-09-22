"""RAG 评测：召回率 + 答案忠实度（抑制幻觉）。

指标：
  - recall_at_k / precision_at_k：纯集合运算，无需 LLM。
  - faithfulness：用 LLM 把答案拆原子陈述 → 逐条判定能否由 contexts 推出，
    返回支持率。忠实度低表示答案中存在幻觉。

使用方式：
    from rag.evaluation import recall_at_k, faithfulness
    r = recall_at_k(retrieved_ids, gold_ids, k=5)
    f = faithfulness(answer, contexts)
"""
from __future__ import annotations

import json
import re
from typing import Any

from .clients import call_llm
from .exceptions import RagError
from .logging_utils import get_logger

log = get_logger(__name__)


# ====================== 离线指标（无 LLM） ======================
def recall_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int = 5
) -> float:
    """top-k 召回率: |retrieved[:k] ∩ relevant| / |relevant|。

    :param retrieved_ids: 检索返回的 id 顺序（按相关度降序）
    :param relevant_ids: 标注的金标相关 id 集合
    :param k: 取前 k 个计算
    :return: 0.0~1.0；relevant 为空返回 0.0
    """
    if not relevant_ids:
        return 0.0
    gold = set(relevant_ids)
    topk = retrieved_ids[:k]
    hit = sum(1 for i in topk if i in gold)
    return hit / len(gold)


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int = 5
) -> float:
    """top-k 精确率: topk 中相关数 / k。

    与召回率互补：召回看"金标被找回多少"，精确看"找回来的对不对"。
    """
    if k <= 0:
        return 0.0
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    gold = set(relevant_ids)
    return sum(1 for i in topk if i in gold) / len(topk)


# ====================== LLM 评测 ======================
_EXTRACT_PROMPT = """你是 RAG 评测助手。请把下面这段答案拆解成原子陈述（每条单一事实、不可再分），以 JSON 数组返回，不要任何解释文字。

【答案】
{answer}

【输出格式】
["陈述1", "陈述2", ...]
"""

_VERIFY_PROMPT = """你是 RAG 评测助手。对下面每条原子陈述，判断它是否能由给定的上下文推出（context 中能直接找到证据，或可由上下文事实合理推出）。
能推出标 "yes"，不能标 "no"。仅返回 JSON 数组，每项形如 {{"statement": "...", "supported": "yes"|"no"}}，不要任何解释。

【上下文】
{context}

【原子陈述列表】
{statements}
"""


def faithfulness(answer: str, contexts: list[str]) -> dict[str, Any]:
    """评估答案对上下文的忠实度（抑制幻觉）。

    流程：LLM 把答案拆原子陈述 → 逐条判定能否由 contexts 推出 → 返回比率。
    忠实度低 = 答案含 contexts 不支持的内容（幻觉）。

    :param answer: 待评答案文本
    :param contexts: 检索到的上下文片段列表
    :return: {"faithfulness": float 0~1, "total": int, "supported": int, "details": list}
    """
    if not answer.strip():
        return {"faithfulness": 0.0, "total": 0, "supported": 0, "details": []}
    context_text = "\n\n".join(contexts) if contexts else "(无上下文)"

    # 1. 拆原子陈述
    try:
        raw = call_llm(
            messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(answer=answer)}]
        )
    except RagError as e:
        log.warning("拆原子陈述失败: %s", e)
        return {
            "faithfulness": 0.0, "total": 0, "supported": 0,
            "details": [], "error": str(e),
        }
    statements = _safe_json_list(raw)
    if not statements:
        log.warning("拆出的原子陈述为空，原始返回: %s", raw[:200])
        return {"faithfulness": 0.0, "total": 0, "supported": 0, "details": []}

    # 2. 批量判定
    try:
        stmts_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        raw2 = call_llm(
            messages=[{
                "role": "user",
                "content": _VERIFY_PROMPT.format(context=context_text, statements=stmts_text),
            }]
        )
    except RagError as e:
        log.warning("忠实度判定失败: %s", e)
        return {
            "faithfulness": 0.0, "total": len(statements), "supported": 0,
            "details": [], "error": str(e),
        }
    details = _safe_json_list_of_dicts(raw2, statements)
    supported = sum(
        1 for d in details
        if str(d.get("supported", "")).lower().strip() == "yes"
    )
    ratio = supported / len(statements) if statements else 0.0
    return {
        "faithfulness": ratio,
        "total": len(statements),
        "supported": supported,
        "details": details,
    }


# ====================== JSON 容错解析 ======================
def _safe_json_list(text: str) -> list[str]:
    """容错解析 LLM 返回的 JSON 字符串数组（兼容代码块包裹 / 散文兜底）。"""
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # 兜底：按行提取，去掉序号/项目符号
    out: list[str] = []
    for line in cleaned.splitlines():
        s = line.strip()
        if not s or s.startswith(("[", "]", "{", "}")):
            continue
        s = re.sub(r"^\d+[\.\、）]\s*", "", s)
        s = re.sub(r"^[-*]\s*", "", s)
        if s:
            out.append(s)
    return out


def _safe_json_list_of_dicts(text: str, fallback: list[str]) -> list[dict]:
    """容错解析 LLM 返回的 JSON 对象数组。解析失败时降级为逐条 no。"""
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            out: list[dict] = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    out.append(item)
                else:
                    out.append({
                        "statement": fallback[i] if i < len(fallback) else str(item),
                        "supported": "no",
                    })
            return out
    except json.JSONDecodeError:
        pass
    # 兜底：每条标 no
    return [{"statement": s, "supported": "no"} for s in fallback]


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 代码围栏。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

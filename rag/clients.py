"""LLM / Embedding 客户端：惰性加载 + 超时 + 可重试错误重试。

设计要点：
  1. 模块级单例，首次调用时构造 client，避免启动即网络握手；
  2. tenacity 重试只覆盖"可重试错误"（超时、连接、5xx、限流），
     对 4xx 鉴权/参数错误立即抛出，避免付费接口烧钱；
  3. Embedding 维度探测：换模型后旧向量库维度不匹配会主动报错提示重建。
"""
from __future__ import annotations

from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import get_settings
from .exceptions import LLMError
from .logging_utils import get_logger

log = get_logger(__name__)

# 可重试错误：网络层 + 服务端 5xx + 限流
_RETRYABLE = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)

# 全局单例
_llm_client: OpenAI | None = None
_embedding_dim: int | None = None


def _get_client() -> OpenAI:
    """惰性构造并缓存 OpenAI 兼容 client。"""
    global _llm_client
    if _llm_client is None:
        s = get_settings()
        log.debug("初始化 OpenAI 客户端: base_url=%s", s.openai_base_url)
        _llm_client = OpenAI(
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
            timeout=s.request_timeout,
        )
    return _llm_client


# 通用重试装饰器：仅对可重试错误生效
_retry_decorator = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)


@_retry_decorator
def call_llm(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """调用对话模型，返回文本内容。

    :param messages: OpenAI 消息格式 [{"role":"user","content":"..."}]
    :param system_prompt: 可选 system 提示
    :param temperature: 默认 0，保证可复现
    :param max_tokens: 可选上限
    :raises LLMError: 鉴权/参数错误或重试后仍失败
    """
    s = get_settings()
    msgs = list(messages)
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}, *msgs]
    kwargs: dict[str, Any] = {
        "model": s.openai_model_id,
        "messages": msgs,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        log.debug("调用 LLM: model=%s, messages=%d 条", s.openai_model_id, len(msgs))
        resp = _get_client().chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        log.debug("LLM 返回 %d 字符", len(text))
        return text
    except (AuthenticationError, BadRequestError, NotFoundError) as e:
        # 4xx 立即抛出，不重试
        raise LLMError(f"LLM 调用失败(不可重试): {type(e).__name__}: {e}") from e
    except _RETRYABLE as e:
        # 重试耗尽后 tenacity 会 re-raise，这里兜底包装
        raise LLMError(f"LLM 调用失败(重试耗尽): {type(e).__name__}: {e}") from e
    except Exception as e:
        raise LLMError(f"LLM 调用未知错误: {type(e).__name__}: {e}") from e


def stream_llm(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
):
    """流式调用对话模型，逐块 yield 增量文本（delta）。

    与 call_llm 参数语义一致、异常类型一致；区别是本函数为生成器：
      - 不做 tenacity 重试（重试装饰器对生成器函数无效，且流式在首 token
        到达后重试没有意义），连接/鉴权错误在首次迭代时直接抛 LLMError；
      - 上游使用 OpenAI 兼容 SDK 的 stream=True。
    :raises LLMError: 鉴权/参数错误或网络失败，由调用方决定降级方式
    """
    s = get_settings()
    msgs = list(messages)
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}, *msgs]
    kwargs: dict[str, Any] = {
        "model": s.openai_model_id,
        "messages": msgs,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        stream = _get_client().chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except (AuthenticationError, BadRequestError, NotFoundError) as e:
        raise LLMError(f"LLM 流式调用失败(不可重试): {type(e).__name__}: {e}") from e
    except _RETRYABLE as e:
        raise LLMError(f"LLM 流式调用失败(网络/超时): {type(e).__name__}: {e}") from e
    except Exception as e:
        raise LLMError(f"LLM 流式调用未知错误: {type(e).__name__}: {e}") from e


@_retry_decorator
def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化，返回与输入等长的向量列表。

    :raises LLMError: 鉴权错误或重试后仍失败
    """
    if not texts:
        return []
    s = get_settings()
    try:
        log.debug("调用 Embedding: model=%s, %d 段文本", s.embedding_model_id, len(texts))
        resp = _get_client().embeddings.create(model=s.embedding_model_id, input=texts)
        # 按 index 排序保证顺序与输入一致
        vecs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        _detect_dim(len(vecs[0]))
        return vecs
    except (AuthenticationError, BadRequestError, NotFoundError) as e:
        raise LLMError(f"Embedding 调用失败(不可重试): {type(e).__name__}: {e}") from e
    except _RETRYABLE as e:
        raise LLMError(f"Embedding 调用失败(重试耗尽): {type(e).__name__}: {e}") from e
    except Exception as e:
        raise LLMError(f"Embedding 未知错误: {type(e).__name__}: {e}") from e


def _detect_dim(dim: int) -> None:
    """记录首次发现的向量维度；后续若维度不一致立即报错。

    场景：换 embedding 模型后，旧 chroma_db 维度不匹配，
    这里主动提示用户重建索引而不是写入脏数据。
    """
    global _embedding_dim
    if _embedding_dim is None:
        _embedding_dim = dim
        log.info("检测到 Embedding 维度: %d", dim)
    elif _embedding_dim != dim:
        raise LLMError(
            f"Embedding 维度不一致: 期望 {_embedding_dim}, 实际 {dim}。"
            "可能是更换了 embedding 模型，请删除 chroma_db/ 重建索引。"
        )


def get_embedding_dim() -> int | None:
    """返回当前已探测的向量维度（未调用过 embed_texts 则为 None）。"""
    return _embedding_dim


def reset_clients() -> None:
    """重置单例（测试用）。"""
    global _llm_client, _embedding_dim
    _llm_client = None
    _embedding_dim = None

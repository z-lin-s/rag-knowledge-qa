"""Gradio 本地演示 UI。

薄壳层：调用 RagPipeline.ask，本文件只做 UI 适配。
启动：python -m rag.webui
"""
from __future__ import annotations

from .config import get_settings
from .logging_utils import get_logger
from .pipeline import get_pipeline

log = get_logger(__name__)


def _format_with_contexts(answer: str, contexts: list) -> str:
    """把检索到的上下文以小字附在答案后，便于调试（流式/非流式共用）。"""
    ctx_preview = "\n".join(f"- {c['content'][:80]}..." for c in contexts[:3])
    if not ctx_preview:
        return answer
    return f"{answer}\n\n---\n📋 参考:\n{ctx_preview}"


def chat(message: str, history: list) -> str:
    """Gradio ChatInterface 回调（非流式，保留原行为）。

    history 由 Gradio 维护，本轮不显式拼入 prompt（pipeline 内部无多轮记忆，
    下轮 Agentic-RAG 再补多轮上下文）。
    """
    try:
        result = get_pipeline().ask(message)
        return _format_with_contexts(result["answer"], result["contexts"])
    except Exception as e:
        log.error("webui chat 失败: %s", e)
        return f"⚠️ 出错了: {e}"


def chat_stream(message: str, history: list):
    """Gradio ChatInterface 流式回调：生成器，每次 yield 累积的完整回答。

    Gradio 对 generator fn 的协议是"用本次 yield 的字符串整体替换助手气泡"，
    因此这里维护 acc 增量累积后整体 yield，实现逐字显示效果。
    """
    acc = ""
    contexts: list = []
    try:
        for event in get_pipeline().ask_stream(message):
            etype = event.get("type")
            if etype == "meta":
                contexts = event.get("contexts", [])
            elif etype == "delta":
                acc += event["content"]
                yield acc
            elif etype == "error":
                yield f"⚠️ 出错了: {event.get('detail', '未知错误')}"
                return
            elif etype == "done":
                acc = event.get("answer", acc)
        yield _format_with_contexts(acc, contexts)
    except Exception as e:
        log.error("webui chat_stream 失败: %s", e)
        yield f"⚠️ 出错了: {e}"


def main() -> None:
    """启动 Gradio 服务。"""
    import gradio as gr

    s = get_settings()
    demo = gr.ChatInterface(
        fn=chat_stream,
        title="RAG 私有知识库问答系统",
        description="混合检索 (BM25 + 向量 + RRF) + 智谱 Rerank（流式输出）",
    )
    log.info("启动 Gradio: port=%d", s.gradio_port)
    demo.launch(server_port=s.gradio_port)


if __name__ == "__main__":
    main()

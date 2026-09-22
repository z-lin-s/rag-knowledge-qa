"""命令行入口：rag-cli 子命令。

子命令：
  build    从 knowledge.txt（或 --file）建库
  ask      单次问答（带参）或交互式 REPL（无参）
  upload   增量入库单个文件
  drop     按 id 删除片段
  serve    启动 FastAPI 后端
  webui    启动 Gradio 演示 UI

入口注册在 pyproject.toml: [project.scripts] rag-cli = "cli:main"
也可直接 python cli.py build
"""
from __future__ import annotations

import sys

import click

from rag.exceptions import RagError
from rag.logging_utils import get_logger

log = get_logger(__name__)


@click.group(help="增强型 RAG 知识库命令行工具")
def main() -> None:
    """顶层命令组。"""


@main.command(help="从文件建库（默认 ./knowledge.txt）")
@click.option(
    "--file",
    "-f",
    default=None,
    help="知识库文档路径，默认走 config.doc_file",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="重建索引：先删除现有 chroma_db 再建（embedding 换模型时用）",
)
def build(file: str | None, rebuild: bool) -> None:
    """建库 / 重建索引。"""
    from pathlib import Path

    from rag.config import get_settings
    from rag.pipeline import RagPipeline

    s = get_settings()
    if rebuild:
        import shutil

        if Path(s.persist_dir).exists():
            log.info("重建模式：删除 %s", s.persist_dir)
            shutil.rmtree(s.persist_dir)
    # 重建后单例已脏，强制重建 pipeline
    from rag import pipeline as _p

    _p.reset_pipeline()
    pipe = RagPipeline()
    n = pipe.build_index(file)
    click.echo(f"✅ 建库完成: {n} 条片段")


def _ask_stream_print(pipe, question: str) -> bool:
    """流式问答并实时打印。返回 False 表示生成阶段出错。"""
    contexts = []
    ok = True
    click.echo("\n回答：", nl=False)
    for event in pipe.ask_stream(question):
        etype = event.get("type")
        if etype == "meta":
            contexts = event.get("contexts", [])
        elif etype == "delta":
            click.echo(event["content"], nl=False)
        elif etype == "error":
            ok = False
            click.echo(f"\n⚠️ 出错: {event.get('detail', '未知错误')}", err=True)
        elif etype == "done":
            pass
    click.echo()  # 收尾换行
    if ok and contexts:
        click.echo("\n--- 参考 ---")
        for i, c in enumerate(contexts, 1):
            click.echo(f"片段{i}: {c['content'][:200]}...")
    return ok


@main.command(help="问答：带参单次问答，无参进交互式 REPL（默认流式）")
@click.argument("question", required=False)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="是否流式逐字输出，默认开启；--no-stream 走一次性返回",
)
def ask(question: str | None, stream: bool) -> None:
    """RAG 问答。"""
    from rag.pipeline import get_pipeline

    pipe = get_pipeline()
    if question:
        # 单次问答
        if stream:
            try:
                ok = _ask_stream_print(pipe, question)
            except RagError as e:
                click.echo(f"⚠️ 出错: {e}", err=True)
                sys.exit(1)
            if not ok:
                sys.exit(1)
            return
        try:
            result = pipe.ask(question)
            click.echo(f"\n回答：{result['answer']}")
            if result["contexts"]:
                click.echo("\n--- 参考 ---")
                for i, c in enumerate(result["contexts"], 1):
                    click.echo(f"片段{i}: {c['content'][:200]}...")
        except RagError as e:
            click.echo(f"⚠️ 出错: {e}", err=True)
            sys.exit(1)
    else:
        # 交互式 REPL
        click.echo("\nRAG 问答启动，输入 exit 退出")
        while True:
            try:
                q = input("\n请提问：").strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\n退出程序")
                break
            if q.lower() in ("exit", "quit", ":q"):
                click.echo("退出程序")
                break
            if not q:
                continue
            if stream:
                try:
                    _ask_stream_print(pipe, q)
                except RagError as e:
                    click.echo(f"⚠️ 出错: {e}", err=True)
                continue
            try:
                result = pipe.ask(q)
                click.echo(f"\n回答：{result['answer']}")
            except RagError as e:
                click.echo(f"⚠️ 出错: {e}", err=True)


@main.command(help="增量入库单个文件")
@click.argument("file_path", type=click.Path(exists=True))
def upload(file_path: str) -> None:
    """增量入库。"""
    from rag.pipeline import get_pipeline

    try:
        n = get_pipeline().ingest_file(file_path)
        click.echo(f"✅ 入库完成: {n} 条片段")
    except RagError as e:
        click.echo(f"⚠️ 出错: {e}", err=True)
        sys.exit(1)


@main.command(help="按 id 删除片段")
@click.argument("ids", nargs=-1, required=True)
def drop(ids: tuple[str, ...]) -> None:
    """删除片段。"""
    from rag.pipeline import get_pipeline

    try:
        get_pipeline().delete(list(ids))
        click.echo(f"✅ 删除 {len(ids)} 条")
    except RagError as e:
        click.echo(f"⚠️ 出错: {e}", err=True)
        sys.exit(1)


@main.command(help="启动 FastAPI 后端")
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=None, type=int, help="端口，默认走 config")
def serve(host: str, port: int | None) -> None:
    """启动 API。"""
    import uvicorn
    from rag.api import app
    from rag.config import get_settings

    s = get_settings()
    p = port or s.api_port
    click.echo(f"🚀 FastAPI 启动: http://{host}:{p}")
    uvicorn.run(app, host=host, port=p)

@main.command(help="启动 Gradio 演示 UI（流式输出）")
@click.option("--port", default=None, type=int, help="端口，默认走 config")
def webui(port: int | None) -> None:
    """启动 Gradio。"""
    from rag.webui import chat_stream
    import gradio as gr
    from rag.config import get_settings

    s = get_settings()
    p = port or s.gradio_port
    demo = gr.ChatInterface(
        fn=chat_stream,
        title="RAG 私有知识库问答系统",
        description="混合检索 (BM25 + 向量 + RRF) + 智谱 Rerank（流式输出）",
    )
    click.echo(f"🚀 Gradio 启动: http://127.0.0.1:{p}")
    demo.launch(server_port=p)


if __name__ == "__main__":
    main()

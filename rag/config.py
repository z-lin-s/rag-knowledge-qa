"""统一配置：从环境变量读取，提供合理默认值。

所有可调参数集中在这里，业务代码不硬编码任何数字/路径/模型名。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .exceptions import ConfigError

# 加载 .env（不存在不报错）
load_dotenv()


def _require(key: str) -> str:
    """读取必填环境变量，缺失抛 ConfigError。"""
    val = os.getenv(key)
    if not val:
        raise ConfigError(f"环境变量 {key} 未配置，请参考 .env.example")
    return val


@dataclass(frozen=True)
class Settings:
    # ---- API 凭证 ----
    openai_api_key: str = field(default_factory=lambda: _require("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _require("OPENAI_BASE_URL"))
    openai_model_id: str = field(default_factory=lambda: _require("OPENAI_MODEL_ID"))
    embedding_model_id: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_ID", "embedding-3")
    )
    zhipuai_api_key: str = field(default_factory=lambda: os.getenv("ZHIPUAI_API_KEY", ""))

    # ---- 路径 ----
    # 项目根 = rag/ 的父目录
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    persist_dir: Path = field(default_factory=lambda: Path("./chroma_db"))
    doc_file: Path = field(default_factory=lambda: Path("./knowledge.txt"))
    upload_dir: Path = field(default_factory=lambda: Path("./data/uploads"))

    # ---- 分块 ----
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---- 检索 ----
    # 每路粗召回数（向量 + BM25 各取这么多）
    retrieve_top_k: int = 20
    # RRF 融合后保留多少条进 Rerank
    rerank_candidates: int = 10
    # Rerank 后最终送入 LLM 的片段数
    rerank_top_n: int = 3
    # RRF 的 k 常数（业界默认 60，越小 top 越敏感）
    rrf_k: int = 60

    # ---- LLM / Embedding 调用 ----
    # 单次请求超时（秒）：非流式调用需等长回答全文生成完，默认放宽到 60s
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "60"))
    )
    # 可重试错误最大重试次数（仅超时/5xx）
    max_retries: int = 3
    retry_min_wait: float = 0.5
    retry_max_wait: float = 5.0

    # ---- 服务端口 ----
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    gradio_port: int = field(default_factory=lambda: int(os.getenv("GRADIO_PORT", "7860")))


# 全局单例：避免每次访问都重新读环境
_settings: Settings | None = None


def get_settings() -> Settings:
    """获取配置单例（首次访问时构造，缺失必填项会抛 ConfigError）。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """重置单例（测试或运行时改 env 后重新加载）。"""
    global _settings
    _settings = None

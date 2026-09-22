"""日志工具：统一格式、级别可配置。

用法：
    from rag.logging_utils import get_logger
    log = get_logger(__name__)
    log.info("向量库已加载")
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root() -> None:
    """配置 root logger，全局只执行一次。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger，自动初始化 root 配置。"""
    _configure_root()
    return logging.getLogger(name)

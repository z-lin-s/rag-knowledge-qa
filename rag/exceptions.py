"""统一异常层级。

所有 RAG 自定义异常都继承自 RagError，便于上层捕获与降级处理。
设计为"窄接口"：每个子类只标识故障来源，不携带复杂逻辑。
"""


class RagError(Exception):
    """增强型 RAG 知识库所有自定义异常的基类。"""


class ConfigError(RagError):
    """配置缺失或非法（如未配置 OPENAI_API_KEY）。"""


class LoaderError(RagError):
    """文档加载失败（文件不存在、格式不支持、解析异常）。"""


class SplitterError(RagError):
    """文本分块失败。"""


class RetrievalError(RagError):
    """检索失败（向量召回、BM25、Rerank）。"""


class StoreError(RagError):
    """向量库操作失败（建库、写入、读取、删除）。"""


class LLMError(RagError):
    """LLM / Embedding 调用失败。"""


class EvaluationError(RagError):
    """评测流程异常。"""

# Enhanced RAG KB

一套增强型 RAG 私有知识库问答系统：**混合检索（BM25 + 向量 + RRF 融合）+ 智谱 Rerank 精排**，配套多分片策略、向量库多后端适配、RAG 评测、Agentic-RAG，提供 CLI / FastAPI / Gradio 三种入口。

不依赖 LangChain，全部使用原生 SDK + 自研轻抽象，便于阅读、调试、定制。

## 特性

### 已实现（本轮）
- **混合检索**：BM25 稀疏召回 + 向量稠密召回，RRF 融合两路排名
- **智谱 Rerank 精排**：保留原 `rag_app.py` 业务逻辑，新增分批与日志
- **文档加载**：TXT / PDF（PyMuPDF，对复杂版式有较好支持）
- **多分片策略**：递归字符分块（中英文友好分隔符）+ 语义分块（embedding 相似度断崖切分）+ 标题感知分块（Markdown `#` / HTML `<h1>~<h6>` 作边界），工厂 `get_splitter(strategy=...)` 切换
- **向量库多后端**：Chroma（开发默认，本地持久化）/ Milvus（生产级，`pip install pymilvus`）/ PGVector（Postgres + pgvector，`pip install pgvector psycopg2-binary`），工厂 `get_store(backend=...)` 切换
- **查询增强**：`rewrite_query`（LLM 把口语化提问改写为检索友好形式）+ `decompose_query`（拆复杂问题为子问题）+ 多子查询召回合并去重，通过 `ask(question, enhance=True)` 启用
- **RAG 评测**：`recall_at_k` / `precision_at_k`（离线指标，无 LLM）+ `faithfulness(answer, contexts)`（LLM 拆原子陈述 → 逐条判定是否被 contexts 支持，返回支持率抑制幻觉）
- **Agentic-RAG**：`AgenticRAG` 用 LLM 路由判断 `need_retrieval`，闲聊 / 常识 / 问候走直 LLM，知识库相关问题才走 RAG；通过 `rag.agent.get_agent().ask(...)` 调用
- **LLM / Embedding 客户端**：惰性加载、超时控制、tenacity 重试（区分可重试错误与鉴权错误）、维度迁移检测
- **三种入口**：CLI（Click 子命令）、FastAPI、Gradio，共用 `RagPipeline` 单一业务入口
- **工程化**：统一异常层级、结构化日志、`.env.example`、`pyproject.toml`、`.gitignore`、可选依赖 `uv sync --extra milvus` / `--extra pgvector`

### 路线图（未来可能方向）
当前版本已覆盖"混合检索 + 多分片 + 多后端 + 评测 + Agentic-RAG + 工程化"。后续可能演进的方向：

- [ ] 多轮上下文记忆（当前 `ask` 无状态，每次独立问答）
- [x] 流式输出（CLI `ask` 默认逐字输出；API `POST /ask/stream`（SSE）；WebUI 流式渲染。非流式 `call_llm` / `ask` / `/ask` 保持不变）
- [ ] 多模态文档加载（图片 / 表格 OCR）
- [ ] 评测数据集批量回归（当前需手动调用 `faithfulness`）

## 已知限制

- **BM25 与现有 chroma_db 的同步**：`RagPipeline` 启动时**不会**自动把 `chroma_db/` 已有数据重建进 BM25 内存索引，BM25 仅覆盖本次进程内 `build` / `ingest_file` 新增的文档。
  - 影响：若直接对旧 `chroma_db` 跑 `ask`，BM25 召回会为 0，混合检索退化为纯向量。
  - 规避：首次使用务必 `rag-cli build --rebuild` 重建；或后续通过 `rag-cli upload` 增量入库的文档会同时进入两路索引。

## 快速开始

### 1. 环境准备
- Python ≥ 3.10
- 推荐 [uv](https://docs.astral.sh/uv/) 管理依赖（也可用 pip）

```bash
# 用 uv（推荐）
uv sync

# 或用 pip
python -m pip install -r requirements.txt
```

### 2. 配置 API Key
复制 `.env.example` 为 `.env`，填入你的 API 凭证。支持任何 OpenAI 兼容接口（智谱 GLM / Kimi / DeepSeek / OpenAI 官方）：

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL_ID=glm-4-flash
EMBEDDING_MODEL_ID=embedding-3
ZHIPUAI_API_KEY=your-zhipu-rerank-key
```

### 3. 建库
默认从 `./knowledge.txt` 加载，也可指定文件：

```bash
# 用项目入口
rag-cli build

# 或直接 python
python cli.py build

# 指定文件
rag-cli build --file ./data/my.pdf

# 重建（换 embedding 模型后维度不匹配时用）
rag-cli build --rebuild
```

### 4. 问答
```bash
# 单次问答
rag-cli ask "你的问题"

# 交互式 REPL（无参）
rag-cli ask
```

### 5. 启动后端服务
```bash
# FastAPI（默认 8000 端口）
rag-cli serve
# 访问 http://127.0.0.1:8000/docs 看 Swagger

# Gradio 演示 UI（默认 7860 端口）
rag-cli webui
```

### 6. 增量入库 / 删除
```bash
# 增量入库新文件
rag-cli upload ./data/new_doc.pdf

# 按 id 删除片段
rag-cli drop <id1> <id2>
```

## HTTP API

| Method | Path | 说明 |
|--------|------|------|
| GET  | `/health` | 健康检查 |
| POST | `/ask` | RAG 问答，body: `{"question": "..."}` |
| POST | `/ask/stream` | RAG 问答（SSE 流式），事件：`meta` / `delta` / `done` / `error` |
| POST | `/ingest` | 增量入库，body: `{"file_path": "..."}` |
| DELETE | `/docs` | 批量删除，body: `{"ids": ["..."]}` |
| GET  | `/stats` | 向量库统计 |

## 项目结构

```
.
├── rag/                       主包
│   ├── __init__.py
│   ├── config.py              配置（环境变量 + 默认值）
│   ├── exceptions.py          统一异常层级
│   ├── logging_utils.py       日志工具
│   ├── clients.py             LLM / Embedding 客户端（惰性 + 重试）
│   ├── loaders.py             TXT / PDF 加载
│   ├── splitters.py           分块：递归 / 语义 / 标题感知
│   ├── store.py               向量库抽象 + Chroma / Milvus / PGVector
│   ├── pipeline.py            ★ RAG 主链路编排（业务唯一入口 + 查询增强）
│   ├── evaluation.py          RAG 评测：recall@k / precision@k / faithfulness
│   ├── agent.py               Agentic-RAG：路由判断 + 执行
│   ├── api.py                 FastAPI 后端
│   ├── webui.py               Gradio 演示
│   └── retrieval/            检索子模块
│       ├── bm25.py            BM25 稀疏
│       ├── vector.py          向量稠密
│       ├── hybrid.py          混合 + RRF
│       └── rerank.py          智谱 Rerank
├── cli.py                     Click 命令行入口
├── knowledge.txt              demo 知识库（可替换）
├── data/                      上传文档存放（gitignore）
├── chroma_db/                 向量库数据（gitignore，本地生成）
├── .env.example               配置模板
├── .gitignore
├── pyproject.toml
├── requirements.txt           pip 兜底（由 uv export 生成）
└── README.md
```

## 检索流程

```
用户提问
   │
   ▼
[Agentic 路由] (可选)  need_retrieval?
   │  ├─ False → 直 LLM 对话（闲聊 / 常识 / 问候）
   │  └─ True ↓
   ▼
[查询增强] (enhance=True 时启用)
   ├─ rewrite_query: LLM 改写为检索友好形式
   └─ decompose_query: 拆子问题
   │
   ▼
┌─────────────────────────────────────┐
│  混合召回（每个子查询各跑一次合并）   │
│  ├─ 向量 dense (Chroma + Embedding) │
│  └─ BM25 sparse (rank_bm25)        │
│           │                         │
│           ▼                         │
│     RRF 融合 (k=60)                │
└─────────────────────────────────────┘
   │
   ▼
[智谱 Rerank 精排]  →  保留 top_n=3
   │
   ▼
[拼装上下文 + Prompt 模板]
   │
   ▼
[LLM 生成] → 返回 answer + contexts

[可选 RAG 评测] faithfulness(answer, contexts) → 抑制幻觉评分
```

## 配置项

所有配置走环境变量，见 `.env.example`。常用项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 必填 | LLM/Embedding 鉴权 |
| `OPENAI_BASE_URL` | 必填 | OpenAI 兼容接口地址 |
| `OPENAI_MODEL_ID` | 必填 | 对话模型 ID |
| `EMBEDDING_MODEL_ID` | `embedding-3` | 向量化模型 ID |
| `ZHIPUAI_API_KEY` | 空 | 智谱 Rerank key，留空则跳过 Rerank 降级 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `API_PORT` | `8000` | FastAPI 端口 |
| `GRADIO_PORT` | `7860` | Gradio 端口 |

检索参数（`rag/config.py` 中可调）：`chunk_size=500`、`chunk_overlap=80`、`retrieve_top_k=20`（每路粗召回）、`rerank_candidates=10`（RRF 融合后进 Rerank）、`rerank_top_n=3`（最终送 LLM）、`rrf_k=60`。

## 设计取舍

- **不用 LangChain**：原生 openai SDK + chromadb SDK + 自实现 RRF / 分块 / BM25 包装，避免过度抽象与依赖地狱。
- **RagPipeline 作为业务唯一入口**：api / webui / cli 都是薄壳层，业务漂移风险低。
- **重试只覆盖可重试错误**：超时 / 连接 / 5xx / 限流才重试；4xx 鉴权 / 参数错误立即抛出，避免付费接口烧钱。
- **Embedding 维度检测**：换模型后旧 `chroma_db` 维度不匹配时主动报错提示重建。
- **Rerank 分批**：智谱 Rerank 单次 documents 数量有上限，超出自动分批调用。
- **进阶能力按需开启**：查询增强（`ask(enhance=True)`）与 Agentic 路由（`rag.agent.get_agent()`）默认关闭，闲聊/简单问题无需走全流程；Milvus / PGVector 后端通过可选依赖 `uv sync --extra milvus` / `--extra pgvector` 启用，避免强制依赖。

## License

见 [LICENSE](./LICENSE)。

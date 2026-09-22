"""RAG 评测脚本：召回率 + 忠实度。

用法: python eval.py
金标策略：知识库中现有文档 id 全部视为相关（语义单一），用于验证召回链路。
"""
from rag.clients import reset_clients
from rag.evaluation import faithfulness, precision_at_k, recall_at_k
from rag.pipeline import get_pipeline, reset_pipeline

reset_clients()
reset_pipeline()

p = get_pipeline()


def collect_ids(result: dict) -> list[str]:
    return [c["meta"].get("id") for c in result["contexts"] if c["meta"].get("id")]


# ============ 准备金标 ============
# 用一个探针问题拿到知识库现有所有 doc id（语义单一的金标集）
probe = p.ask("RAG 是什么")
gold_ids = list({i for i in collect_ids(probe) if i})
print(f"知识库现有 doc id（金标）: {gold_ids}")
print(f"向量库片段数: {p.store.count()}")

# ============ 1. 召回率测试 ============
print("\n=== 召回率测试 ===")
questions = [
    "RAG 是什么",
    "RAG 的工作流程是什么",
    "本项目用什么向量数据库",
    "嵌入模型是什么",
    "智谱模型用哪个",
    "文档加载与切分是哪一步",
]
recalls: list[float] = []
precisions: list[float] = []
for q in questions:
    r = p.ask(q)
    retrieved = collect_ids(r)
    rec = recall_at_k(retrieved, gold_ids, k=3)
    prec = precision_at_k(retrieved, gold_ids, k=3)
    recalls.append(rec)
    precisions.append(prec)
    print(f"  Q: {q}")
    print(f"    retrieved_ids={retrieved}")
    print(f"    recall@3={rec:.2f}  precision@3={prec:.2f}")

print(f"\n[召回率] 平均 recall@3 = {sum(recalls) / len(recalls):.2f}")
print(f"[精确率] 平均 precision@3 = {sum(precisions) / len(precisions):.2f}")

# ============ 2. 忠实度测试 ============
print("\n=== 忠实度测试（高分组）===")
test_q = "RAG 的工作流程分为哪几步？"
r = p.ask(test_q)
ctx = [c["content"] for c in r["contexts"]]
print(f"  Q: {test_q}")
print(f"  Answer: {r['answer']}")
f_good = faithfulness(r["answer"], ctx)
print(
    f"  Faithfulness: {f_good['faithfulness']:.2f} "
    f"(total={f_good['total']}, supported={f_good['supported']})"
)
for d in f_good["details"]:
    print(f"    [{d.get('supported', '?')}] {d.get('statement', '')}")

print("\n=== 忠实度测试（低分对照）===")
bad_answer = (
    "RAG 全称是 Random Access Generator，由 Google 在 2010 年发布。"
    "本项目使用 OpenAI 的 GPT-5 模型和 Pinecone 向量数据库，"
    "工作流程只需一步：直接调用 GPT-5 生成答案。"
)
print(f"  Bad Answer: {bad_answer}")
f_bad = faithfulness(bad_answer, ctx)
print(
    f"  Faithfulness: {f_bad['faithfulness']:.2f} "
    f"(total={f_bad['total']}, supported={f_bad['supported']})"
)
for d in f_bad["details"]:
    print(f"    [{d.get('supported', '?')}] {d.get('statement', '')}")

print("\n=== 评测完成 ===")

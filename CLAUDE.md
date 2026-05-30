# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

"智扫通机器人智能客服" — 基于 LangChain/LangGraph 的 RAG 智能体应用，面向扫地/扫拖机器人领域。
用户通过 Streamlit 聊天界面提问，Agent 按照 ReAct 模式（思考→行动→观察→再思考）调用工具，
结合向量+BM25混合检索知识库和外部数据，生成专业回答。

## 运行方式

```bash
# 启动 Streamlit 前端（主入口）
streamlit run app.py

# 直接测试 Agent（命令行）
python agent/react_agent.py

# 测试 RAG 检索服务（含混合检索）
python rag/rag_service.py

# 测试 BM25 混合搜索（含 RRF 融合验证）
python rag/bm25_search.py
```

没有 lint/格式化配置，项目无 `requirements.txt` 或 `pyproject.toml`，依赖需手动安装。

## 技术栈与关键依赖

- **大模型**: 阿里通义千问 Qwen3-Max（`ChatTongyi`），通过 LangChain 社区集成调用
- **嵌入模型**: DashScope `text-embedding-v4`（`DashScopeEmbeddings`）
- **Agent 框架**: `langchain.agents.create_agent` 构建 ReAct Agent，`agent.stream` 流式输出
- **向量库**: Chroma（本地持久化到 `chroma_db/`）
- **前端**: Streamlit（`st.chat_input` + `st.chat_message` + `write_stream`）
- **混合检索**: 向量语义检索（Chroma） + BM25 关键词检索（jieba 中文分词），通过 RRF（Reciprocal Rank Fusion）融合排序，取 Top-5
- **分词**: jieba（作为 BM25 的 `preprocess_func` 处理中文分词）

## 架构分层

```
app.py                     ← Streamlit UI 入口，管理 session_state，调用 ReactAgent.execute_stream()
  └─ agent/react_agent.py  ← Agent 大脑：组装模型+工具+中间件，暴露 execute_stream() 流式接口
       ├─ model/factory.py       ← 模型工厂（抽象工厂模式）：ChatModelFactory / EmbeddingModelFactory
       ├─ agent/tools/agent_tools.py  ← 7 个 @tool 工具（见下方工具清单）
       ├─ agent/tools/middleware.py   ← 3 层中间件（见下方中间件说明）
       └─ rag/rag_service.py    ← RAG 总结服务：混合检索 → 拼接上下文 → LLM 生成回答
            ├─ rag/vector_store.py   ← Chroma 向量库：文档加载、分块、MD5 去重、向量检索
            └─ rag/bm25_search.py   ← BM25 检索 + RRF 融合：双路召回 → 加权融合排序
```

## 三层中间件体系（agent/tools/middleware.py）

执行顺序：每次模型调用前触发 → 模型思考 → 调用工具时触发 → 下次模型调用前重新评估提示词。

1. **`@wrap_tool_call` — `monitor_tool`**: 拦截所有工具调用，记录工具名和参数到日志；当工具名为 `fill_context_for_report` 时，在 `runtime.context` 中写入 `report=True` 标记
2. **`@before_model` — `log_before_model`**: 模型调用前打印当前消息数量和最新消息内容，用于调试追踪
3. **`@dynamic_prompt` — `report_prompt_switch`**: 每次生成提示词前读取 `runtime.context.report`，为 `True` 时切换到报告生成提示词，否则使用默认系统提示词

## 可用工具（agent/tools/agent_tools.py）

| 工具名 | 入参 | 用途 |
|---|---|---|
| `rag_summarize` | `query: str` | 向量+BM25混合检索知识库，LLM总结回答 |
| `get_weather` | `city: str` | 获取指定城市天气（Mock） |
| `get_user_location` | 无 | 随机返回用户城市 |
| `get_user_id` | 无 | 随机返回用户ID（1001-1010） |
| `get_current_month` | 无 | 随机返回月份（2025-01 ~ 2025-12） |
| `fetch_external_data` | `user_id, month` | 从 CSV 读取用户使用记录 |
| `fill_context_for_report` | 无 | **触发标记**：调用后中间件注入 `report=True`，后续自动切换为报告提示词 |

## 混合检索与 RRF 融合（rag/bm25_search.py）

当用户调用 `rag_summarize` 工具时，检索不走单一通道，而是**双路并行召回 → RRF 融合排序**：

```
用户 query
    │
    ├──→ 向量检索（Chroma semantic search）──→ 语义相关排序结果
    │
    └──→ BM25 检索（jieba 分词 + 关键词匹配）──→ 关键词相关排序结果
              │
              ▼
    RRF 融合（Reciprocal Rank Fusion）
    公式: score(d) = Σ 1/(rank_i(d) + m)
    其中 m=60（平滑常数，调节排名权重差距）
              │
              ▼
    融合排序 Top-K（k=5）→ 返回给 LLM 作为参考资料
```

### 核心实现细节

- **`preprocessing_func(text)`**: 传入 `BM25Retriever.from_documents()` 作为中文分词器，内部调用 `jieba.cut()` 将中文文本切词后再建倒排索引
- **文档 ID 映射**: 以 `page_content` 为桥梁，将向量召回和 BM25 召回的文档分别映射到 `VectorStoreService.documents` 中的全局索引，再对索引 ID 做 RRF 融合
- **容错降级**: 向量检索返回空时，退化为纯 BM25 排序；文档在全局索引中找不到时跳过

### BM25 模块独立测试

`bm25_search.py` 的 `__main__` 块包含 5 个自测场景，可直接运行验证：
1. RRF 融合算法数学逻辑验证（手算 score 对比）
2. BM25HybridSearch 初始化
3. 纯 BM25 检索（不融合）
4. 混合检索（Mock 向量检索器 + BM25 → RRF 融合，覆盖"小户型/宠物/故障/选购"四个查询）
5. 边界情况（空查询、无匹配查询、向量为空降级、空文档集）

## 关键数据流

**普通问答流程**:
用户提问 → `execute_stream()` 注入 `context={"report": False}` → Agent 判断需补充专业知识 → 调用 `rag_summarize` → `RagSummarizeService.retriever_docs()` 触发混合检索（向量+BM25 双路召回 → RRF 融合取 Top-5）→ 拼接上下文 → LLM 基于参考资料生成回答 → 流式返回

**报告生成流程**:
用户提问（"生成我的使用报告"）→ Agent 识别报告意图 → 调用 `get_user_id` → `get_current_month` → `fill_context_for_report`（此时中间件设置 `context.report=True`）→ `fetch_external_data` → 模型二次调用时 `report_prompt_switch` 检测到标记，切换到报告提示词 → 生成 Markdown 格式报告 → 流式返回

## 配置体系

所有配置集中在 `config/` 目录，通过 `utils/config_handler.py` 统一加载为模块级全局变量：

- **`config/rag.yml`**: 聊天模型名 `qwen3-max`、嵌入模型名 `text-embedding-v4`
- **`config/chroma.yml`**: 向量库集合名、持久化目录、检索Top-K、分块大小(200)/重叠(20)、分隔符、数据目录、MD5存储文件
- **`config/prompts.yml`**: 三个提示词文件的路径映射
- **`config/agent.yml`**: 外部数据 CSV 路径 `data/external/records.csv`

提示词文件在 `prompts/` 目录：
- `main_prompt.txt` — 默认系统提示词，定义 ReAct 思考准则、工具使用规则
- `rag_summarize.txt` — RAG 总结模板，含 `{input}` 和 `{context}` 占位符
- `report_prompt.txt` — 报告生成专用提示词，要求输出 Markdown 格式报告

## 知识库数据流

`data/` 目录下的 PDF/TXT 文件 → `VectorStoreService.load_document()` → MD5 去重（对比 `md5.text`）→ `RecursiveCharacterTextSplitter` 按中文标点分块 → 存入 Chroma（向量检索通道）→ **同时**分块文档追加到 `self.documents` 列表（供 `BM25HybridSearch` 复用，作为 BM25 全文检索通道的索引源）。

两个检索通道共享同一份文档分块，各自独立索引：
- Chroma 负责**语义向量索引**（embedding → 余弦相似度）
- BM25 负责**关键词倒排索引**（jieba 分词 → TF-IDF 权重）

`RagSummarizeService` 初始化时自动调用 `load_document()`，增量添加文件只需放入 `data/` 目录并清空 `md5.text` 中的对应 MD5 行（或删除 `md5.text` 全部重建）。

## 路径处理约定

所有路径通过 `utils/path_tool.py` 的 `get_abs_path()` 解析为相对于项目根目录的绝对路径。`get_project_root()` 基于 `path_tool.py` 所在位置向上两级（`utils/` → 项目根目录）。

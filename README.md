# AI_RAG_Agent

基于 LangChain/LangGraph 的 RAG + Agent 智能体应用，支持动态提示词切换、工具调用拦截、混合检索、SSE 流式对话。

> 扫地机器人智能客服场景 | 机械转行 AI 项目 | [GitHub](https://github.com/nunuli9527/AI_RAG_Agent)

---

## 项目结构

```
AI_RAG_Agent/
├── agent/                     # Agent 核心
│   ├── react_agent.py         # ReAct Agent 封装，流式对话入口
│   └── tools/
│       ├── agent_tools.py     # 工具集：RAG检索、天气查询、用户信息、报告标记
│       └── middleware.py       # 三层中间件：工具拦截 / 模型日志 / 动态提示词
├── rag/                       # RAG 检索链
│   ├── rag_service.py         # RAG 总结服务（检索 + LLM 生成）
│   ├── vector_store.py        # Chroma 向量库、MD5 去重
│   ├── bm25_RRF.py            # BM25 关键词检索 + RRF 融合
│   └── reranker.py            # Reranker 精排
├── model/
│   └── factory.py             # 模型工厂：聊天模型 + 嵌入模型
├── config/                    # YAML 配置
│   ├── agent.yml              # Agent 配置（工具 URL、外部数据路径）
│   ├── chroma.yml             # Chroma 向量库配置
│   ├── prompts.yml            # 提示词文件路径配置
│   └── rag.yml                # RAG 模型配置
├── prompts/                   # 提示词模板
│   ├── main_prompt.txt        # 默认系统提示词
│   ├── rag_summarize.txt      # RAG 总结提示词
│   └── report_prompt.txt      # 报告生成提示词
├── utils/                     # 工具函数
│   ├── config_handler.py      # YAML 配置加载
│   ├── prompt_loader.py       # 提示词加载（异常安全）
│   ├── logger_handler.py      # 日志封装
│   ├── file_handler.py        # 文件处理
│   └── path_tool.py           # 路径工具
├── tests/                     # pytest 单元测试（33 个）
│   ├── conftest.py            # 全局 fixtures
│   ├── test_prompt_loader.py  # 提示词加载测试
│   ├── test_config.py         # 配置加载测试
│   ├── test_api.py            # FastAPI SSE 接口测试
│   └── test_tools.py          # 工具函数测试
├── app.py                     # Streamlit 聊天界面入口
├── api.py                     # FastAPI SSE 流式接口
├── data/                      # 知识库文档 & 外部数据
├── chroma_db/                 # 向量库持久化目录
├── logs/                      # 日志文件
└── README.md
```

---

## 核心功能

### 1. 三层中间件体系

| 中间件 | 装饰器 | 作用 |
|---|---|---|
| 工具拦截 | `@wrap_tool_call` | 全局监听工具调用，日志记录，注入运行时上下文标记 |
| 模型日志 | `@before_model` | 模型调用前打印对话状态，便于调试 |
| 动态提示词 | `@dynamic_prompt` | 根据 `runtime.context` 自动切换系统提示词 |

**流程**：用户说"生成报告" → `fill_context_for_report` 工具写入 `context.report=True` → 下轮模型调用时 `@dynamic_prompt` 自动加载报告生成提示词。

### 2. RAG 混合检索链

```
用户提问 → 向量检索(Dense) + BM25(Sparse) → RRF 融合粗排 → Reranker 精排 → Top-5 文档 → LLM 总结
```

### 3. 双入口架构

| 入口 | 命令 | 用途 |
|---|---|---|
| Streamlit | `streamlit run app.py` | 内部演示、聊天调试 |
| FastAPI SSE | `uvicorn api:app --port 8000` | 外部系统调用、流式接口 |

### 4. 真实 API 对接

- **天气**：对接 [wttr.in](https://wttr.in) 实时天气 API（无需注册）
- **大模型**：阿里百炼 DashScope（ChatTongyi + DashScopeEmbeddings）

---

## 快速开始

### 环境要求

- Python 3.12+
- 阿里百炼 API Key（`DASHSCOPE_API_KEY` 环境变量）

### 安装

```bash
pip install langchain langchain-community langgraph chromadb streamlit fastapi uvicorn pytest jieba pyyaml dashscope
```

### 配置

1. 按需修改 `config/*.yml`
2. 设置环境变量 `DASHSCOPE_API_KEY`

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "你的key"
```

3. （如果机器有代理）API 调用会自动绕过，无需手动配置

### 启动

```bash
# 界面模式
streamlit run app.py

# API 模式（另开终端）
uvicorn api:app --reload --port 8000
```

浏览器打开：
- 界面：`http://localhost:8501`
- API 文档：`http://localhost:8000/docs`

### 测试

```bash
# 单元测试（不含集成）
pytest tests/ -v -m "not integration"

# 集成测试（会调 LLM，较慢）
pytest tests/ -v -m "integration"
```

---

## API 接口

### `GET /chat/stream?query=你好`

SSE 流式对话，响应格式：

```
data: 回答内容片段
data: 下一段内容
event: error
data: 错误信息（仅在异常时）
event: done
data: [DONE]
```

示例：

```bash
curl "http://localhost:8000/chat/stream?query=扫地机器人怎么保养"
```

---

## 技术栈

- **框架**：LangChain / LangGraph（Agent 编排）
- **检索**：Chroma 向量库 + BM25 + RRF 融合 + Reranker 重排序
- **模型**：阿里百炼 DashScope（qwen-turbo / text-embedding-v3）
- **前端**：Streamlit（聊天界面）
- **接口**：FastAPI SSE（流式 HTTP）
- **测试**：pytest（33 个单元测试）

---

## 面试要点

> 1. **中间件机制**：基于 LangGraph Runtime 的三层中间件，`@wrap_tool_call` + `@dynamic_prompt` 实现报告场景下的提示词自动切换，单次会话上下文隔离。
> 2. **混合检索**：Dense + Sparse 双路检索，RRF 融合取 Top-K 候选，Reranker 精排取 Top-5，比单纯向量检索提升召回质量。
> 3. **工程实践**：MD5 去重避免重复入库、YAML 配置分离、pytest 测试覆盖、SSE 流式输出 + 异常兜底。
> 4. **双入口**：Streamlit 内部演示 + FastAPI SSE 对外集成，共享同一套 Agent 核心代码。

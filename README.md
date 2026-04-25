# OpenDify Lite

> 把 Dify 应用通过一个轻量代理暴露为 OpenAI 兼容的 ChatCompletion API，让你用熟悉的 SDK/生态调用 Dify 的能力。
>
> 特别优化：为弱模型设计的工具调用强制约束系统，从 "礼貌提示" 到 "文言文劝谏" 六个档位可选。

## 功能特性

### 🎯 核心能力

- **OpenAI 兼容接口** — 标准 `/v1/chat/completions`，直接替换 `base_url` 即可用
- **多 Dify 应用映射** — 一个代理后端可绑定多个 Dify 应用，通过 `model` 参数路由
- **完整工具调用支持** — `tools` / `tool_choice` 完整支持，确保弱模型也能按格式输出
- **流式响应** — 支持 SSE 流式输出，可带 `stream_options` 返回 usage
- **会话管理** — 自动追踪 Dify `conversation_id`，支持长对话记忆与 usage 累积
- **多种提示词方言** — Generic / Claude / OpenAI 三种渲染模式，适配不同模型

### 🛡️ 工具调用约束系统（专为弱模型设计）

| 级别 | 名称 | 适用场景 | 特点 |
|------|------|----------|------|
| 0 | off | 强模型（GPT-4 / Claude） | 极简提示，不加任何警告 |
| 1 | polite | 中等模型 | 结尾加硬约束列表 |
| 2 | assertive | 弱模型（默认） | 强命令式 + 前后夹击 + role=tool 追加提醒 |
| 3 | aggressive | 非常弱的模型 | + 后果警告 "输出将被丢弃" |
| 4 | nuclear | 救命稻草 | + 规则重复 + "最后一次机会" |
| 5 | savage | 本地调试 | 脏话发泄模式（日志自动脱敏） |
| 6 | classical | 文言文爱好者 | 劝谏体，非辱骂，实验档位 |

### 📦 优化特性

- **令牌防串扰** — 每轮对话生成唯一 token 嵌入标签，防止旧对话干扰
- **提示词裁剪** — 可配置系统提示词截断、仅保留最近 N 条消息
- **工具描述摘要** — 实验功能：后台异步生成极简工具描述缓存，缩短上下文
- **流量日志** — 可选记录完整请求/响应流程，便于调试
- **连接池** — 配置化 HTTP 连接池，提升并发性能

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

**最小配置示例：**

```env
# 认证
VALID_API_KEYS=sk-your-secret-key
AUTH_MODE=required

# Dify 配置
DIFY_API_KEY=app-xxxxxxxxxxxxxxxx
DIFY_MODEL_NAME=my-dify-app
```

### 启动服务

```bash
python app.py
```

服务默认监听：`http://127.0.0.1:8000`

### 使用示例

#### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-secret-key",
    base_url="http://127.0.0.1:8000/v1",
)

resp = client.chat.completions.create(
    model="my-dify-app",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

#### 带工具调用示例

```python
resp = client.chat.completions.create(
    model="my-dify-app",
    messages=[{"role": "user", "content": "读取当前目录"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "列出目录内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "目录路径"}
                    },
                    "required": ["path"]
                }
            }
        }
    ],
)
print(resp.choices[0].message.tool_calls)
```

#### 流式输出

```python
stream = client.chat.completions.create(
    model="my-dify-app",
    messages=[{"role": "user", "content": "写一篇关于AI的短文"}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-dify-app",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 配置详解

### 服务器配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SERVER_HOST` | `127.0.0.1` | 监听地址 |
| `SERVER_PORT` | `8000` | 监听端口 |
| `LOG_LEVEL` | `WARNING` | 日志级别：DEBUG \| INFO \| WARNING \| ERROR |

### 认证配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AUTH_MODE` | `required` | `required` 校验 API Key，`disabled` 不校验（仅内网） |
| `VALID_API_KEYS` | - | 代理层 API Key，多个用逗号分隔 |

### Dify 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DIFY_API_BASE` | `https://api.dify.ai/v1` | Dify API 地址 |
| `DIFY_API_KEY` | - | 单个 Dify 应用的 API Key |
| `DIFY_MODEL_NAME` | `dify-model` | 单应用模式下的 model 名称 |
| `DIFY_MODEL_MAP` | - | 多应用映射：`模型名:app-key,模型名2:app-key2` |
| `TIMEOUT` | `120` | Dify 请求超时（秒） |
| `POOL_SIZE` | `50` | HTTP 连接池大小 |

### 会话模式

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONVERSATION_MODE` | `auto` | `none` 每次新对话，`auto` 自动复用 Dify 会话 |

### 弱模型优化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STRIP_SYSTEM_AFTER_FIRST` | `false` | 首次之后是否省略系统提示词（利用 Dify 会话记忆） |
| `SYSTEM_PROMPT_MAX_LENGTH` | `0` | 系统提示词最大长度，0=不截断 |
| `SIMPLIFIED_TOOL_DEFS` | `true` | 使用简化的工具定义格式（推荐弱模型开启） |
| `TOOL_DESC_MAX_LENGTH` | `120` | 工具描述最大长度 |
| `ONLY_RECENT_MESSAGES` | `0` | 仅发送最近 N 条消息，0=全部 |

### 工具调用约束

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOOL_CALL_STRICTNESS` | `2` | 0~6 级强度（见上文表格） |
| `USE_TOOL_TOKEN` | 自动 | 是否使用 token 标签防串扰（默认 strictness>=1 开启） |
| `AGGRESSIVE_TOOL_RECOVERY` | 自动 | 是否启用正则兜底抢救（默认 strictness>=3 开启） |
| `SAVAGE_LOG_REDACT` | `true` | Level 5 脏话模式下是否脱敏日志 |

### 工具描述摘要缓存（实验）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOOL_DESC_DIGEST_ENABLED` | `false` | 是否开启摘要缓存 |
| `TOOL_DESC_DIGEST_DIR` | `.cache/prompt_digest` | 缓存目录 |
| `TOOL_DESC_DIGEST_MAX_CHARS` | `40` | 摘要最大字符数 |

### 流量日志

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `REQUEST_LOG_ENABLED` | `false` | 是否开启请求/响应日志 |
| `REQUEST_LOG_DIR` | `logs` | 日志目录 |
| `REQUEST_LOG_MAX_SIZE` | `52428800` | 单日志文件最大字节（50MB） |
| `REQUEST_LOG_BACKUP_COUNT` | `5` | 保留历史日志数量 |
| `REQUEST_LOG_MAX_BODY` | `0` | 单条记录 body 最大字符，0=不截断 |

## 提示词方言

OpenDify Lite 支持三种提示词渲染方言，通过 `PROMPT_DIALECT` 配置：

### Generic（默认）

使用 `[Role]: content` 的纯文本格式 + `<tool-calls>[JSON]</tool-calls>` 标签。

```
[System]: 你是一个有用的助手

[User]: 你好

[Assistant]: 你好！有什么我可以帮你的？

# 可用工具
...
```

### Claude

使用 Claude 原生的 `<system>` / `<user>` / `<assistant>` 标签格式。

### OpenAI

使用 ChatML 风格 `<|im_start|>` / `<|im_end|>` 标签，适配 Hermes / Qwen 等模型。

## API 端点

### `POST /v1/chat/completions`

OpenAI 兼容的聊天补全接口。

**请求参数：**
- `model` - 模型名称（对应 DIFY_MODEL_MAP 中的键）
- `messages` - 消息列表
- `tools` - 工具定义列表（可选）
- `tool_choice` - 工具选择策略（可选）
- `stream` - 是否流式输出（可选）
- `stream_options` - 流式选项（可选）
- `conversation_id` - 指定 Dify 会话 ID（可选）

**响应头：**
- `X-Dify-Conversation-Id` - Dify 会话 ID，可用于后续请求复用

### `GET /v1/models`

列出可用模型。

### `GET /v1/models/{model_id}`

获取指定模型信息。

## 会话管理

OpenDify Lite 按 `(model, system_prompt)` 维度管理会话：

1. 首次请求：创建新会话，获得 Dify `conversation_id`
2. 后续请求：自动复用会话，累积 usage
3. 新任务检测：当消息数从 >1 回落至 1 时，自动重置会话
4. 手动指定：可通过 `X-Dify-Conversation-Id` header 或 `conversation_id` body 字段强制指定

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     客户端 (OpenAI SDK)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     FastAPI 入口层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Auth 验证   │  │  流量日志    │  │  异常处理        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     请求转换层                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  会话管理 (SessionStore)                              │  │
│  │  - 按 (model, system_prompt) 分组                     │  │
│  │  - 追踪 conversation_id / token / 累计 usage         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  方言渲染 (Generic / Claude / OpenAI)                 │  │
│  │  - 消息格式化                                         │  │
│  │  - 工具提示词注入（含强度约束）                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     Dify 上游调用                            │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │  流式 (SSE)      │  │  非流式 (Blocking)            │    │
│  └──────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     响应转换层                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  工具调用提取 (XML 标签解析 + 正则兜底)               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  OpenAI 格式封装 (流式/非流式)                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     返回客户端                               │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
OpenDify Lite/
├── app.py                      # 入口启动脚本
├── requirements.txt            # 依赖
├── .env.example               # 配置示例
├── README.md                  # 本文档
└── opendify/
    ├── __init__.py
    ├── config.py              # 配置加载、全局常量
    ├── server.py              # FastAPI 应用、路由
    ├── auth.py                # API Key 验证
    ├── sessions.py            # 会话管理
    ├── transforms.py          # OpenAI → Dify 请求转换
    ├── responses.py           # Dify → OpenAI 响应转换
    ├── streaming.py           # 流式处理
    ├── tool_prompt.py         # 工具提示词生成（含 0~6 级约束）
    ├── tool_calls.py          # 工具调用提取
    ├── tool_digest.py         # 工具描述摘要缓存（实验）
    ├── traffic_log.py         # 流量日志
    ├── http_client.py         # HTTP 客户端
    ├── errors.py              # 错误定义
    ├── utils.py               # 工具函数
    └── dialects/              # 提示词方言
        ├── __init__.py
        ├── generic.py         # Generic 方言
        ├── claude.py          # Claude 方言
        └── openai.py          # OpenAI 方言
```

## 测试

```bash
python test.py
```

测试包含单元测试和集成测试。集成测试需要先启动服务，默认从 `.env` 读取配置。

## 兼容性说明

- 仅支持 `n=1`，其他值返回 400
- 支持 OpenAI Chat Completions 核心参数
- 暂不支持 Embeddings / Fine-tuning 等其他接口

## 许可证

MIT License

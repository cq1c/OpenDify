# OpenDify

把 Dify 应用通过一个轻量代理暴露为多种兼容接口，方便直接用现有 SDK/生态调用。

## 支持的接口

- `POST /v1/chat/completions`
- `POST /anthropic/v1/chat/completions`（可选：OpenAI Chat Completions -> Anthropic Messages；需配置 `ANTHROPIC_API_KEY`）
- `POST /v1/responses`
- `POST /v1/messages`（Claude / Anthropic Messages -> Dify）
- `POST /v1/messages/count_tokens`（Claude / Anthropic Token Count）
- `POST /anthropic/v1/messages`（2anthropic：Claude Code / Anthropic SDK -> OpenAI Chat Completions 上游；未配置上游则 fallback 到 Dify）
- `POST /anthropic/v1/messages/count_tokens`
- `GET /anthropic/v1/models`
- `GET /anthropic/v1/models/{model_id}`
- `GET /v1/models`
- `GET /v1/models/{model_id}`

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
python app.py
```

默认监听：`http://127.0.0.1:8000`

## 配置（.env）

| 变量 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `AUTH_MODE` | 否 | `required` | `required` 校验 `Authorization`；`disabled` 不校验（仅建议内网） |
| `VALID_API_KEYS` | 否* | - | 代理层 API Key（逗号分隔）；`AUTH_MODE=required` 时必填 |
| `DIFY_API_BASE` | 否 | `https://api.dify.ai/v1` | Dify API Base（会自动去掉末尾 `/`） |
| `DIFY_API_KEYS` | 是 | - | Dify 应用 API Key（逗号分隔） |
| `DIFY_SSL_VERIFY` | 否 | `true` | TLS 校验（自签证书可设为 `false`） |
| `CONVERSATION_MEMORY_MODE` | 否 | `1` | `1` 全量 messages（最兼容）；`2` 提供 conversation_id 时仅发送增量（长对话更快） |
| `TIMEOUT` | 否 | `30.0` | 访问 Dify 超时（秒） |
| `SERVER_HOST` | 否 | `127.0.0.1` | 监听地址 |
| `SERVER_PORT` | 否 | `8000` | 监听端口 |
| `WORKERS` | 否 | `1` | Uvicorn workers |
| `LOG_LEVEL` | 否 | `WARNING` | `DEBUG/INFO/WARNING/ERROR` |
| `ANTHROPIC_API_BASE` | 否 | `https://api.anthropic.com` | Anthropic API Base |
| `ANTHROPIC_API_KEY` | 否* | - | Anthropic API Key（启用 `/anthropic/*` 时必填） |
| `ANTHROPIC_MODEL` | 否* | - | Anthropic 上游模型（不填则使用请求里的 `model`） |
| `ANTHROPIC_VERSION` | 否 | `2023-06-01` | Anthropic `anthropic-version` header |
| `ANTHROPIC_MAX_TOKENS` | 否 | `1024` | 未传 `max_tokens` 时的默认值 |
| `ANTHROPIC_SSL_VERIFY` | 否 | `true` | TLS 校验（自签证书可设为 `false`） |
| `UPSTREAM_OPENAI_BASE_URL` | 否* | - | 2anthropic 上游 OpenAI ChatCompletions（可为 base 或完整 `/v1/chat/completions`） |
| `UPSTREAM_OPENAI_API_KEY` | 否* | - | 2anthropic 上游 API Key（Authorization Bearer） |
| `UPSTREAM_OPENAI_MODEL` | 否 | - | 2anthropic 上游默认模型（不填则使用请求里的 `model`） |
| `UPSTREAM_OPENAI_MODEL_MAP` | 否 | - | 2anthropic model 映射（`in:out`，逗号分隔，优先级高于 `UPSTREAM_OPENAI_MODEL`） |
| `UPSTREAM_OPENAI_MAX_TOKENS` | 否 | `1024` | 2anthropic 未传 `max_tokens` 时默认值 |
| `UPSTREAM_OPENAI_SSL_VERIFY` | 否 | `true` | 2anthropic TLS 校验（自签证书可设为 `false`） |

兼容旧变量名（2anthropic 上游）：`base_url` / `api_key` / `model`（等价于 `UPSTREAM_OPENAI_BASE_URL` / `UPSTREAM_OPENAI_API_KEY` / `UPSTREAM_OPENAI_MODEL`）。

## 模型映射

启动时会用每个 `DIFY_API_KEY` 调用 Dify `/info` 获取应用名，并在 `/v1/models` 中用“应用名”作为 `model` 返回。

## 使用示例

### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-abc123" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Your-Dify-App-Name\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

### OpenAI Python SDK（Chat Completions）

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-abc123",
    base_url="http://127.0.0.1:8000/v1",
)

resp = client.chat.completions.create(
    model="Your-Dify-App-Name",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

### OpenAI Python SDK（Anthropic 后端 / Chat Completions）

注：该接口需配置 `.env` 的 `ANTHROPIC_API_KEY`；OpenAI SDK 的 `base_url` 需要包含 `/v1`；如果是 Claude Code/Anthropic SDK，请使用 `http://127.0.0.1:8000/anthropic` 作为 base（SDK 会自动拼 `/v1/...`）。

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-abc123",
    base_url="http://127.0.0.1:8000/anthropic/v1",
)

resp = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",  # 也可在 .env 里通过 ANTHROPIC_MODEL 固定上游模型
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

### OpenAI Python SDK（Responses）

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-abc123",
    base_url="http://127.0.0.1:8000/v1",
)

resp = client.responses.create(
    model="Your-Dify-App-Name",
    input="你好",
)
print(resp.output[0].content[0].text)
```

### Claude / Anthropic Messages（curl）

```bash
curl http://127.0.0.1:8000/anthropic/v1/messages \
  -H "X-API-Key: sk-abc123" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"claude-opus-4-5\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

注：也可直接用 `POST /v1/messages`（Dify 后端）；鉴权也可使用 `Authorization: Bearer sk-abc123`（与 OpenAI 接口一致）。

### Claude Code / Anthropic SDK（推荐）

你需要配置两部分：

1) OpenDify 服务端（`.env`）：配置上游 OpenAI ChatCompletions（这是“被转换的接口”）

- `UPSTREAM_OPENAI_BASE_URL`（或旧变量 `base_url`）指向你的 OpenAI 兼容地址（例如 `https://127.0.0.1:8080/v1/chat/completions`）
- `UPSTREAM_OPENAI_API_KEY`（或旧变量 `api_key`）为上游 key
- `UPSTREAM_OPENAI_MODEL`（或旧变量 `model`）为上游默认模型（可选）

2) Claude Code / Anthropic SDK 客户端：把 base_url 指向 OpenDify（这是“转换后的 Anthropic 接口”）

- base_url：`http://127.0.0.1:8000/anthropic`（不要带 `/v1`）
- api_key：使用 OpenDify 的 `VALID_API_KEYS` 之一

说明：`ANTHROPIC_API_BASE` 是 OpenDify 用来访问“真实 Anthropic 上游（/v1/messages）”的配置，只影响 `/anthropic/v1/chat/completions`；它不能指向 OpenAI 的 `/v1/chat/completions`。
要求：你的上游 OpenAI 兼容接口需要支持 `tools/tool_calls`（函数调用）；否则 Claude Code 会因无法执行工具调用而报错（常见提示：`Improperly formed request`）。

## Tool calls（tools/tool_calls）

- 请求中可带 `tools` / `tool_choice`（OpenAI 标准参数）。
- 响应会尽量输出标准 `tool_calls`，并保证 `function.arguments` 为 JSON 字符串。
- 注意：工具函数不会在 OpenDify/模型侧自动执行；需要由你的客户端执行工具并把结果以 `role="tool"` 消息回传给 `/v1/chat/completions`。

## 流式（SSE）

- 传 `stream=true` 返回 `text/event-stream`，格式为 OpenAI 的 `chat.completion.chunk`。
- 可选支持 `stream_options: {"include_usage": true}`（若 Dify 结束事件提供 usage，则透传；否则返回 0）。
- `/v1/responses` 流式会输出带 `type` + `sequence_number` 的事件，并以 `data: [DONE]` 结束。
- `/v1/messages` 流式为 Anthropic/Claude 事件格式（`event:` + `data:`）。

## conversation_id（可选）

为了在 Dify 侧复用对话上下文：

- 非流式响应会在 header 中返回 `X-Dify-Conversation-Id`（若 Dify 返回了 conversation_id）。
- 下次请求可通过 header `X-Dify-Conversation-Id` 或 body 字段 `conversation_id` 传回；并可配合 `CONVERSATION_MEMORY_MODE=2` 提升长对话性能。

## 测试

```bash
python test.py
```

`test.py`：包含 Unit + Integration；Integration 需先启动服务（`python app.py`），并默认从 `.env` 读取配置（也可用 `TEST_SERVER_ORIGIN/TEST_PROXY_API_KEY/TEST_DIFY_MODEL/TEST_CLAUDE_MODEL` 覆盖）。

## 兼容性说明（当前实现）

- 仅支持 `n=1`（其它会返回 400）。
- `/v1/responses` 与 `/v1/messages` 覆盖常用字段与流式输出（含 `sequence_number`/事件格式）；未实现 Embeddings 等其它接口。

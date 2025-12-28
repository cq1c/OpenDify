# OpenDify

把 Dify 应用通过一个轻量代理暴露为多种兼容接口，方便直接用现有 SDK/生态调用。

## 支持的接口

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`（Claude / Anthropic Messages）
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
curl http://127.0.0.1:8000/v1/messages \
  -H "X-API-Key: sk-abc123" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Your-Dify-App-Name\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

注：鉴权也可使用 `Authorization: Bearer sk-abc123`（与 OpenAI 接口一致）。

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

运行前请在 `test.py` 里修改：`BASE_URL` / `API_KEY` / `MODEL`。

## 兼容性说明（当前实现）

- 仅支持 `n=1`（其它会返回 400）。
- `/v1/responses` 与 `/v1/messages` 覆盖常用字段与流式输出（含 `sequence_number`/事件格式）；未实现 Embeddings 等其它接口。

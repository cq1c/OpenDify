# OpenDify

**🚀 将 Dify 转换为标准 OpenAI API 的高性能代理服务**

将 [Dify](https://dify.ai) 应用完美转换为 OpenAI 兼容接口，让你可以像调用 OpenAI API 一样使用 Dify！

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ✨ 核心特性

### 🎯 完全符合 OpenAI API 标准
- ✅ **标准接口**：`/v1/chat/completions` 和 `/v1/models`
- ✅ **无缝兼容**：直接使用官方 OpenAI SDK，无需修改代码
- ✅ **标准格式**：请求/响应格式 100% 符合 OpenAI 规范
- ✅ **错误处理**：标准的 OpenAI 错误格式

### 🛠️ 完整的 Tool Calls 支持
- ✅ **标准格式**：`tool_calls` 和 `tool` 消息完全符合规范
- ✅ **参数验证**：`arguments` 字段正确为 JSON 字符串
- ✅ **多轮对话**：支持工具调用的完整流程
- ✅ **自动提取**：智能从模型响应中提取工具调用

### 🌊 高性能流式响应
- ✅ **SSE 标准**：Server-Sent Events 格式完全标准
- ✅ **增量更新**：正确的 `delta` 增量格式
- ✅ **工具流式**：支持流式返回工具调用
- ✅ **性能优化**：HTTP/2、连接池、智能缓存

### 💬 灵活的对话管理
- ✅ **无状态设计**：客户端维护对话历史（OpenAI 标准）
- ✅ **对话上下文**：可选的 Dify conversation_id 支持
- ✅ **多模型管理**：自动映射 Dify 应用名到模型

## 🚀 快速开始

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/realnghon/OpenDify.git
cd OpenDify

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件：

```bash
# 代理服务的 API 密钥（可设置多个，逗号分隔）
VALID_API_KEYS=sk-your-secret-key-1,sk-your-secret-key-2

# Dify 应用的 API 密钥（可设置多个应用）
DIFY_API_KEYS=app-xxx,app-yyy,app-zzz

# Dify API 基础 URL
DIFY_API_BASE=https://api.dify.ai/v1

# 服务器配置（可选）
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
WORKERS=1

# 超时设置（可选）
TIMEOUT=30.0
```

### 启动服务

```bash
python app.py
```

服务将在 `http://127.0.0.1:8000` 启动。

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 必需 | 默认值 |
|--------|------|------|--------|
| `VALID_API_KEYS` | 代理服务的 API 密钥（逗号分隔） | ✅ | - |
| `DIFY_API_KEYS` | Dify 应用 API 密钥（逗号分隔） | ✅ | - |
| `DIFY_API_BASE` | Dify API 基础 URL | ✅ | - |
| `SERVER_HOST` | 服务器监听地址 | ❌ | `127.0.0.1` |
| `SERVER_PORT` | 服务器监听端口 | ❌ | `8000` |
| `WORKERS` | Worker 进程数 | ❌ | `1` |
| `TIMEOUT` | 请求超时时间（秒） | ❌ | `30.0` |

### 模型映射

代理服务会自动从 Dify 获取应用名称并映射为模型：

1. 启动时自动调用 Dify `/info` 接口
2. 获取每个应用的名称
3. 在 `/v1/models` 接口中列出
4. 使用应用名称作为 `model` 参数

示例：
- Dify 应用名：`ChatGPT-Assistant`
- 调用时使用：`model="ChatGPT-Assistant"`

## 📋 API 端点

### POST /v1/chat/completions

标准的 OpenAI Chat Completions 接口。

**请求参数：**
- `model` (string, 必需): Dify 应用名称
- `messages` (array, 必需): 对话消息数组
- `stream` (boolean, 可选): 是否流式响应，默认 `false`
- `tools` (array, 可选): 工具定义数组
- `tool_choice` (string|object, 可选): 工具选择策略
- `user` (string, 可选): 用户标识

**响应格式：**

非流式：
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "your-dify-app-name",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

流式（SSE）：
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"your-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"your-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"your-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### GET /v1/models

获取可用模型列表。

**响应格式：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "ChatGPT-Assistant",
      "object": "model",
      "created": 1234567890,
      "owned_by": "dify"
    }
  ]
}
```

## 🧪 测试

项目包含完整的测试套件：

```bash
# 运行标准兼容性测试
python test_openai_standard.py
```

测试覆盖：
- ✅ 基础对话
- ✅ 多轮对话
- ✅ 工具调用
- ✅ 多轮对话 + 工具调用
- ✅ 流式响应

## 🐛 故障排查

### 401 Unauthorized

**原因**：API 密钥无效

**解决**：
1. 检查 `.env` 中 `VALID_API_KEYS` 配置
2. 确认请求头 `Authorization: Bearer sk-xxx` 格式正确

### 404 Model not configured

**原因**：模型名称不存在

**解决**：
1. 访问 `http://127.0.0.1:8000/v1/models` 查看可用模型
2. 确认 `.env` 中 `DIFY_API_KEYS` 配置正确
3. 检查 Dify 应用是否正常运行

### 503 Service Unavailable

**原因**：Dify 后端模型过载

**解决**：
1. 稍后重试
2. 检查 Dify 后端状态
3. 查看 Dify 控制台的配额使用情况

### 工具调用未返回

**原因**：模型未理解工具定义

**解决**：
1. 确认工具定义格式正确
2. 提供更详细的 `description`
3. 在 `messages` 中明确要求使用工具

## 🚀 性能优化

代理服务已经过多重优化：

- ✅ **HTTP/2 支持**：提升并发性能
- ✅ **连接池**：100 个并发连接
- ✅ **智能缓存**：应用信息缓存 30 分钟
- ✅ **异步处理**：完全异步的请求处理
- ✅ **流式传输**：8KB 缓冲区，减少延迟

## 📊 架构设计

```
┌─────────────┐      HTTP      ┌──────────────┐      Dify API      ┌─────────┐
│   Client    │───────────────>│   OpenDify   │──────────────────>│  Dify   │
│ (OpenAI SDK)│                 │    Proxy     │                    │ Backend │
└─────────────┘<───────────────└──────────────┘<──────────────────└─────────┘
                  OpenAI                           Dify
                  Format                          Format

核心流程：
1. 接收 OpenAI 格式请求
2. 转换为 Dify 格式
3. 调用 Dify API
4. 转换响应为 OpenAI 格式
5. 返回给客户端
```

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件

---

⭐ **觉得有用？给个 Star 吧！** | Made with ❤️ by [realnghon](https://github.com/realnghon)

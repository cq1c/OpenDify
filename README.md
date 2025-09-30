# OpenDify

<div align="center">

**🚀 极致优化的高性能 Dify-to-OpenAI 代理服务 🚀**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 简介

**OpenDify** 是一个 **极致优化的高性能 FastAPI 代理服务**，可以将 [Dify](https://dify.ai) API 无缝转换为 **OpenAI 兼容格式 API**。

借助 OpenDify，您可以：
- ✅ 使用任何 OpenAI 风格的 SDK 或客户端直接访问 Dify 应用
- ✅ 无需修改现有基于 OpenAI 的代码或工作流
- ✅ 享受高性能的 HTTP/2 连接池和智能缓存
- ✅ 完整支持 OpenAI Function Calling（工具调用）
- ✅ 支持流式和非流式响应
- ✅ 灵活的对话记忆模式

---

## ✨ 核心特性

### 🎯 API 兼容性
- **完整 OpenAI API 兼容**：支持 `/v1/chat/completions` 和 `/v1/models` 端点
- **工具调用支持**：完整支持 OpenAI Function Calling，自动转换工具定义和响应
- **流式响应**：支持 Server-Sent Events (SSE) 流式输出
- **多模型管理**：自动映射 Dify 应用名称到 API 密钥


### 🧠 对话记忆模式
支持两种对话上下文管理方式：

**模式 1 - 历史消息模式**
- 将对话历史拼接到当前请求中
- 适合无状态场景或调试

**模式 2 - Conversation ID 模式**
- 使用零宽字符编码 Dify 的 `conversation_id`
- 实现真正的有状态对话
- 对终端用户完全透明

### 🛠️ 工具调用处理
- **多策略提取**：
  1. JSON 代码块提取（```json ... ```）
  2. 括号平衡的内联 JSON 提取
  3. 自然语言函数调用解析
- **自动清理**：从响应文本中移除工具 JSON 内容
- **流式工具支持**：增量解析工具调用，支持部分响应
- **原生 Dify 工具**：兼容 Dify 原生 `tool_calls` 事件

---

## 📂 项目结构

```
OpenDify/
├── app.py                 # 主应用文件（核心逻辑）
├── test.py                # 测试脚本
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量示例
├── .env                   # 环境变量配置（需自行创建）
└── README.md              # 本文件
```

---

## 🛠 快速开始

### 1. 环境要求
- Python 3.8+
- 支持的操作系统：Linux、macOS、Windows

### 2. 安装依赖
```bash
# 克隆项目
git clone https://github.com/realnghon/OpenDify.git
cd OpenDify

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境
在项目根目录创建 `.env` 文件：
```env
# 必需配置
VALID_API_KEYS=sk-your-proxy-key-1,sk-your-proxy-key-2
DIFY_API_KEYS=app-your-dify-key-1,app-your-dify-key-2
DIFY_API_BASE=https://api.dify.ai/v1

# 可选配置（以下为默认值）
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
CONVERSATION_MEMORY_MODE=1
WORKERS=1
```

### 4. 启动服务

**开发模式（推荐用于测试）**
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**生产模式（推荐用于部署）**
```bash
python app.py
```

### 5. 测试服务
```bash
python test.py
```

---

## 🔧 配置项说明

| 环境变量               | 说明                                                                                                                   | 必需 | 默认值       | 示例值 |
|------------------------|------------------------------------------------------------------------------------------------------------------------|------|--------------|--------|
| `VALID_API_KEYS`       | 允许访问代理的 API Key 列表（逗号分隔）                                                                                             | ✅   | `-`          | `sk-abc123,sk-def456` |
| `DIFY_API_KEYS`        | Dify 应用的 API 密钥列表（逗号分隔，支持多个应用）                                                                                                | ✅   | `-`          | `app-abc123,app-def456` |
| `DIFY_API_BASE`        | Dify API 的基础 URL                                                                                                     | ✅   | `-`          | `https://api.dify.ai/v1` |
| `TIMEOUT`              | HTTP 请求超时时间（秒）                                                                                                       | ❌   | `30.0`       | `60.0` |
| `SERVER_HOST`          | 服务监听主机地址                                                                                                         | ❌   | `127.0.0.1`  | `0.0.0.0` |
| `SERVER_PORT`          | 服务监听端口                                                                                                         | ❌   | `8000`       | `8080` |
| `CONVERSATION_MEMORY_MODE`| 对话记忆模式：<br>• `1` - 历史消息模式（将历史拼接到当前请求）<br>• `2` - Conversation ID 模式（零宽字符编码）  | ❌   | `1`          | `2` |
| `WORKERS`              | 生产模式 uvicorn worker 进程数（多核 CPU 推荐 2-4）                                                                   | ❌   | `1`          | `4` |

---

## 📚 使用示例

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-proxy-key",
    base_url="http://127.0.0.1:8000/v1"
)

# 非流式调用
response = client.chat.completions.create(
    model="your-dify-app-name",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="your-dify-app-name",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 工具调用示例
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="your-dify-app-name",
    messages=[{"role": "user", "content": "北京今天天气如何？"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"调用工具: {tool_call.function.name}")
        print(f"参数: {tool_call.function.arguments}")
```

### cURL 请求
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-proxy-key" \
  -d '{
    "model": "your-dify-app-name",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

---

## 🏗️ 架构设计

### 核心组件

#### **DifyModelManager**
- 管理 Dify 应用名称到 API 密钥的映射
- 自动从 Dify `/info` 端点获取应用信息
- 30 分钟 TTL 缓存减少 API 调用
- HTTP/2 连接池（100 连接，30 秒 keepalive）

#### **转换流程**
```
OpenAI Request → 验证 API Key → 模型查找 → 消息处理 →
工具注入 → Dify API 调用 → 响应转换 → OpenAI Response
```

#### **工具调用处理**
1. **注入阶段**：将工具定义注入到 system prompt
2. **提取阶段**：从响应中提取工具调用（3 种策略）
3. **清理阶段**：移除工具 JSON，保留文本内容
4. **转换阶段**：转换为 OpenAI 格式（id、type、function）

#### **零宽字符编码（模式 2）**
- Base64 编码 conversation_id
- 每 4 个字符插入零宽字符（`\u200b`、`\u200c`、`\u200d`、`\ufeff`）
- 对用户不可见，实现隐藏的会话状态传递
- LRU 缓存（256 条目）提升编码性能

---

## 🐛 故障排查

### 常见问题

**Q: 提示 "Invalid API key" 错误？**
- 检查 `.env` 文件中的 `VALID_API_KEYS` 配置
- 确保请求头 `Authorization: Bearer <key>` 格式正确

**Q: 提示 "Model not configured" 错误？**
- 运行 `curl http://127.0.0.1:8000/v1/models` 查看可用模型列表
- 检查 `DIFY_API_KEYS` 是否正确配置
- 确认 Dify 应用名称与请求的 `model` 字段匹配

**Q: 工具调用不生效？**
- 确认 `TOOL_SUPPORT = True`（默认开启）
- 检查工具定义格式是否符合 OpenAI 规范
- 查看日志中是否有提取错误

**Q: 对话上下文丢失？**
- 如果使用模式 1，确保客户端传递完整历史消息
- 如果使用模式 2，检查 assistant 响应中是否包含零宽字符编码的 conversation_id

**Q: 性能问题？**
- 增加 `WORKERS` 数量（多核 CPU）
- 调整 `TIMEOUT` 值
- 检查网络延迟到 Dify API

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！⭐**

**Made with ❤️ by [realnghon](https://github.com/realnghon)**

</div>

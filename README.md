# OpenDify

**🚀 高性能 Dify-to-OpenAI 代理服务**

将 [Dify](https://dify.ai) API 转换为 OpenAI 兼容格式，支持流式响应、工具调用、多模型管理。

## 核心特性

- ✅ **完整 OpenAI API 兼容**：`/v1/chat/completions` 和 `/v1/models`
- ✅ **高准确率工具调用**：精确的参数提示、自动类型推断、完整性验证
- ✅ **流式响应**：SSE 流式输出，性能极致优化
- ✅ **对话记忆**：支持历史消息模式和零宽字符编码的 Conversation ID 模式
- ✅ **高性能**：HTTP/2 连接池、智能缓存、多重优化

## 快速开始

### 安装
```bash
# 克隆项目
git clone https://github.com/realnghon/OpenDify.git
cd OpenDify

# 安装依赖
pip install -r requirements.txt

# 配置 .env 文件
VALID_API_KEYS=sk-your-key
DIFY_API_KEYS=app-your-dify-key
DIFY_API_BASE=https://api.dify.ai/v1

# 启动服务
python app.py
```

### 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `VALID_API_KEYS` | 代理 API 密钥（逗号分隔） | - |
| `DIFY_API_KEYS` | Dify 应用密钥（逗号分隔） | - |
| `DIFY_API_BASE` | Dify API 基础 URL | - |
| `CONVERSATION_MEMORY_MODE` | `1`=历史消息 `2`=零宽字符编码 | `1` |
| `SERVER_PORT` | 监听端口 | `8000` |
| `WORKERS` | Worker 进程数 | `1` |

## 使用示例

### Python
```python
from openai import OpenAI

client = OpenAI(api_key="sk-your-key", base_url="http://127.0.0.1:8000/v1")

# 流式调用
stream = client.chat.completions.create(
    model="your-dify-app-name",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

# 工具调用
response = client.chat.completions.create(
    model="your-dify-app-name",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }]
)
```

### cURL
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "your-app", "messages": [{"role": "user", "content": "Hi"}]}'
```

## 工具调用优化

### 高准确率设计
- ✅ **精确参数提示**：每个工具生成具体示例，显示确切参数名和类型
- ✅ **结构化说明**：object/array 类型展示内部结构
- ✅ **关键规则强调**：用 `<CRITICAL_RULES>` 突出重要约束
- ✅ **完整性验证**：提取时验证必需字段（function.name、arguments）
- ✅ **递归类型处理**：自动为嵌套 object/array 生成正确示例

## 故障排查

- **Invalid API key**: 检查 `.env` 中 `VALID_API_KEYS`
- **Model not configured**: 访问 `/v1/models` 查看可用模型
- **工具调用失败**: 检查工具定义格式
- **上下文丢失**: 确认 `CONVERSATION_MEMORY_MODE` 配置

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

⭐ **Star this repo if helpful!** | Made with ❤️ by [realnghon](https://github.com/realnghon)

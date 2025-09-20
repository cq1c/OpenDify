# OpenDify

## 简介

OpenDify 是一个 **极致优化的高性能 FastAPI 代理服务**，可以将 [Dify](https://dify.ai) API 转换为 **OpenAI 兼容格式 API**。
借助 OpenDify，您可以用任何 OpenAI 风格的 SDK 或客户端直接访问 Dify 应用，无需修改现有基于 OpenAI 的代码或工作流。

## 🛠 快速开始

### 1. 环境要求
- Python 3.8+

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
VALID_API_KEYS=your_proxy_access_key_1,your_proxy_access_key_2
DIFY_API_KEYS=your_dify_api_key_1,your_dify_api_key_2
DIFY_API_BASE=https://api.dify.ai/v1

# 可选配置
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

---

## 🔧 配置项说明

| 环境变量               | 说明                                                                                                                   | 默认值       | 示例值 |
|------------------------|------------------------------------------------------------------------------------------------------------------------|--------------|--------|
| `VALID_API_KEYS`       | 允许访问代理的 API Key 列表（逗号分隔）                                                                                             | `-`          | `sk-abc123,sk-def456` |
| `DIFY_API_KEYS`        | 用于请求 Dify API 的密钥列表（逗号分隔，支持多个应用）                                                                                                | `-`          | `app-abc123,app-def456` |
| `DIFY_API_BASE`        | Dify API 的基础 URL                                                                                                     | `-`          | `https://api.dify.ai/v1` |
| `TIMEOUT`              | 请求超时时间（秒）                                                                                                       | `30.0`       | `60.0` |
| `SERVER_HOST`          | 代理服务监听主机                                                                                                         | `127.0.0.1`  | `0.0.0.0` |
| `SERVER_PORT`          | 代理服务监听端口                                                                                                         | `8000`       | `8080` |
| `CONVERSATION_MEMORY_MODE`| 对话记忆模式：<br>• `1` - 历史消息模式（将历史消息拼接到当前请求）<br>• `2` - conversation_id 模式（使用 Dify 的对话 ID 机制）  | `1`          | `2` |
| `WORKERS`              | 生产模式下 uvicorn 的 worker 数量，用于支持多进程并发                                                                   | `1`          | `4` |

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！⭐**

</div>

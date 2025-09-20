import asyncio
import logging
import time
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import ujson
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from functools import lru_cache
import base64
from datetime import datetime, timedelta
import uuid

# 优化日志配置
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 禁用第三方库日志
for lib in ["httpx", "httpcore", "uvicorn.access"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]
CONVERSATION_MEMORY_MODE = int(os.getenv('CONVERSATION_MEMORY_MODE', '1'))
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")
TIMEOUT = float(os.getenv("TIMEOUT", 30.0))

# 性能优化常量
CONNECTION_POOL_SIZE = 100
CONNECTION_TIMEOUT = float(os.getenv("TIMEOUT", 30.0))
KEEPALIVE_TIMEOUT = 30.0
TTL_APP_CACHE = timedelta(minutes=30)

# 简化零宽字符映射
ZERO_WIDTH_CHARS = ['\u200b', '\u200c', '\u200d', '\ufeff']


class DifyModelManager:
    """管理Dify模型与API密钥映射"""
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}
        self._app_cache = {}  # api_key -> (app_name, cached_time)

        # 优化的HTTP客户端
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=CONNECTION_TIMEOUT, connect=5.0),
            limits=httpx.Limits(
                max_keepalive_connections=CONNECTION_POOL_SIZE,
                max_connections=CONNECTION_POOL_SIZE,
                keepalive_expiry=KEEPALIVE_TIMEOUT,
            ),
            http2=True,
            verify=False,
        )
        self.load_api_keys()

    def load_api_keys(self):
        keys_str = os.getenv('DIFY_API_KEYS', '')
        self.api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]

    async def fetch_app_info(self, api_key: str) -> Optional[str]:
        try:
            now = datetime.utcnow()
            # 检查缓存
            if api_key in self._app_cache:
                cached_name, cached_time = self._app_cache[api_key]
                if now - cached_time < TTL_APP_CACHE:
                    return cached_name

            headers = {"Authorization": f"Bearer {api_key}"}
            rsp = await self._client.get(
                f"{DIFY_API_BASE}/info",
                headers=headers,
                params={"user": "default_user"},
                timeout=10.0
            )

            if rsp.status_code == 200:
                app_info = ujson.loads(rsp.content)
                app_name = app_info.get("name", "Unknown App")
                self._app_cache[api_key] = (app_name, now)
                return app_name
            return None
        except Exception:
            return None

    async def refresh_model_info(self):
        """刷新模型映射"""
        self.name_to_api_key.clear()
        tasks = [self.fetch_app_info(key) for key in self.api_keys]
        names = await asyncio.gather(*tasks, return_exceptions=True)

        for key, name in zip(self.api_keys, names):
            if isinstance(name, str):
                self.name_to_api_key[name] = key

    def get_api_key(self, model: str) -> Optional[str]:
        return self.name_to_api_key.get(model)

    def get_available_models(self) -> List[Dict[str, Any]]:
        timestamp = int(time.time())
        return [
            {"id": name, "object": "model", "created": timestamp, "owned_by": "dify"}
            for name in self.name_to_api_key.keys()
        ]

    async def close(self):
        await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client


# 全局单例
model_manager = DifyModelManager()

app = FastAPI(title="Dify to OpenAI API Proxy", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)


# 简化的零宽字符编码
@lru_cache(maxsize=256)
def encode_conversation_id(conversation_id: str) -> str:
    """简化的conversation_id编码"""
    if not conversation_id:
        return ""

    # 使用简单的base64编码 + 零宽字符
    encoded = base64.b64encode(conversation_id.encode('utf-8')).decode('ascii')
    # 用零宽字符替换部分字符使其不可见
    result = ""
    for i, char in enumerate(encoded):
        if i % 4 == 0:  # 每4个字符插入一个零宽字符
            result += ZERO_WIDTH_CHARS[i % len(ZERO_WIDTH_CHARS)]
        result += char
    return result

def decode_conversation_id(content: str) -> Optional[str]:
    """简化的conversation_id解码"""
    if not content:
        return None

    try:
        # 移除零宽字符
        cleaned = ""
        for char in content:
            if char not in ZERO_WIDTH_CHARS:
                cleaned += char

        # 反向查找base64字符串
        for i in range(len(cleaned) - 4, -1, -1):
            try:
                potential_b64 = cleaned[i:]
                if len(potential_b64) % 4 == 0:
                    decoded = base64.b64decode(potential_b64).decode('utf-8')
                    return decoded
            except:
                continue
        return None
    except Exception:
        return None


# 自定义异常
class HTTPUnauthorized(HTTPException):
    def __init__(self, message):
        super().__init__(
            status_code=401,
            detail={"error": {"message": message, "type": "invalid_request_error"}}
        )

# 依赖注入
async def verify_api_key(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPUnauthorized("Invalid Authorization header")

    key = auth_header[7:]
    if not key or key not in VALID_API_KEYS:
        raise HTTPUnauthorized("Invalid API key")
    return key


# 业务函数 - JSON转换
def transform_openai_to_dify(openai_request: Dict, endpoint: str) -> Optional[Dict]:
    if endpoint != "/chat/completions" or not openai_request.get("messages"):
        return None

    messages = openai_request["messages"]
    stream = openai_request.get("stream", False)
    system_content = ""
    user_query = ""

    # 提取system消息
    for m in messages:
        if m.get("role") == "system":
            system_content = m.get("content", "")
            break

    # 处理最后一条用户消息
    last_message = messages[-1]
    if last_message.get("role") == "user":
        content = last_message.get("content", "")
        if isinstance(content, list):
            # 多模态内容，只提取文本
            user_query = " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            user_query = content

    # 处理工具调用
    tools = openai_request.get("tools", [])
    tool_choice = openai_request.get("tool_choice", "auto")

    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = None
        if len(messages) > 1:
            for m in reversed(messages[:-1]):
                if m.get("role") == "assistant":
                    conversation_id = decode_conversation_id(m.get("content", ""))
                    if conversation_id:
                        break

        if system_content and not conversation_id:
            user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"

        dify_request = {
            "inputs": {},
            "query": user_query,
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": conversation_id,
            "user": openai_request.get("user", "default_user")
        }
    else:
        # history模式
        if len(messages) > 1:
            history_msg = []
            has_system = any(m.get("role") == "system" for m in messages[:-1])
            for m in messages[:-1]:
                role, content = m.get("role", ""), m.get("content", "")
                if role and content:
                    history_msg.append(f"{role}: {content}")

            if system_content and not has_system:
                history_msg.insert(0, f"system: {system_content}")

            if history_msg:
                history_txt = "\n\n".join(history_msg)
                user_query = f"<history>\n{history_txt}\n</history>\n\n用户当前问题: {user_query}"
        elif system_content:
            user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"

        dify_request = {
            "inputs": {},
            "query": user_query,
            "response_mode": "streaming" if stream else "blocking",
            "user": openai_request.get("user", "default_user")
        }

    # 添加工具支持
    if tools:
        dify_request["tools"] = transform_tools_to_dify(tools, tool_choice)

    return dify_request


def transform_tools_to_dify(tools: List[Dict], tool_choice: Union[str, Dict] = "auto") -> Dict:
    """将OpenAI工具格式转换为Dify格式"""
    dify_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.get("function", {}).get("name", ""),
                "description": tool.get("function", {}).get("description", ""),
                "parameters": tool.get("function", {}).get("parameters", {})
            }
        }
        for tool in tools if tool.get("type") == "function"
    ]

    result = {"tools": dify_tools}

    # 处理tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        result["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice.get("function", {}).get("name", "")}
        }
    elif tool_choice == "none":
        result["tool_choice"] = "none"
    else:
        result["tool_choice"] = "auto"

    return result


def stream_transform(dify_response: Dict, model: str, stream: bool) -> Dict:
    if stream:
        return dify_response

    answer = dify_response.get("answer", "")
    tool_calls = []

    # 处理工具调用
    if "tool_calls" in dify_response:
        tool_calls = [
            {
                "id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": ujson.dumps(tc.get("function", {}).get("arguments", {}))
                }
            }
            for tc in dify_response["tool_calls"]
        ]

    # 处理agent_thoughts
    if not answer:
        agent_thoughts = dify_response.get("agent_thoughts")
        if agent_thoughts:
            for thought in reversed(agent_thoughts):
                thought_content = thought.get("thought")
                if thought_content:
                    answer = thought_content
                    break

    # 对话ID编码
    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = dify_response.get("conversation_id", "")
        if conversation_id:
            history = dify_response.get("conversation_history", [])
            has_id = any(
                msg.get("role") == "assistant" and decode_conversation_id(msg.get("content", ""))
                for msg in history
            )
            if not has_id:
                answer += encode_conversation_id(conversation_id)

    # 构建响应
    message_content = {"role": "assistant", "content": answer}
    if tool_calls:
        message_content["tool_calls"] = tool_calls

    return {
        "id": dify_response.get("message_id", ""),
        "object": "chat.completion",
        "created": dify_response.get("created", int(time.time())),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message_content,
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }]
    }


# --------------------------
# 核心路由 - /chat/completions
# --------------------------
async def stream_response(dify_request: Dict, api_key: str, model: str) -> AsyncGenerator[str, None]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = f"{DIFY_API_BASE}/chat-messages"
    message_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    def make_chunk(content: str = "", tool_calls: List[Dict] = None, final=False):
        chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }]
        }

        if content:
            chunk["choices"][0]["delta"]["content"] = content
        if tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
        if final:
            chunk["choices"][0]["finish_reason"] = "tool_calls" if tool_calls else "stop"

        return f"data: {ujson.dumps(chunk)}\n\n"

    async with model_manager.client.stream('POST', endpoint, json=dify_request, headers=headers) as rsp:
        if rsp.status_code != 200:
            yield make_chunk("Stream connection failed", final=True)
            yield "data: [DONE]\n\n"
            return

        buffer = b""
        async for chunk in rsp.aiter_bytes(8192):
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()

                if not line.startswith(b"data: "):
                    continue

                try:
                    data = ujson.loads(line[6:])
                    event_type = data.get("event")

                    if event_type in ("message", "agent_message"):
                        answer = data.get("answer", "")
                        if answer:
                            yield make_chunk(answer)

                    elif event_type == "agent_thought":
                        thought = data.get("thought", "")
                        if thought:
                            yield make_chunk(thought)

                    elif event_type == "tool_calls":
                        tool_calls = data.get("tool_calls", [])
                        if tool_calls:
                            openai_tools = [
                                {
                                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": ujson.dumps(tc.get("function", {}).get("arguments", {}))
                                    }
                                }
                                for tc in tool_calls
                            ]
                            yield make_chunk(tool_calls=openai_tools)

                    elif event_type == "message_end":
                        yield make_chunk("", final=True)
                        yield "data: [DONE]\n\n"
                        return

                except:
                    continue

        yield make_chunk("", final=True)
        yield "data: [DONE]\n\n"
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        openai_request = ujson.loads(request_body)
        model = openai_request.get("model", "claude-3-5-sonnet-v2")

        # 检查模型可用性
        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise HTTPException(status_code=404, detail={"error": {"message": f"Model {model} not configured"}})

        dify_req = transform_openai_to_dify(openai_request, "/chat/completions")
        if not dify_req:
            raise HTTPException(status_code=400, detail={"error": {"message": "Invalid format"}})

        stream = openai_request.get("stream", False)
        if stream:
            return StreamingResponse(
                stream_response(dify_req, dify_key, model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )

        # 非流式调用
        headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
        endpoint = f"{DIFY_API_BASE}/chat-messages"

        resp = await model_manager.client.post(
            endpoint,
            content=ujson.dumps(dify_req),
            headers=headers,
            timeout=20.0
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        dify_resp = ujson.loads(resp.content)
        openai_resp = stream_transform(dify_resp, model, stream=False)

        return Response(
            content=ujson.dumps(openai_resp),
            media_type="application/json",
            headers={"access-control-allow-origin": "*"}
        )

    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise HTTPException(status_code=400, detail={"error": {"message": "Invalid JSON format"}})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"message": str(e)}})


@app.get("/v1/models")
async def list_models():
    if not model_manager.name_to_api_key:
        await model_manager.refresh_model_info()

    models = model_manager.get_available_models()
    return Response(
        content=ujson.dumps({"object": "list", "data": models}),
        media_type="application/json",
        headers={"access-control-allow-origin": "*"}
    )


# 应用生命周期
@app.on_event("startup")
async def startup():
    if not VALID_API_KEYS:
        logger.warning("VALID_API_KEYS not configured")
    await model_manager.refresh_model_info()

@app.on_event("shutdown")
async def shutdown():
    await model_manager.close()


if __name__ == '__main__':
    import uvicorn

    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        access_log=False,
        server_header=False,
        date_header=False,
        loop="uvloop" if os.name != 'nt' else "asyncio",
    )

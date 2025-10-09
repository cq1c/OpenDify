import asyncio
import logging
import time
import re
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import ujson
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime, timedelta
import secrets

# 优化日志配置
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
for lib in ["httpx", "httpcore", "uvicorn.access"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]
VALID_API_KEYS_SET = frozenset(VALID_API_KEYS)
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")
TIMEOUT = float(os.getenv("TIMEOUT", 30.0))

# 性能优化常量
CONNECTION_POOL_SIZE = 100
CONNECTION_TIMEOUT = TIMEOUT
KEEPALIVE_TIMEOUT = 30.0
TTL_APP_CACHE = timedelta(minutes=30)

# 工具支持配置
TOOL_SUPPORT = True
SCAN_LIMIT = 8000

def fast_uuid() -> str:
    """快速生成UUID hex"""
    return secrets.token_hex(16)


class DifyModelManager:
    """管理Dify模型与API密钥映射"""
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}
        self._app_cache = {}

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


class HTTPUnauthorized(HTTPException):
    def __init__(self, message):
        super().__init__(
            status_code=401,
            detail={"error": {"message": message, "type": "invalid_request_error"}}
        )


async def verify_api_key(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPUnauthorized("Invalid Authorization header")

    key = auth_header[7:]
    if not key or key not in VALID_API_KEYS_SET:
        raise HTTPUnauthorized("Invalid API key")
    return key


def transform_openai_to_dify(openai_request: Dict) -> tuple[Optional[Dict], str]:
    """
    将OpenAI请求转换为Dify请求（符合OpenAI标准）
    返回: (dify_request, conversation_id_for_response)
    """
    messages = openai_request.get("messages", [])
    if not messages:
        return None, ""

    tools = openai_request.get("tools", [])
    tool_choice = openai_request.get("tool_choice", "auto")

    # 将messages序列化为Dify query
    query_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # 处理多模态内容
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = " ".join(text_parts)

        if role == "system":
            query_parts.append(f"[System]: {content}")
        elif role == "user":
            query_parts.append(f"[User]: {content}")
        elif role == "assistant":
            if "tool_calls" in msg:
                tool_calls_text = []
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    tool_calls_text.append(f"调用工具 {func.get('name')}: {func.get('arguments')}")
                query_parts.append(f"[Assistant]: {content or ''} {'; '.join(tool_calls_text)}")
            else:
                query_parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            query_parts.append(f"[Tool '{tool_name}' Result]: {content}")

    # 添加工具定义
    if tools and TOOL_SUPPORT and tool_choice != "none":
        tools_prompt = generate_tool_prompt(tools)
        query_parts.insert(0, tools_prompt)

        if tool_choice == "required":
            query_parts.append("\n[重要]: 必须使用提供的工具函数。")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname:
                query_parts.append(f"\n[重要]: 使用 {fname} 函数。")

    user_query = "\n\n".join(query_parts)

    # 从system_fingerprint获取conversation_id（客户端传递）
    conversation_id = openai_request.get("system_fingerprint", "")

    dify_request = {
        "inputs": {},
        "query": user_query,
        "response_mode": "streaming" if openai_request.get("stream", False) else "blocking",
        "user": openai_request.get("user", "default_user")
    }

    if conversation_id:
        dify_request["conversation_id"] = conversation_id

    return dify_request, conversation_id


def generate_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """生成工具定义提示"""
    if not tools:
        return ""

    tool_definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue

        function_spec = tool.get("function", {}) or {}
        function_name = function_spec.get("name", "unknown")
        function_description = function_spec.get("description", "")
        parameters = function_spec.get("parameters", {}) or {}

        tool_info = [f"## {function_name}", f"**Purpose**: {function_description}"]

        parameter_properties = parameters.get("properties", {}) or {}
        required_parameters = set(parameters.get("required", []) or [])

        if parameter_properties:
            tool_info.append("**Parameters**:")
            for param_name, param_details in parameter_properties.items():
                param_type = (param_details or {}).get("type", "any")
                param_desc = (param_details or {}).get("description", "")
                requirement_flag = "**Required**" if param_name in required_parameters else "*Optional*"
                tool_info.append(f"- `{param_name}` ({param_type}) - {requirement_flag}: {param_desc}")

        tool_definitions.append("\n".join(tool_info))

    if not tool_definitions:
        return ""

    return (
        "\n\n# AVAILABLE FUNCTIONS\n" + "\n\n---\n".join(tool_definitions) +
        "\n\n# USAGE INSTRUCTIONS\n"
        "When you need to call a function, respond with JSON:\n"
        "```json\n"
        '{"tool_calls": [{"id": "call_xxx", "type": "function", "function": {"name": "function_name", "arguments": "{\\"param\\": \\"value\\"}"}}]}\n'
        "```\n"
        "Important: 'arguments' must be a JSON string, not an object.\n"
    )


# 工具提取
TOOL_CALL_FENCE_PATTERN = re.compile(r"```json\s*(\{[^`]+\})\s*```", re.DOTALL)

def extract_tool_invocations(text: str) -> Optional[List[Dict[str, Any]]]:
    """从响应文本中提取工具调用"""
    if not text:
        return None

    scannable_text = text[:SCAN_LIMIT]

    # 尝试从JSON代码块提取
    json_blocks = TOOL_CALL_FENCE_PATTERN.findall(scannable_text)
    for json_block in json_blocks:
        try:
            parsed_data = ujson.loads(json_block)
            tool_calls = parsed_data.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                # 标准化格式
                for tc in tool_calls:
                    if "function" in tc and "arguments" in tc["function"]:
                        if not isinstance(tc["function"]["arguments"], str):
                            tc["function"]["arguments"] = ujson.dumps(tc["function"]["arguments"], ensure_ascii=False)
                return tool_calls
        except (ujson.JSONDecodeError, AttributeError):
            continue

    return None


def remove_tool_json_content(text: str) -> str:
    """从响应文本中移除工具JSON内容"""
    def remove_tool_call_block(match: re.Match) -> str:
        json_content = match.group(1)
        try:
            parsed_data = ujson.loads(json_content)
            if "tool_calls" in parsed_data:
                return ""
        except:
            pass
        return match.group(0)

    return TOOL_CALL_FENCE_PATTERN.sub(remove_tool_call_block, text).strip()


def transform_dify_to_openai_response(dify_response: Dict, model: str) -> Dict:
    """将Dify响应转换为OpenAI标准格式"""
    answer = dify_response.get("answer", "")
    tool_calls = []

    # 提取工具调用
    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = extracted_tools
            answer = remove_tool_json_content(answer)

    # Dify原生工具调用
    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = [
            {
                "id": tc.get("id", f"call_{fast_uuid()}"),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": ujson.dumps(tc.get("function", {}).get("arguments", {}), ensure_ascii=False)
                }
            }
            for tc in dify_response["tool_calls"]
        ]

    # 构建message
    message_content = {
        "role": "assistant",
        "content": answer.strip() if answer else None
    }

    if tool_calls:
        message_content["tool_calls"] = tool_calls
        if not message_content["content"]:
            message_content["content"] = None

    # 构建响应
    openai_response = {
        "id": dify_response.get("message_id", f"chatcmpl-{fast_uuid()}"),
        "object": "chat.completion",
        "created": dify_response.get("created_at", int(time.time())),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message_content,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "logprobs": None
        }],
        "usage": {
            "prompt_tokens": dify_response.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": dify_response.get("metadata", {}).get("usage", {}).get("completion_tokens", 0),
            "total_tokens": dify_response.get("metadata", {}).get("usage", {}).get("total_tokens", 0)
        }
    }

    # 传递conversation_id供客户端使用
    conversation_id = dify_response.get("conversation_id")
    if conversation_id:
        openai_response["system_fingerprint"] = conversation_id

    return openai_response


async def stream_openai_response(dify_request: Dict, api_key: str, model: str) -> AsyncGenerator[str, None]:
    """流式返回OpenAI标准SSE响应"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = f"{DIFY_API_BASE}/chat-messages"
    message_id = f"chatcmpl-{fast_uuid()}"

    first_chunk = True
    accumulated_content = ""
    tool_calls_sent = False

    def make_sse_chunk(delta_content: str = None, delta_role: str = None,
                       delta_tool_calls: List[Dict] = None, finish_reason: str = None) -> str:
        chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": finish_reason
            }]
        }

        if delta_role:
            chunk["choices"][0]["delta"]["role"] = delta_role
        if delta_content is not None:
            chunk["choices"][0]["delta"]["content"] = delta_content
        if delta_tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = delta_tool_calls

        return f"data: {ujson.dumps(chunk, ensure_ascii=False)}\n\n"

    try:
        async with model_manager.client.stream('POST', endpoint, json=dify_request, headers=headers) as rsp:
            if rsp.status_code != 200:
                yield make_sse_chunk(delta_role="assistant", delta_content=f"Error: HTTP {rsp.status_code}", finish_reason="error")
                yield "data: [DONE]\n\n"
                return

            buffer = bytearray()
            async for chunk in rsp.aiter_bytes(8192):
                buffer.extend(chunk)

                while b"\n" in buffer:
                    idx = buffer.find(b"\n")
                    line = bytes(buffer[:idx]).strip()
                    buffer = buffer[idx+1:]

                    if not line.startswith(b"data: "):
                        continue

                    try:
                        data = ujson.loads(line[6:])
                        event_type = data.get("event")

                        if event_type in ("message", "agent_message"):
                            answer_delta = data.get("answer", "")
                            if answer_delta:
                                accumulated_content += answer_delta

                                if first_chunk:
                                    yield make_sse_chunk(delta_role="assistant", delta_content=answer_delta)
                                    first_chunk = False
                                else:
                                    yield make_sse_chunk(delta_content=answer_delta)

                        elif event_type == "tool_calls":
                            dify_tool_calls = data.get("tool_calls", [])
                            if dify_tool_calls:
                                tool_calls_sent = True

                                openai_tool_calls = []
                                for idx, tc in enumerate(dify_tool_calls):
                                    func = tc.get("function", {})
                                    openai_tool_calls.append({
                                        "index": idx,
                                        "id": tc.get("id", f"call_{fast_uuid()}"),
                                        "type": "function",
                                        "function": {
                                            "name": func.get("name", ""),
                                            "arguments": ujson.dumps(func.get("arguments", {}), ensure_ascii=False)
                                        }
                                    })

                                if first_chunk:
                                    yield make_sse_chunk(delta_role="assistant")
                                    first_chunk = False

                                yield make_sse_chunk(delta_tool_calls=openai_tool_calls)

                        elif event_type == "message_end":
                            # 最后尝试提取工具调用
                            if TOOL_SUPPORT and not tool_calls_sent and accumulated_content:
                                extracted = extract_tool_invocations(accumulated_content)
                                if extracted:
                                    tool_calls_sent = True

                                    if first_chunk:
                                        yield make_sse_chunk(delta_role="assistant")
                                        first_chunk = False

                                    openai_tool_calls = []
                                    for idx, tc in enumerate(extracted):
                                        openai_tool_calls.append({
                                            "index": idx,
                                            "id": tc.get("id", f"call_{fast_uuid()}"),
                                            "type": tc.get("type", "function"),
                                            "function": tc.get("function", {})
                                        })
                                    yield make_sse_chunk(delta_tool_calls=openai_tool_calls)

                            finish_reason = "tool_calls" if tool_calls_sent else "stop"
                            yield make_sse_chunk(finish_reason=finish_reason)
                            yield "data: [DONE]\n\n"
                            return

                    except Exception as e:
                        logger.error(f"Stream processing error: {e}")
                        continue

            yield make_sse_chunk(finish_reason="stop")
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield make_sse_chunk(delta_role="assistant", delta_content=f"Error: {str(e)}", finish_reason="error")
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        openai_request = ujson.loads(request_body)
        model = openai_request.get("model", "claude-3-5-sonnet-v2")

        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise HTTPException(status_code=404, detail={"error": {"message": f"Model {model} not configured"}})

        dify_req, _ = transform_openai_to_dify(openai_request)
        if not dify_req:
            raise HTTPException(status_code=400, detail={"error": {"message": "Invalid format"}})

        stream = openai_request.get("stream", False)
        if stream:
            return StreamingResponse(
                stream_openai_response(dify_req, dify_key, model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )

        # 非流式
        headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
        endpoint = f"{DIFY_API_BASE}/chat-messages"

        resp = await model_manager.client.post(
            endpoint,
            content=ujson.dumps(dify_req),
            headers=headers,
            timeout=20.0
        )

        if resp.status_code != 200:
            # 解析 Dify 错误并转换为 OpenAI 格式
            try:
                dify_error = ujson.loads(resp.text)
                error_message = dify_error.get("message", resp.text)
                error_code = dify_error.get("code", "unknown_error")
            except:
                error_message = resp.text
                error_code = "unknown_error"

            # 映射 HTTP 状态码
            if resp.status_code == 503:
                openai_error_type = "server_error"
                error_message = f"The model is currently overloaded. Please try again later. (Original: {error_message})"
            elif resp.status_code >= 500:
                openai_error_type = "server_error"
            elif resp.status_code == 429:
                openai_error_type = "rate_limit_exceeded"
            elif resp.status_code >= 400:
                openai_error_type = "invalid_request_error"
            else:
                openai_error_type = "api_error"

            raise HTTPException(
                status_code=resp.status_code,
                detail={
                    "error": {
                        "message": error_message,
                        "type": openai_error_type,
                        "code": error_code
                    }
                }
            )

        dify_resp = ujson.loads(resp.content)
        openai_resp = transform_dify_to_openai_response(dify_resp, model)

        return Response(
            content=ujson.dumps(openai_resp),
            media_type="application/json",
            headers={"access-control-allow-origin": "*"}
        )

    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Invalid JSON format in request",
                    "type": "invalid_request_error",
                    "code": "invalid_json"
                }
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "internal_error"
                }
            }
        )


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

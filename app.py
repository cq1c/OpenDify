import asyncio
import logging
import time
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, UJSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import ujson
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple
from datetime import datetime, timedelta, timezone
import secrets
from starlette.background import BackgroundTask

# 加载环境变量（优先于日志配置）
from dotenv import load_dotenv
load_dotenv()

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").strip().upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING), format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
for lib in ["httpx", "httpcore", "uvicorn.access"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]
VALID_API_KEYS_SET = frozenset(VALID_API_KEYS)
DIFY_API_BASE = (os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1") or "").rstrip("/")
TIMEOUT = float(os.getenv("TIMEOUT", 30.0))
AUTH_MODE = os.getenv("AUTH_MODE", "required").strip().lower()  # required|disabled
try:
    CONVERSATION_MEMORY_MODE = int(os.getenv("CONVERSATION_MEMORY_MODE", "1").strip() or "1")
except Exception:
    CONVERSATION_MEMORY_MODE = 1

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _openai_error_payload(
    message: str,
    *,
    type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> Dict[str, Any]:
    return {"error": {"message": message, "type": type, "param": param, "code": code}}


class OpenAIHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type
        self.code = code
        self.param = param

    def payload(self) -> Dict[str, Any]:
        return _openai_error_payload(self.message, type=self.type, code=self.code, param=self.param)


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
            verify=_env_flag("DIFY_SSL_VERIFY", True),
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

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not VALID_API_KEYS:
        logger.warning("VALID_API_KEYS not configured")
    await model_manager.refresh_model_info()
    yield
    await model_manager.close()


app = FastAPI(title="Dify to OpenAI API Proxy", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)


@app.exception_handler(OpenAIHTTPError)
async def _openai_http_error_handler(_: Request, exc: OpenAIHTTPError) -> UJSONResponse:
    return UJSONResponse(status_code=exc.status_code, content=exc.payload())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> UJSONResponse:
    # 将 FastAPI 请求校验错误转换为 OpenAI 标准错误体
    return UJSONResponse(
        status_code=400,
        content=_openai_error_payload(
            "Invalid request parameters",
            type="invalid_request_error",
            code="invalid_request_error",
            param=None,
        ),
    )


@app.exception_handler(HTTPException)
async def _http_exception_handler(_: Request, exc: HTTPException) -> UJSONResponse:
    # FastAPI 默认会把 detail 包在 {"detail": ...}，这里统一转换为 OpenAI 错误体
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail and isinstance(detail["error"], dict):
        return UJSONResponse(status_code=exc.status_code, content=detail)

    message = str(detail) if detail is not None else "Request failed"
    err_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return UJSONResponse(
        status_code=exc.status_code,
        content=_openai_error_payload(message, type=err_type, code=None, param=None),
    )


async def verify_api_key(request: Request) -> str:
    if AUTH_MODE == "disabled":
        return ""
    auth_header = request.headers.get("Authorization")
    x_api_key = request.headers.get("X-API-Key")

    key: Optional[str] = None
    if isinstance(auth_header, str) and auth_header.strip():
        if auth_header.lower().startswith("bearer "):
            key = auth_header[7:]
        else:
            # Allow passing raw key in Authorization header for convenience.
            key = auth_header.strip()
    elif isinstance(x_api_key, str) and x_api_key.strip():
        key = x_api_key.strip()

    if not key:
        raise OpenAIHTTPError(
            401,
            "Invalid Authorization header",
            type="invalid_request_error",
            code="invalid_api_key",
            param=None,
        )

    if not key or key not in VALID_API_KEYS_SET:
        raise OpenAIHTTPError(
            401,
            "Invalid API key",
            type="invalid_request_error",
            code="invalid_api_key",
            param=None,
        )
    return key


def transform_openai_to_dify(openai_request: Dict[str, Any], conversation_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    将OpenAI请求转换为Dify请求（符合OpenAI标准）
    """
    messages = openai_request.get("messages", [])
    if not messages:
        return None

    tools = openai_request.get("tools", [])
    tool_choice = openai_request.get("tool_choice", "auto")

    # 将 messages 序列化为 Dify query
    #
    # mode=1: 完整拼接历史（OpenAI 无状态标准做法）
    # mode=2: 当提供 conversation_id 时，仅拼接“最新增量”消息（提升长对话性能）
    if CONVERSATION_MEMORY_MODE == 2 and conversation_id:
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if (messages[idx] or {}).get("role") == "assistant":
                last_assistant_idx = idx
                break
        system_messages = [m for m in messages if (m or {}).get("role") == "system"]
        delta_messages = messages[last_assistant_idx + 1:] if last_assistant_idx >= 0 else messages
        messages_for_query = system_messages + [m for m in delta_messages if (m or {}).get("role") != "system"]
    else:
        messages_for_query = messages

    query_parts: List[str] = []

    for msg in messages_for_query:
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
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                query_parts.append(f"[Tool '{tool_name}' (tool_call_id={tool_call_id}) Result]: {content}")
            else:
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

    dify_request = {
        "inputs": {},
        "query": user_query,
        "response_mode": "streaming" if openai_request.get("stream", False) else "blocking",
        "user": openai_request.get("user", "default_user")
    }

    if conversation_id:
        dify_request["conversation_id"] = conversation_id

    return dify_request


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


def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        tc_type = tc.get("type") or "function"
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            func = {}

        name = func.get("name") or ""
        args = func.get("arguments")
        if args is None:
            args = ""
        if not isinstance(args, str):
            args = ujson.dumps(args, ensure_ascii=False)

        call_id = tc.get("id") or f"call_{fast_uuid()}"
        normalized.append(
            {
                "id": call_id,
                "type": tc_type,
                "function": {"name": name, "arguments": args},
            }
        )
    return normalized


def _to_unix_timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return int(raw)
        try:
            # 支持 ISO8601（含 Z）
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return int(time.time())
    return int(time.time())


def _ensure_chatcmpl_id(value: Any) -> str:
    if not value:
        return f"chatcmpl-{fast_uuid()}"
    s = str(value)
    if s.startswith("chatcmpl-"):
        return s
    return f"chatcmpl-{s}"


def _ensure_resp_id(value: Any) -> str:
    if not value:
        return f"resp_{fast_uuid()}"
    s = str(value)
    if s.startswith("resp_"):
        return s
    if s.startswith("chatcmpl-"):
        s = s[len("chatcmpl-") :]
    return f"resp_{s}"


def _ensure_anthropic_msg_id(value: Any) -> str:
    if not value:
        return f"msg_{fast_uuid()}"
    s = str(value)
    if s.startswith("msg_"):
        return s
    if s.startswith("chatcmpl-"):
        s = s[len("chatcmpl-") :]
    return f"msg_{s}"


def _extract_text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return " ".join(parts)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return _extract_text_from_content([value])
    return str(value)


def _responses_input_to_chat_messages(input_value: Any) -> List[Dict[str, Any]]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if isinstance(input_value, dict):
        if isinstance(input_value.get("role"), str):
            return [
                {
                    "role": input_value.get("role") or "user",
                    "content": _extract_text_from_content(input_value.get("content")),
                }
            ]
        return [{"role": "user", "content": _extract_text_from_content([input_value])}]

    if isinstance(input_value, list):
        # Either a list of messages (each has role), or a list of content parts.
        looks_like_messages = any(isinstance(it, dict) and isinstance(it.get("role"), str) for it in input_value)
        if looks_like_messages:
            messages: List[Dict[str, Any]] = []
            for it in input_value:
                if not isinstance(it, dict):
                    continue
                role = it.get("role") or "user"
                content = _extract_text_from_content(it.get("content"))
                messages.append({"role": role, "content": content})
            return messages
        return [{"role": "user", "content": _extract_text_from_content(input_value)}]

    return [{"role": "user", "content": str(input_value)}]


def transform_dify_to_openai_response(dify_response: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """将Dify响应转换为OpenAI标准格式"""
    answer = dify_response.get("answer", "")
    tool_calls: List[Dict[str, Any]] = []

    # 提取工具调用
    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = _normalize_tool_calls(extracted_tools)
            answer = remove_tool_json_content(answer)

    # Dify原生工具调用
    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = _normalize_tool_calls(list(dify_response.get("tool_calls") or []))

    # 构建message
    message_content = {
        "role": "assistant",
        "content": answer.strip() if isinstance(answer, str) and answer.strip() else None
    }

    if tool_calls:
        message_content["tool_calls"] = tool_calls

    # 构建响应
    openai_response = {
        "id": _ensure_chatcmpl_id(dify_response.get("message_id")),
        "object": "chat.completion",
        "created": _to_unix_timestamp(dify_response.get("created_at")),
        "model": model,
        "system_fingerprint": "fp_dify",
        "service_tier": None,
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

    conversation_id = dify_response.get("conversation_id")
    return openai_response, conversation_id


def transform_openai_responses_to_dify(
    responses_request: Dict[str, Any],
    conversation_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    model = responses_request.get("model")
    if not isinstance(model, str) or not model.strip():
        return None

    messages: List[Dict[str, Any]] = []

    instructions = responses_request.get("instructions")
    instructions_text = _extract_text_from_content(instructions)
    if instructions_text.strip():
        messages.append({"role": "system", "content": instructions_text})

    input_value = responses_request.get("input")
    messages.extend(_responses_input_to_chat_messages(input_value))

    # Convert to the existing OpenAI-chat-like shape and reuse the old transformer.
    openai_like: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(responses_request.get("stream", False)),
    }

    if "tools" in responses_request:
        openai_like["tools"] = responses_request.get("tools") or []
    if "tool_choice" in responses_request:
        openai_like["tool_choice"] = responses_request.get("tool_choice")

    user = responses_request.get("user")
    if isinstance(user, str) and user.strip():
        openai_like["user"] = user
    elif isinstance(responses_request.get("metadata"), dict):
        meta_user = (responses_request.get("metadata") or {}).get("user_id")
        if meta_user is not None:
            openai_like["user"] = str(meta_user)

    return transform_openai_to_dify(openai_like, conversation_id)


def transform_dify_to_openai_responses(dify_response: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_response.get("answer", "")
    tool_calls: List[Dict[str, Any]] = []

    # Extract tool calls from fenced JSON (OpenDify prompt convention)
    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = _normalize_tool_calls(extracted_tools)
            answer = remove_tool_json_content(answer)

    # Dify native tool calls
    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = _normalize_tool_calls(list(dify_response.get("tool_calls") or []))

    output: List[Dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function") or {}
        call_id = tc.get("id") or f"call_{fast_uuid()}"
        output.append(
            {
                "id": call_id,
                "type": "function_call",
                "call_id": call_id,
                "name": (func or {}).get("name") or "",
                "arguments": (func or {}).get("arguments") or "",
                "status": "completed",
            }
        )

    answer_text = answer.strip() if isinstance(answer, str) else str(answer)
    if answer_text:
        output.append(
            {
                "id": f"msg_{fast_uuid()}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer_text, "annotations": []}],
            }
        )

    usage = (dify_response.get("metadata") or {}).get("usage") if isinstance(dify_response.get("metadata"), dict) else None
    if not isinstance(usage, dict):
        usage = {}

    if not output:
        output.append(
            {
                "id": f"msg_{fast_uuid()}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )

    resp = {
        "id": _ensure_resp_id(dify_response.get("message_id")),
        "object": "response",
        "created_at": _to_unix_timestamp(dify_response.get("created_at")),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
        "error": None,
    }

    conversation_id = dify_response.get("conversation_id")
    return resp, conversation_id


def _claude_tools_to_openai_tools(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        parameters = tool.get("input_schema") or tool.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": parameters,
                },
            }
        )
    return converted


def _claude_message_to_openai_messages(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = msg.get("role") or "user"
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": _extract_text_from_content(content)}]

    text_parts: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    tool_uses: List[Dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt)
        elif btype == "tool_result":
            tool_results.append(block)
        elif btype == "tool_use":
            tool_uses.append(block)

    primary: Dict[str, Any] = {"role": role, "content": " ".join(text_parts).strip() if text_parts else ""}

    if role == "assistant" and tool_uses:
        tool_calls: List[Dict[str, Any]] = []
        for tu in tool_uses:
            call_id = tu.get("id") or f"call_{fast_uuid()}"
            name = tu.get("name") or ""
            raw_input = tu.get("input")
            if raw_input is None:
                args_str = ""
            elif isinstance(raw_input, str):
                args_str = raw_input
            else:
                args_str = ujson.dumps(raw_input, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        primary["tool_calls"] = tool_calls

    out: List[Dict[str, Any]] = [primary]

    for tr in tool_results:
        tool_use_id = tr.get("tool_use_id") or tr.get("tool_call_id") or tr.get("id")
        tr_content = tr.get("content")
        tool_content = _extract_text_from_content(tr_content)
        tool_msg: Dict[str, Any] = {"role": "tool", "name": tr.get("name") or "tool_result", "content": tool_content}
        if tool_use_id is not None:
            tool_msg["tool_call_id"] = str(tool_use_id)
        out.append(tool_msg)

    return out


def _claude_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type in (None, "auto"):
        return "auto"
    if choice_type in ("any", "required"):
        return "required"
    if choice_type in ("tool", "function"):
        name = tool_choice.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        return {"type": "function", "function": {"name": name}}
    return None


def transform_claude_messages_to_dify(
    claude_request: Dict[str, Any],
    conversation_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    model = claude_request.get("model")
    if not isinstance(model, str) or not model.strip():
        return None

    claude_messages = claude_request.get("messages")
    if not isinstance(claude_messages, list) or not claude_messages:
        return None

    messages: List[Dict[str, Any]] = []

    system = claude_request.get("system")
    system_text = _extract_text_from_content(system)
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})

    for msg in claude_messages:
        if not isinstance(msg, dict):
            continue
        messages.extend(_claude_message_to_openai_messages(msg))

    openai_like: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(claude_request.get("stream", False)),
    }

    openai_tools = _claude_tools_to_openai_tools(claude_request.get("tools"))
    if openai_tools:
        openai_like["tools"] = openai_tools

    mapped_tool_choice = _claude_tool_choice_to_openai(claude_request.get("tool_choice"))
    if mapped_tool_choice is not None:
        openai_like["tool_choice"] = mapped_tool_choice

    metadata = claude_request.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id") is not None:
        openai_like["user"] = str(metadata.get("user_id"))

    return transform_openai_to_dify(openai_like, conversation_id)


def _openai_tool_call_to_anthropic_tool_use(tc: Dict[str, Any]) -> Dict[str, Any]:
    call_id = tc.get("id") or f"call_{fast_uuid()}"
    func = tc.get("function") or {}
    name = (func or {}).get("name") or ""
    args = (func or {}).get("arguments") or ""
    input_obj: Any = {}
    if isinstance(args, str) and args.strip():
        try:
            input_obj = ujson.loads(args)
        except Exception:
            input_obj = {"_raw": args}
    return {"type": "tool_use", "id": call_id, "name": name, "input": input_obj}


def transform_dify_to_claude_message(dify_response: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_response.get("answer", "")
    tool_calls: List[Dict[str, Any]] = []

    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = _normalize_tool_calls(extracted_tools)
            answer = remove_tool_json_content(answer)

    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = _normalize_tool_calls(list(dify_response.get("tool_calls") or []))

    content: List[Dict[str, Any]] = []
    answer_text = answer.strip() if isinstance(answer, str) else str(answer)
    if answer_text:
        content.append({"type": "text", "text": answer_text})

    if tool_calls:
        for tc in tool_calls:
            content.append(_openai_tool_call_to_anthropic_tool_use(tc))

    if not content:
        content.append({"type": "text", "text": ""})

    meta_usage = (dify_response.get("metadata") or {}).get("usage") if isinstance(dify_response.get("metadata"), dict) else None
    if not isinstance(meta_usage, dict):
        meta_usage = {}

    stop_reason = "tool_use" if tool_calls else "end_turn"

    resp = {
        "id": _ensure_anthropic_msg_id(dify_response.get("message_id")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": meta_usage.get("prompt_tokens", 0),
            "output_tokens": meta_usage.get("completion_tokens", 0),
        },
    }

    conversation_id = dify_response.get("conversation_id")
    return resp, conversation_id


async def _iter_dify_sse_json(rsp: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    buffer = bytearray()
    async for chunk in rsp.aiter_bytes(8192):
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.find(b"\n")
            line = bytes(buffer[:idx]).strip()
            buffer = buffer[idx + 1:]
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if not payload:
                continue
            try:
                data = ujson.loads(payload)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data


async def stream_openai_response(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
    stream_include_usage: bool,
) -> AsyncGenerator[str, None]:
    """流式返回OpenAI标准SSE响应"""
    tool_calls_sent = False
    accumulated_for_tools = ""  # 仅用于工具提取（限制长度）
    usage_obj: Optional[Dict[str, Any]] = None

    def make_sse_chunk(delta_content: str = None, delta_role: str = None,
                       delta_tool_calls: List[Dict] = None, finish_reason: str = None) -> str:
        chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": "fp_dify",
            "service_tier": None,
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
        # OpenAI 习惯先发送一个仅包含 role 的 chunk
        yield make_sse_chunk(delta_role="assistant")

        buffer = bytearray()
        async for chunk in rsp.aiter_bytes(8192):
            buffer.extend(chunk)

            while b"\n" in buffer:
                idx = buffer.find(b"\n")
                line = bytes(buffer[:idx]).strip()
                buffer = buffer[idx + 1:]

                if not line.startswith(b"data: "):
                    continue

                try:
                    data = ujson.loads(line[6:])
                    event_type = data.get("event")

                    if event_type in ("message", "agent_message"):
                        answer_delta = data.get("answer", "")
                        if answer_delta:
                            # 限制工具提取缓存，避免长文本占用内存
                            if len(accumulated_for_tools) < SCAN_LIMIT:
                                remaining = SCAN_LIMIT - len(accumulated_for_tools)
                                accumulated_for_tools += answer_delta[:remaining]

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
                                        "name": (func or {}).get("name", ""),
                                        "arguments": ujson.dumps((func or {}).get("arguments", {}), ensure_ascii=False)
                                    }
                                })

                            yield make_sse_chunk(delta_tool_calls=openai_tool_calls)

                    elif event_type == "message_end":
                        # 记录 usage（如果 Dify 在结束事件里提供）
                        meta_usage = (data.get("metadata") or {}).get("usage") if isinstance(data.get("metadata"), dict) else None
                        if isinstance(meta_usage, dict):
                            usage_obj = {
                                "prompt_tokens": meta_usage.get("prompt_tokens", 0),
                                "completion_tokens": meta_usage.get("completion_tokens", 0),
                                "total_tokens": meta_usage.get("total_tokens", 0),
                            }

                        # 最后尝试提取工具调用
                        if TOOL_SUPPORT and not tool_calls_sent and accumulated_for_tools:
                            extracted = extract_tool_invocations(accumulated_for_tools)
                            if extracted:
                                tool_calls_sent = True
                                normalized = _normalize_tool_calls(extracted)

                                openai_tool_calls = []
                                for idx, tc in enumerate(normalized):
                                    openai_tool_calls.append({
                                        "index": idx,
                                        "id": tc["id"],
                                        "type": tc.get("type", "function"),
                                        "function": tc.get("function", {})
                                    })
                                yield make_sse_chunk(delta_tool_calls=openai_tool_calls)

                        finish_reason = "tool_calls" if tool_calls_sent else "stop"
                        yield make_sse_chunk(finish_reason=finish_reason)

                        # 可选：按 OpenAI stream_options.include_usage 发送一个 choices=[] 的 usage chunk（通常在 finish chunk 之后）
                        if stream_include_usage:
                            yield (
                                "data: "
                                + ujson.dumps(
                                    {
                                        "id": message_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "system_fingerprint": "fp_dify",
                                        "service_tier": None,
                                        "choices": [],
                                        "usage": usage_obj or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
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


async def stream_openai_responses(
    rsp: httpx.Response,
    *,
    model: str,
    response_id: str,
) -> AsyncGenerator[str, None]:
    created_at = int(time.time())
    sequence_number = 0

    output_index = 0  # current output item index (message uses this until done)
    content_index = 0

    message_item_id: Optional[str] = None
    message_open = False
    content_part_open = False
    accumulated_text = ""

    tool_items_in_order: List[Dict[str, Any]] = []
    tool_items_by_id: Dict[str, Dict[str, Any]] = {}

    final_output_by_index: Dict[int, Dict[str, Any]] = {}
    usage_obj: Dict[str, Any] = {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }

    def sse(event: Dict[str, Any]) -> str:
        nonlocal sequence_number
        payload = dict(event)
        payload["sequence_number"] = sequence_number
        sequence_number += 1
        return f"data: {ujson.dumps(payload, ensure_ascii=False)}\n\n"

    def _start_message_item() -> List[Dict[str, Any]]:
        nonlocal message_item_id, message_open, content_part_open, content_index
        if message_open:
            return []
        message_item_id = f"msg_{fast_uuid()}"
        message_open = True
        content_part_open = True
        content_index = 0
        return [
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "response.content_part.added",
                "item_id": message_item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": ""},
            },
        ]

    def _close_message_item(final_text: str) -> List[Dict[str, Any]]:
        nonlocal message_item_id, message_open, content_part_open, output_index, accumulated_text, content_index
        if not message_open or not message_item_id:
            return []

        events: List[Dict[str, Any]] = []
        if content_part_open:
            events.append(
                {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": final_text,
                }
            )
            events.append(
                {
                    "type": "response.content_part.done",
                    "item_id": message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": final_text},
                }
            )
            content_part_open = False

        item = {
            "id": message_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        }
        events.append({"type": "response.output_item.done", "output_index": output_index, "item": item})
        final_output_by_index[output_index] = item

        message_item_id = None
        message_open = False
        accumulated_text = ""
        content_index = 0

        output_index += 1
        return events

    def _start_tool_items(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nonlocal output_index
        events: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            call_id = tc.get("id") or f"call_{fast_uuid()}"
            if call_id in tool_items_by_id:
                continue
            func = tc.get("function") or {}
            name = (func or {}).get("name") or ""
            args = (func or {}).get("arguments") or ""
            tool_item = {"id": call_id, "output_index": output_index, "call_id": call_id, "name": name, "arguments": ""}
            tool_items_by_id[call_id] = tool_item
            tool_items_in_order.append(tool_item)

            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {"id": call_id, "type": "function_call", "status": "in_progress", "call_id": call_id, "name": name},
                }
            )
            if isinstance(args, str) and args:
                tool_item["arguments"] = args
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": call_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": args,
                    }
                )

            output_index += 1
        return events

    def _finish_tool_items() -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for tool_item in tool_items_in_order:
            call_id = tool_item["id"]
            out_idx = tool_item["output_index"]
            args = tool_item.get("arguments") or ""
            name = tool_item.get("name") or ""

            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": call_id,
                    "output_index": out_idx,
                    "arguments": args,
                }
            )
            item = {
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": args,
            }
            events.append({"type": "response.output_item.done", "output_index": out_idx, "item": item})
            final_output_by_index[out_idx] = item

        tool_items_in_order.clear()
        tool_items_by_id.clear()
        return events

    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "in_progress",
        "output": [],
        "error": None,
    }

    # Initial events
    yield sse({"type": "response.created", "response": base_response})
    yield sse({"type": "response.in_progress", "response": base_response})
    for ev in _start_message_item():
        yield sse(ev)

    try:
        async for data in _iter_dify_sse_json(rsp):
            event_type = data.get("event")

            if event_type in ("message", "agent_message"):
                answer_delta = data.get("answer", "")
                if answer_delta:
                    if not message_open:
                        for ev in _start_message_item():
                            yield sse(ev)

                    if not isinstance(answer_delta, str):
                        answer_delta = str(answer_delta)
                    accumulated_text += answer_delta
                    yield sse(
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": answer_delta,
                        }
                    )

            elif event_type == "tool_calls":
                dify_tool_calls = data.get("tool_calls", [])
                if isinstance(dify_tool_calls, list) and dify_tool_calls:
                    normalized = _normalize_tool_calls(dify_tool_calls)
                    if message_open:
                        for ev in _close_message_item(accumulated_text):
                            yield sse(ev)
                    for ev in _start_tool_items(normalized):
                        yield sse(ev)

            elif event_type == "message_end":
                meta_usage = (data.get("metadata") or {}).get("usage") if isinstance(data.get("metadata"), dict) else None
                if isinstance(meta_usage, dict):
                    usage_obj = {
                        "input_tokens": meta_usage.get("prompt_tokens", 0),
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": meta_usage.get("completion_tokens", 0),
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": meta_usage.get("total_tokens", 0),
                    }

                extracted_tool_calls: List[Dict[str, Any]] = []
                if TOOL_SUPPORT and not tool_items_in_order and accumulated_text:
                    extracted = extract_tool_invocations(accumulated_text)
                    if extracted:
                        extracted_tool_calls = _normalize_tool_calls(extracted)

                if message_open:
                    for ev in _close_message_item(accumulated_text):
                        yield sse(ev)

                if extracted_tool_calls:
                    for ev in _start_tool_items(extracted_tool_calls):
                        yield sse(ev)

                for ev in _finish_tool_items():
                    yield sse(ev)

                completed_response: Dict[str, Any] = {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": model,
                    "status": "completed",
                    "output": [final_output_by_index[idx] for idx in sorted(final_output_by_index.keys())],
                    "usage": usage_obj,
                    "error": None,
                }
                yield sse({"type": "response.completed", "response": completed_response})
                yield "data: [DONE]\n\n"
                return

    except Exception as e:
        logger.error(f"Responses stream error: {e}")
        failed_response: Dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "failed",
            "output": [],
            "usage": usage_obj,
            "error": {"code": 500, "message": str(e)},
        }
        yield sse({"type": "response.completed", "response": failed_response})
        yield "data: [DONE]\n\n"


async def stream_claude_message(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
) -> AsyncGenerator[str, None]:
    accumulated_text = ""
    tool_calls_sent = False
    content_index = 0
    text_block_open = True
    current_text_index = 0

    def sse(event_name: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {ujson.dumps(payload, ensure_ascii=False)}\n\n"

    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield sse(
        "content_block_start",
        {"type": "content_block_start", "index": current_text_index, "content_block": {"type": "text", "text": ""}},
    )

    usage_obj: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    try:
        async for data in _iter_dify_sse_json(rsp):
            event_type = data.get("event")

            if event_type in ("message", "agent_message"):
                answer_delta = data.get("answer", "")
                if answer_delta:
                    if not isinstance(answer_delta, str):
                        answer_delta = str(answer_delta)

                    if not text_block_open:
                        current_text_index = content_index
                        text_block_open = True
                        yield sse(
                            "content_block_start",
                            {"type": "content_block_start", "index": current_text_index, "content_block": {"type": "text", "text": ""}},
                        )

                    accumulated_text += answer_delta
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": current_text_index,
                            "delta": {"type": "text_delta", "text": answer_delta},
                        },
                    )

            elif event_type == "tool_calls":
                dify_tool_calls = data.get("tool_calls", [])
                if isinstance(dify_tool_calls, list) and dify_tool_calls:
                    normalized = _normalize_tool_calls(dify_tool_calls)
                    tool_calls_sent = True

                    if text_block_open:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
                        text_block_open = False
                        content_index = current_text_index + 1

                    for tc in normalized:
                        call_id = tc.get("id") or f"call_{fast_uuid()}"
                        func = tc.get("function") or {}
                        name = (func or {}).get("name") or ""
                        args = (func or {}).get("arguments") or ""

                        tool_idx = content_index
                        yield sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_idx,
                                "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
                            },
                        )
                        if isinstance(args, str) and args:
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": tool_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": args},
                                },
                            )
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": tool_idx})
                        content_index += 1

            elif event_type == "message_end":
                meta_usage = (data.get("metadata") or {}).get("usage") if isinstance(data.get("metadata"), dict) else None
                if isinstance(meta_usage, dict):
                    usage_obj = {
                        "input_tokens": meta_usage.get("prompt_tokens", 0),
                        "output_tokens": meta_usage.get("completion_tokens", 0),
                    }

                extracted_tool_calls: List[Dict[str, Any]] = []
                if TOOL_SUPPORT and not tool_calls_sent and accumulated_text:
                    extracted = extract_tool_invocations(accumulated_text)
                    if extracted:
                        tool_calls_sent = True
                        extracted_tool_calls = _normalize_tool_calls(extracted)

                if extracted_tool_calls:
                    if text_block_open:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
                        text_block_open = False
                        content_index = current_text_index + 1

                    for tc in extracted_tool_calls:
                        call_id = tc.get("id") or f"call_{fast_uuid()}"
                        func = tc.get("function") or {}
                        name = (func or {}).get("name") or ""
                        args = (func or {}).get("arguments") or ""

                        tool_idx = content_index
                        yield sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_idx,
                                "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
                            },
                        )
                        if isinstance(args, str) and args:
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": tool_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": args},
                                },
                            )
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": tool_idx})
                        content_index += 1

                if text_block_open:
                    yield sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
                    text_block_open = False

                stop_reason = "tool_use" if tool_calls_sent else "end_turn"
                yield sse(
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage_obj},
                )
                yield sse("message_stop", {"type": "message_stop"})
                return

    except Exception as e:
        logger.error(f"Claude stream error: {e}")
        yield sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage_obj})
        yield sse("message_stop", {"type": "message_stop"})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        openai_request = ujson.loads(request_body)
        model = openai_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        messages = openai_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        n = openai_request.get("n")
        if n is not None and n != 1:
            raise OpenAIHTTPError(400, "Only 'n=1' is supported", param="n", code="invalid_request_error")

        conversation_id = request.headers.get("X-Dify-Conversation-Id") or openai_request.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            conversation_id = None

        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise OpenAIHTTPError(
                404,
                f"The model '{model}' does not exist",
                type="invalid_request_error",
                code="model_not_found",
                param="model",
            )

        dify_req = transform_openai_to_dify(openai_request, conversation_id)
        if not dify_req:
            raise OpenAIHTTPError(400, "Invalid format", code="invalid_request_error", param=None)

        stream = openai_request.get("stream", False)
        if stream:
            stream_options = openai_request.get("stream_options") or {}
            stream_include_usage = bool((stream_options or {}).get("include_usage")) if isinstance(stream_options, dict) else False

            headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
            endpoint = f"{DIFY_API_BASE}/chat-messages"
            message_id = f"chatcmpl-{fast_uuid()}"

            upstream_cm = model_manager.client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(dify_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                try:
                    dify_error = ujson.loads(body or b"{}")
                    error_message = dify_error.get("message") or (body.decode("utf-8", errors="ignore") if body else "Upstream error")
                    error_code = dify_error.get("code", "unknown_error")
                except Exception:
                    error_message = body.decode("utf-8", errors="ignore") if body else "Upstream error"
                    error_code = "unknown_error"

                if upstream_rsp.status_code == 429:
                    err_type = "rate_limit_error"
                elif upstream_rsp.status_code >= 500:
                    err_type = "server_error"
                else:
                    err_type = "invalid_request_error"
                raise OpenAIHTTPError(upstream_rsp.status_code, error_message, type=err_type, code=error_code, param=None)

            return StreamingResponse(
                stream_openai_response(
                    upstream_rsp,
                    model=model,
                    message_id=message_id,
                    stream_include_usage=stream_include_usage,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        # 非流式
        headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
        endpoint = f"{DIFY_API_BASE}/chat-messages"

        resp = await model_manager.client.post(
            endpoint,
            content=ujson.dumps(dify_req),
            headers=headers,
            timeout=TIMEOUT
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
                openai_error_type = "rate_limit_error"
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
                        "param": None,
                        "code": error_code
                    }
                }
            )

        dify_resp = ujson.loads(resp.content)
        openai_resp, resp_conversation_id = transform_dify_to_openai_response(dify_resp, model)

        response_headers = {"access-control-allow-origin": "*"}
        if resp_conversation_id:
            response_headers["X-Dify-Conversation-Id"] = str(resp_conversation_id)
        return UJSONResponse(content=openai_resp, headers=response_headers)

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)


@app.post("/v1/responses")
async def responses_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        responses_request = ujson.loads(request_body)

        model = responses_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        if "input" not in responses_request:
            raise OpenAIHTTPError(400, "Missing required parameter: 'input'", param="input", code="missing_required_parameter")

        conversation_id = request.headers.get("X-Dify-Conversation-Id") or responses_request.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            conversation_id = None

        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise OpenAIHTTPError(
                404,
                f"The model '{model}' does not exist",
                type="invalid_request_error",
                code="model_not_found",
                param="model",
            )

        dify_req = transform_openai_responses_to_dify(responses_request, conversation_id)
        if not dify_req:
            raise OpenAIHTTPError(400, "Invalid format", code="invalid_request_error", param=None)

        stream = bool(responses_request.get("stream", False))
        if stream:
            headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
            endpoint = f"{DIFY_API_BASE}/chat-messages"
            response_id = f"resp_{fast_uuid()}"

            upstream_cm = model_manager.client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(dify_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                try:
                    dify_error = ujson.loads(body or b"{}")
                    error_message = dify_error.get("message") or (body.decode("utf-8", errors="ignore") if body else "Upstream error")
                    error_code = dify_error.get("code", "unknown_error")
                except Exception:
                    error_message = body.decode("utf-8", errors="ignore") if body else "Upstream error"
                    error_code = "unknown_error"

                if upstream_rsp.status_code == 429:
                    err_type = "rate_limit_error"
                elif upstream_rsp.status_code >= 500:
                    err_type = "server_error"
                else:
                    err_type = "invalid_request_error"
                raise OpenAIHTTPError(upstream_rsp.status_code, error_message, type=err_type, code=error_code, param=None)

            return StreamingResponse(
                stream_openai_responses(upstream_rsp, model=model, response_id=response_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
        endpoint = f"{DIFY_API_BASE}/chat-messages"
        resp = await model_manager.client.post(
            endpoint,
            content=ujson.dumps(dify_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            try:
                dify_error = ujson.loads(resp.text)
                error_message = dify_error.get("message", resp.text)
                error_code = dify_error.get("code", "unknown_error")
            except Exception:
                error_message = resp.text
                error_code = "unknown_error"

            if resp.status_code == 503:
                openai_error_type = "server_error"
                error_message = f"The model is currently overloaded. Please try again later. (Original: {error_message})"
            elif resp.status_code >= 500:
                openai_error_type = "server_error"
            elif resp.status_code == 429:
                openai_error_type = "rate_limit_error"
            elif resp.status_code >= 400:
                openai_error_type = "invalid_request_error"
            else:
                openai_error_type = "api_error"

            raise HTTPException(
                status_code=resp.status_code,
                detail={"error": {"message": error_message, "type": openai_error_type, "param": None, "code": error_code}},
            )

        dify_resp = ujson.loads(resp.content)
        out_resp, resp_conversation_id = transform_dify_to_openai_responses(dify_resp, model)

        response_headers = {"access-control-allow-origin": "*"}
        if resp_conversation_id:
            response_headers["X-Dify-Conversation-Id"] = str(resp_conversation_id)
        return UJSONResponse(content=out_resp, headers=response_headers)

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)


@app.post("/v1/messages")
async def claude_messages_api(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request_body = await request.body()
        claude_request = ujson.loads(request_body)

        model = claude_request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise OpenAIHTTPError(400, "Missing required parameter: 'model'", param="model", code="missing_required_parameter")

        messages = claude_request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OpenAIHTTPError(400, "Missing required parameter: 'messages'", param="messages", code="missing_required_parameter")

        conversation_id = request.headers.get("X-Dify-Conversation-Id") or claude_request.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            conversation_id = None

        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            raise OpenAIHTTPError(
                404,
                f"The model '{model}' does not exist",
                type="invalid_request_error",
                code="model_not_found",
                param="model",
            )

        dify_req = transform_claude_messages_to_dify(claude_request, conversation_id)
        if not dify_req:
            raise OpenAIHTTPError(400, "Invalid format", code="invalid_request_error", param=None)

        stream = bool(claude_request.get("stream", False))
        if stream:
            headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
            endpoint = f"{DIFY_API_BASE}/chat-messages"
            message_id = f"msg_{fast_uuid()}"

            upstream_cm = model_manager.client.stream(
                "POST",
                endpoint,
                content=ujson.dumps(dify_req, ensure_ascii=False),
                headers=headers,
                timeout=TIMEOUT,
            )
            upstream_rsp = await upstream_cm.__aenter__()
            if upstream_rsp.status_code != 200:
                body = await upstream_rsp.aread()
                await upstream_cm.__aexit__(None, None, None)
                try:
                    dify_error = ujson.loads(body or b"{}")
                    error_message = dify_error.get("message") or (body.decode("utf-8", errors="ignore") if body else "Upstream error")
                    error_code = dify_error.get("code", "unknown_error")
                except Exception:
                    error_message = body.decode("utf-8", errors="ignore") if body else "Upstream error"
                    error_code = "unknown_error"

                if upstream_rsp.status_code == 429:
                    err_type = "rate_limit_error"
                elif upstream_rsp.status_code >= 500:
                    err_type = "server_error"
                else:
                    err_type = "invalid_request_error"
                raise OpenAIHTTPError(upstream_rsp.status_code, error_message, type=err_type, code=error_code, param=None)

            return StreamingResponse(
                stream_claude_message(upstream_rsp, model=model, message_id=message_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
                background=BackgroundTask(upstream_cm.__aexit__, None, None, None),
            )

        headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
        endpoint = f"{DIFY_API_BASE}/chat-messages"
        resp = await model_manager.client.post(
            endpoint,
            content=ujson.dumps(dify_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            try:
                dify_error = ujson.loads(resp.text)
                error_message = dify_error.get("message", resp.text)
                error_code = dify_error.get("code", "unknown_error")
            except Exception:
                error_message = resp.text
                error_code = "unknown_error"

            if resp.status_code == 503:
                openai_error_type = "server_error"
                error_message = f"The model is currently overloaded. Please try again later. (Original: {error_message})"
            elif resp.status_code >= 500:
                openai_error_type = "server_error"
            elif resp.status_code == 429:
                openai_error_type = "rate_limit_error"
            elif resp.status_code >= 400:
                openai_error_type = "invalid_request_error"
            else:
                openai_error_type = "api_error"

            raise HTTPException(
                status_code=resp.status_code,
                detail={"error": {"message": error_message, "type": openai_error_type, "param": None, "code": error_code}},
            )

        dify_resp = ujson.loads(resp.content)
        out_resp, resp_conversation_id = transform_dify_to_claude_message(dify_resp, model)

        response_headers = {"access-control-allow-origin": "*"}
        if resp_conversation_id:
            response_headers["X-Dify-Conversation-Id"] = str(resp_conversation_id)
        return UJSONResponse(content=out_resp, headers=response_headers)

    except OpenAIHTTPError:
        raise
    except HTTPException:
        raise
    except ujson.JSONDecodeError:
        raise OpenAIHTTPError(400, "Invalid JSON format in request", type="invalid_request_error", code="invalid_json", param=None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise OpenAIHTTPError(500, str(e), type="server_error", code="internal_error", param=None)


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    if not model_manager.name_to_api_key:
        await model_manager.refresh_model_info()

    models = model_manager.get_available_models()
    return UJSONResponse(content={"object": "list", "data": models}, headers={"access-control-allow-origin": "*"})


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, api_key: str = Depends(verify_api_key)):
    if not model_manager.name_to_api_key:
        await model_manager.refresh_model_info()

    dify_key = model_manager.get_api_key(model_id)
    if not dify_key:
        raise OpenAIHTTPError(
            404,
            f"The model '{model_id}' does not exist",
            type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    timestamp = int(time.time())
    return UJSONResponse(
        content={"id": model_id, "object": "model", "created": timestamp, "owned_by": "dify"},
        headers={"access-control-allow-origin": "*"},
    )


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

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
VALID_API_KEYS_SET = frozenset(VALID_API_KEYS)  # 使用set加速查找，O(1)
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
ZERO_WIDTH_CHARS_SET = frozenset(ZERO_WIDTH_CHARS)  # 使用set加速查找

# 工具支持配置
TOOL_SUPPORT = True
SCAN_LIMIT = 8000  # 工具调用扫描限制

# 预计算的常量字符串，避免重复拼接
SYSTEM_DEFAULT_PROMPT = "你是一个有用的助手。"
TOOL_USE_HINT = "\n\n请根据需要使用提供的工具函数。"
TOOL_RESULT_PREFIX = "工具 "
TOOL_RESULT_SUFFIX = " 返回结果:\n```json\n"
TOOL_RESULT_END = "\n```"
TOOL_COMPLETE_SUFFIX = " 执行完成"

# 预编译的UUID格式，避免重复字符串操作
import secrets
def fast_uuid() -> str:
    """快速生成UUID hex（比uuid.uuid4().hex更快）"""
    return secrets.token_hex(16)


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
    """简化的conversation_id编码（优化版本）"""
    if not conversation_id:
        return ""

    # 使用简单的base64编码 + 零宽字符
    encoded = base64.b64encode(conversation_id.encode('utf-8')).decode('ascii')
    # 用零宽字符替换部分字符使其不可见 - 使用列表拼接优化
    result = []
    zero_width_len = len(ZERO_WIDTH_CHARS)
    for i, char in enumerate(encoded):
        if i % 4 == 0:  # 每4个字符插入一个零宽字符
            result.append(ZERO_WIDTH_CHARS[i % zero_width_len])
        result.append(char)
    return ''.join(result)

def decode_conversation_id(content: str) -> Optional[str]:
    """简化的conversation_id解码（优化版本）"""
    if not content:
        return None

    try:
        # 移除零宽字符 - 使用列表推导式和join优化
        cleaned = ''.join(char for char in content if char not in ZERO_WIDTH_CHARS_SET)

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
    if not key or key not in VALID_API_KEYS_SET:  # 使用set加速验证
        raise HTTPUnauthorized("Invalid API key")
    return key


# 业务函数 - JSON转换
def transform_openai_to_dify(openai_request: Dict, endpoint: str) -> Optional[Dict]:
    if endpoint != "/chat/completions" or not openai_request.get("messages"):
        return None

    messages = openai_request["messages"]
    stream = openai_request.get("stream", False)
    tools = openai_request.get("tools", [])
    tool_choice = openai_request.get("tool_choice", "auto")

    # 处理工具注入到消息中
    processed_messages = []
    if tools and TOOL_SUPPORT and tool_choice != "none":
        tools_prompt = generate_tool_prompt(tools)
        has_system = False

        # 一次遍历同时检查和处理
        for m in messages:
            if m.get("role") == "system":
                has_system = True
                mm = dict(m)
                content = content_to_string(mm.get("content", ""))
                mm["content"] = content + tools_prompt
                processed_messages.append(mm)
            else:
                processed_messages.append(m)

        # 如果没有system消息，在开头添加
        if not has_system:
            processed_messages.insert(0, {"role": "system", "content": SYSTEM_DEFAULT_PROMPT + tools_prompt})

        # 添加工具选择提示
        if tool_choice in ("required", "auto"):
            if processed_messages and processed_messages[-1].get("role") == "user":
                last = dict(processed_messages[-1])
                content = content_to_string(last.get("content", ""))
                last["content"] = content + TOOL_USE_HINT
                processed_messages[-1] = last
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname and processed_messages and processed_messages[-1].get("role") == "user":
                last = dict(processed_messages[-1])
                content = content_to_string(last.get("content", ""))
                last["content"] = content + f"\n\n请使用 {fname} 函数来处理这个请求。"
                processed_messages[-1] = last
    else:
        processed_messages = list(messages)

    # 处理tool/function消息
    final_messages = []
    for m in processed_messages:
        role = m.get("role")
        if role in ("tool", "function"):
            tool_name = m.get("name", "unknown")
            tool_content = content_to_string(m.get("content", ""))
            if isinstance(tool_content, dict):
                tool_content = ujson.dumps(tool_content, ensure_ascii=False)

            # 使用预计算的常量字符串
            if tool_content.strip():
                content = TOOL_RESULT_PREFIX + tool_name + TOOL_RESULT_SUFFIX + tool_content + TOOL_RESULT_END
            else:
                content = TOOL_RESULT_PREFIX + tool_name + TOOL_COMPLETE_SUFFIX

            final_messages.append({
                "role": "assistant",
                "content": content,
            })
        else:
            final_msg = dict(m)
            content = content_to_string(final_msg.get("content", ""))
            final_msg["content"] = content
            final_messages.append(final_msg)

    # 现在从处理后的消息中提取system和user query
    system_content = ""
    user_query = ""

    # 提取system消息
    for m in final_messages:
        if m.get("role") == "system":
            system_content = m.get("content", "")
            break

    # 处理最后一条用户消息
    last_message = final_messages[-1] if final_messages else {}
    if last_message.get("role") == "user":
        user_query = last_message.get("content", "")

    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = None
        # 反向查找assistant消息，提前退出
        if len(final_messages) > 1:
            for m in reversed(final_messages[:-1]):
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
        if len(final_messages) > 1:
            history_msg = []
            has_system = any(m.get("role") == "system" for m in final_messages[:-1])
            for m in final_messages[:-1]:
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

    return dify_request


def content_to_string(content: Any) -> str:
    """将各种格式的content转换为字符串"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return ""


def generate_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """生成工具注入提示"""
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

    prompt_template = (
        "\n\n# AVAILABLE FUNCTIONS\n" + "\n\n---\n".join(tool_definitions) + "\n\n# USAGE INSTRUCTIONS\n"
        "When you need to execute a function, respond ONLY with a JSON object containing tool_calls:\n"
        "```json\n"
        "{\n"
        '  "tool_calls": [\n'
        "    {\n"
        '      "id": "call_xxx",\n'
        '      "type": "function",\n'
        '      "function": {\n'
        '        "name": "function_name",\n'
        '        "arguments": "{\\"param1\\": \\"value1\\"}"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Important: No explanatory text before or after the JSON. The 'arguments' field must be a JSON string, not an object.\n"
    )

    return prompt_template


# 工具提取的正则模式
TOOL_CALL_FENCE_PATTERN = re.compile(r"```json\s*(\{[^`]+\})\s*```", re.DOTALL)
FUNCTION_CALL_PATTERN = re.compile(r"调用函数\s*[：:]\s*([\w\-\.]+)\s*(?:参数|arguments)[：:]\s*(\{.*?\})", re.DOTALL)

# LRU缓存用于工具调用提取结果
@lru_cache(maxsize=128)
def _cached_extract_tool_invocations(text_hash: int, text_len: int) -> Optional[str]:
    """内部缓存函数，返回JSON字符串"""
    return None  # 实际由extract_tool_invocations填充


def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """标准化工具调用，确保arguments字段为字符串"""
    for tc in tool_calls:
        if "function" in tc:
            func = tc["function"]
            if "arguments" in func:
                if not isinstance(func["arguments"], str):
                    func["arguments"] = ujson.dumps(func["arguments"], ensure_ascii=False)
    return tool_calls


def _find_balanced_json(text: str, start_pos: int = 0) -> Optional[tuple[str, int]]:
    """使用括号平衡查找JSON对象，返回(json_str, end_pos)"""
    i = start_pos
    while i < len(text):
        if text[i] == '{':
            brace_count = 1
            j = i + 1
            in_string = False
            escape_next = False

            while j < len(text) and brace_count > 0:
                char = text[j]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                j += 1

            if brace_count == 0:
                return text[i:j], j
            i = j
        else:
            i += 1
    return None


def extract_tool_invocations(text: str) -> Optional[List[Dict[str, Any]]]:
    """从响应文本中提取工具调用（优化版本）"""
    if not text:
        return None

    scannable_text = text[:SCAN_LIMIT]

    # 尝试1: 从JSON代码块提取
    json_blocks = TOOL_CALL_FENCE_PATTERN.findall(scannable_text)
    for json_block in json_blocks:
        try:
            parsed_data = ujson.loads(json_block)
            tool_calls = parsed_data.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                return _normalize_tool_calls(tool_calls)
        except (ujson.JSONDecodeError, AttributeError):
            continue

    # 尝试2: 使用括号平衡方法提取内联JSON
    pos = 0
    while pos < len(scannable_text):
        result = _find_balanced_json(scannable_text, pos)
        if not result:
            break

        json_str, end_pos = result
        try:
            parsed_data = ujson.loads(json_str)
            tool_calls = parsed_data.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                return _normalize_tool_calls(tool_calls)
        except (ujson.JSONDecodeError, AttributeError):
            pass

        pos = end_pos

    # 尝试3: 解析自然语言函数调用
    natural_lang_match = FUNCTION_CALL_PATTERN.search(scannable_text)
    if natural_lang_match:
        function_name = natural_lang_match.group(1).strip()
        arguments_str = natural_lang_match.group(2).strip()
        try:
            ujson.loads(arguments_str)
            return [
                {
                    "id": f"call_{int(time.time() * 1000000)}",
                    "type": "function",
                    "function": {"name": function_name, "arguments": arguments_str},
                }
            ]
        except ujson.JSONDecodeError:
            pass

    return None


def remove_tool_json_content(text: str) -> str:
    """从响应文本中移除工具JSON内容（优化版本）"""
    def remove_tool_call_block(match: re.Match) -> str:
        json_content = match.group(1)
        try:
            parsed_data = ujson.loads(json_content)
            if "tool_calls" in parsed_data:
                return ""
        except (ujson.JSONDecodeError, AttributeError):
            pass
        return match.group(0)

    cleaned_text = TOOL_CALL_FENCE_PATTERN.sub(remove_tool_call_block, text)

    # 移除内联工具JSON - 使用共享的括号平衡函数
    result = []
    pos = 0

    while pos < len(cleaned_text):
        # 检查是否遇到JSON对象
        if cleaned_text[pos] == '{':
            json_result = _find_balanced_json(cleaned_text, pos)
            if json_result:
                json_str, end_pos = json_result
                try:
                    parsed = ujson.loads(json_str)
                    if "tool_calls" in parsed:
                        # 跳过这个工具调用JSON
                        pos = end_pos
                        continue
                except:
                    pass

        result.append(cleaned_text[pos])
        pos += 1

    return ''.join(result).strip()


def stream_transform(dify_response: Dict, model: str, stream: bool) -> Dict:
    if stream:
        return dify_response

    answer = dify_response.get("answer", "")
    tool_calls = []

    # 先尝试从answer中提取工具调用
    if answer and TOOL_SUPPORT:
        extracted_tools = extract_tool_invocations(answer)
        if extracted_tools:
            tool_calls = extracted_tools
            # 从answer中移除工具JSON
            answer = remove_tool_json_content(answer)

    # 如果Dify响应中包含tool_calls（原生支持）
    if "tool_calls" in dify_response and not tool_calls:
        tool_calls = [
            {
                "id": tc.get("id", f"call_{fast_uuid()}"),
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
    message_content = {"role": "assistant", "content": answer if not tool_calls else None}
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
    message_id = f"chatcmpl-{fast_uuid()[:8]}"  # 使用更快的UUID生成

    # 用于累积内容以检测工具调用
    accumulated_content = ""
    detected_tools = []
    tool_calls_sent = False
    last_extraction_length = 0  # 跟踪上次提取时的内容长度

    def make_chunk(content: str = "", tool_calls: List[Dict] = None, final=False, role: str = None):
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

        if role:
            chunk["choices"][0]["delta"]["role"] = role
        if content:
            chunk["choices"][0]["delta"]["content"] = content
        if tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
        if final:
            chunk["choices"][0]["finish_reason"] = "tool_calls" if tool_calls or tool_calls_sent else "stop"

        return f"data: {ujson.dumps(chunk)}\n\n"

    async with model_manager.client.stream('POST', endpoint, json=dify_request, headers=headers) as rsp:
        if rsp.status_code != 200:
            yield make_chunk("Stream connection failed", final=True)
            yield "data: [DONE]\n\n"
            return

        first_chunk = True
        buffer = bytearray()  # 使用bytearray代替bytes，避免重复分配
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
                        answer = data.get("answer", "")
                        if answer:
                            accumulated_content += answer

                            # 尝试检测工具调用 - 仅在内容显著增长时检测
                            if TOOL_SUPPORT and not tool_calls_sent:
                                content_length = len(accumulated_content)
                                # 内容增长至少100字符才重新检测，避免频繁调用
                                if content_length - last_extraction_length >= 100:
                                    extracted = extract_tool_invocations(accumulated_content)
                                    last_extraction_length = content_length

                                    if extracted:
                                        # 检测到工具调用
                                        detected_tools = extracted
                                        tool_calls_sent = True

                                        # 发送角色
                                        if first_chunk:
                                            yield make_chunk(role="assistant")
                                            first_chunk = False

                                        # 发送工具调用
                                        yield make_chunk(tool_calls=detected_tools)
                                        continue

                            # 正常内容流式输出
                            if not tool_calls_sent:
                                if first_chunk:
                                    yield make_chunk(content=answer, role="assistant")
                                    first_chunk = False
                                else:
                                    yield make_chunk(answer)

                    elif event_type == "agent_thought":
                        thought = data.get("thought", "")
                        if thought and not tool_calls_sent:
                            accumulated_content += thought
                            if first_chunk:
                                yield make_chunk(content=thought, role="assistant")
                                first_chunk = False
                            else:
                                yield make_chunk(thought)

                    elif event_type == "tool_calls":
                        # Dify原生工具调用支持
                        tool_calls = data.get("tool_calls", [])
                        if tool_calls:
                            tool_calls_sent = True
                            openai_tools = [
                                {
                                    "id": tc.get("id", f"call_{fast_uuid()[:8]}"),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": ujson.dumps(tc.get("function", {}).get("arguments", {}))
                                    }
                                }
                                for tc in tool_calls
                            ]
                            if first_chunk:
                                yield make_chunk(role="assistant")
                                first_chunk = False
                            yield make_chunk(tool_calls=openai_tools)

                    elif event_type == "message_end":
                        # 最后检查一次是否有工具调用
                        if TOOL_SUPPORT and not tool_calls_sent and accumulated_content:
                            extracted = extract_tool_invocations(accumulated_content)
                            if extracted:
                                detected_tools = extracted
                                tool_calls_sent = True
                                if first_chunk:
                                    yield make_chunk(role="assistant")
                                yield make_chunk(tool_calls=detected_tools)

                        yield make_chunk("", final=True)
                        yield "data: [DONE]\n\n"
                        return

                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
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

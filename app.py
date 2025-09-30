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
TOOL_DETECTION_THRESHOLD = 500  # 工具检测阈值，从100增加到500减少检测频率

# 预计算的常量字符串，避免重复拼接
SYSTEM_DEFAULT_PROMPT = "你是一个有用的助手。"
TOOL_USE_HINT = "\n\n<INSTRUCTION>You MUST use one of the available tools to answer this request. Review the tool definitions carefully and ensure ALL required parameters are included with EXACT parameter names.</INSTRUCTION>"
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

@lru_cache(maxsize=512)
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
        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "system":
                has_system = True
                content = content_to_string(m.get("content", ""))
                processed_messages.append({"role": "system", "content": content + tools_prompt})
            else:
                processed_messages.append(m)

        # 如果没有system消息，在开头添加
        if not has_system:
            processed_messages.insert(0, {"role": "system", "content": SYSTEM_DEFAULT_PROMPT + tools_prompt})

        # 添加工具选择提示
        if tool_choice in ("required", "auto"):
            if processed_messages and processed_messages[-1].get("role") == "user":
                last = processed_messages[-1]
                content = content_to_string(last.get("content", ""))
                processed_messages[-1] = {"role": "user", "content": content + TOOL_USE_HINT}
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname and processed_messages and processed_messages[-1].get("role") == "user":
                last = processed_messages[-1]
                content = content_to_string(last.get("content", ""))
                processed_messages[-1] = {"role": "user", "content": content + f"\n\n<INSTRUCTION>You MUST use the '{fname}' function to handle this request. Include ALL required parameters with EXACT names.</INSTRUCTION>"}
    else:
        processed_messages = messages

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

            final_messages.append({"role": "assistant", "content": content})
        else:
            content = content_to_string(m.get("content", ""))
            final_messages.append({"role": role, "content": content})

    # 一次遍历提取所需信息
    system_content = ""
    user_query = ""
    conversation_id = None

    # 提取最后一条用户消息
    if final_messages:
        last_message = final_messages[-1]
        if last_message.get("role") == "user":
            user_query = last_message.get("content", "")

    # 单次遍历提取system和conversation_id
    if CONVERSATION_MEMORY_MODE == 2 and len(final_messages) > 1:
        # 从前往后找system，从后往前找assistant（跳过最后一条用户消息）
        for i, m in enumerate(final_messages):
            role = m.get("role")
            if role == "system" and not system_content:
                system_content = m.get("content", "")
            # 从倒数第二条开始往前找assistant
            if not conversation_id and i < len(final_messages) - 1:
                rev_idx = len(final_messages) - 2 - i
                if rev_idx >= 0:
                    rev_m = final_messages[rev_idx]
                    if rev_m.get("role") == "assistant":
                        conversation_id = decode_conversation_id(rev_m.get("content", ""))
            if system_content and conversation_id:
                break
    else:
        # 非mode 2只需要找system
        for m in final_messages:
            if m.get("role") == "system":
                system_content = m.get("content", "")
                break

    if CONVERSATION_MEMORY_MODE == 2:

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
        # history模式 - 优化：单次遍历检查和构建
        if len(final_messages) > 1:
            history_msg = []
            has_system = False

            # 单次遍历同时检查system和构建历史
            for m in final_messages[:-1]:
                role = m.get("role")
                if role == "system":
                    has_system = True
                content = m.get("content")
                if role and content:
                    history_msg.append(f"{role}: {content}")

            if system_content and not has_system:
                history_msg.insert(0, f"system: {system_content}")

            if history_msg:
                user_query = f"<history>\n{'\n\n'.join(history_msg)}\n</history>\n\n用户当前问题: {user_query}"
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
    """将各种格式的content转换为字符串（优化版本）"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 使用列表推导式 + 生成器优化
        parts = [
            p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else p
            for p in content
            if (isinstance(p, dict) and p.get("type") == "text") or isinstance(p, str)
        ]
        return " ".join(parts) if parts else ""
    return ""


def generate_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """生成工具注入提示（精确版本 - 提高调用成功率）"""
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

        parameter_properties = parameters.get("properties", {}) or {}
        required_parameters = set(parameters.get("required", []) or [])

        # 构建工具定义
        tool_info = [
            f"## Function: {function_name}",
            f"Description: {function_description}",
        ]

        if parameter_properties:
            tool_info.append("\nParameters (EXACT names required):")
            for param_name, param_details in parameter_properties.items():
                param_type = (param_details or {}).get("type", "any")
                param_desc = (param_details or {}).get("description", "")
                is_required = param_name in required_parameters

                # 详细参数说明
                req_marker = "[REQUIRED]" if is_required else "[OPTIONAL]"
                tool_info.append(f"  • {req_marker} {param_name}: {param_type}")
                if param_desc:
                    tool_info.append(f"    → {param_desc}")

                # 对于 object 类型，显示其内部结构
                if param_type == "object":
                    obj_props = (param_details or {}).get("properties", {})
                    if obj_props:
                        tool_info.append(f"    Object structure:")
                        for obj_key, obj_val in obj_props.items():
                            obj_type = (obj_val or {}).get("type", "any")
                            tool_info.append(f"      - {obj_key}: {obj_type}")

                # 对于 array 类型，显示元素类型
                elif param_type == "array":
                    items_schema = (param_details or {}).get("items", {})
                    if items_schema:
                        items_type = items_schema.get("type", "any")
                        tool_info.append(f"    Array of: {items_type}")

        # 为每个工具生成具体示例
        def generate_example_value(param_details, depth=0):
            """递归生成示例值"""
            if depth > 3:  # 防止无限递归
                return {}

            param_type = (param_details or {}).get("type", "string")

            if param_type == "string":
                return "example_string"
            elif param_type == "number":
                return 42
            elif param_type == "integer":
                return 42
            elif param_type == "boolean":
                return True
            elif param_type == "array":
                items_schema = (param_details or {}).get("items", {})
                if items_schema:
                    return [generate_example_value(items_schema, depth + 1)]
                return ["item1", "item2"]
            elif param_type == "object":
                properties = (param_details or {}).get("properties", {})
                if properties:
                    obj = {}
                    for prop_name, prop_details in properties.items():
                        obj[prop_name] = generate_example_value(prop_details, depth + 1)
                    return obj
                return {"key": "value"}
            else:
                return "value"

        example_args = {}
        for param_name, param_details in parameter_properties.items():
            example_args[param_name] = generate_example_value(param_details)

        if example_args:
            # 生成 JSON 字符串，需要转义引号
            args_json = ujson.dumps(example_args, ensure_ascii=False)
            # 转义双引号用于嵌入到字符串中
            args_json_escaped = args_json.replace('"', '\\"')

            tool_info.append(f"\nExample call for {function_name}:")
            tool_info.append('```json')
            tool_info.append('{')
            tool_info.append('  "tool_calls": [{')
            tool_info.append(f'    "id": "call_{function_name}_1",')
            tool_info.append('    "type": "function",')
            tool_info.append('    "function": {')
            tool_info.append(f'      "name": "{function_name}",')
            tool_info.append(f'      "arguments": "{args_json_escaped}"')
            tool_info.append('    }')
            tool_info.append('  }]')
            tool_info.append('}')
            tool_info.append('```')

        tool_definitions.append("\n".join(tool_info))

    if not tool_definitions:
        return ""

    # 构建最终提示
    return ''.join([
        "\n\n<AVAILABLE_TOOLS>\n",
        "\n\n---\n".join(tool_definitions),
        "\n\n</AVAILABLE_TOOLS>\n\n"
        "<CRITICAL_RULES>\n"
        "1. USE EXACT parameter names from the function definition above\n"
        "2. Include ALL [REQUIRED] parameters - missing parameters will cause errors\n"
        "3. Match parameter types exactly (string, number, boolean, array, object)\n"
        "4. The 'arguments' field MUST be a JSON STRING (with escaped quotes), NOT a JSON object\n"
        "5. Response format: ONLY output the JSON block, NO explanatory text before or after\n"
        "6. Each tool call needs a unique 'id' like 'call_functionname_1'\n"
        "</CRITICAL_RULES>\n\n"
        "<RESPONSE_FORMAT>\n"
        "```json\n"
        '{"tool_calls": [{"id": "call_xxx", "type": "function", "function": {"name": "exact_function_name", "arguments": "{\\"param\\": \\"value\\"}"}}}]}\n'
        "```\n"
        "</RESPONSE_FORMAT>\n"
    ])


# 工具提取的正则模式 - 优化：使用非贪婪匹配和更精确的模式
TOOL_CALL_FENCE_PATTERN = re.compile(r"```json\s*(\{[^`]+?\})\s*```", re.DOTALL)
FUNCTION_CALL_PATTERN = re.compile(r"调用函数\s*[：:]\s*([\w\-\.]+)\s*(?:参数|arguments)[：:]\s*(\{.+?\})", re.DOTALL)

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
    text_len = len(text)
    i = start_pos

    # 快速跳过非 '{' 字符
    while i < text_len and text[i] != '{':
        i += 1

    if i >= text_len:
        return None

    # 找到 '{'，开始括号平衡
    brace_count = 1
    j = i + 1
    in_string = False

    while j < text_len and brace_count > 0:
        char = text[j]

        if char == '\\' and in_string:
            # 跳过转义字符
            j += 2
            continue
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        j += 1

    if brace_count == 0:
        return text[i:j], j
    return None


def extract_tool_invocations(text: str) -> Optional[List[Dict[str, Any]]]:
    """从响应文本中提取工具调用（优化版本 - 提高准确率）"""
    if not text:
        return None

    scannable_text = text[:SCAN_LIMIT]

    # 快速检查：如果没有可能的工具调用标记，直接返回
    if 'tool_calls' not in scannable_text and '```json' not in scannable_text:
        return None

    # 尝试1: 从JSON代码块提取（最常见情况）
    if '```json' in scannable_text:
        json_blocks = TOOL_CALL_FENCE_PATTERN.findall(scannable_text)
        for json_block in json_blocks:
            try:
                parsed_data = ujson.loads(json_block)
                tool_calls = parsed_data.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # 验证工具调用的完整性
                    valid_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict) and tc.get("function") and tc.get("function").get("name"):
                            valid_calls.append(tc)
                    if valid_calls:
                        return _normalize_tool_calls(valid_calls)
            except (ujson.JSONDecodeError, AttributeError):
                continue

    # 尝试2: 使用括号平衡方法提取内联JSON（限制搜索范围）
    if 'tool_calls' in scannable_text and '{' in scannable_text:
        # 只搜索前2000字符以提高性能
        search_text = scannable_text[:2000]
        pos = 0
        attempts = 0
        max_attempts = 10  # 限制尝试次数

        while pos < len(search_text) and attempts < max_attempts:
            result = _find_balanced_json(search_text, pos)
            if not result:
                break

            json_str, end_pos = result
            attempts += 1

            try:
                parsed_data = ujson.loads(json_str)
                tool_calls = parsed_data.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # 验证工具调用的完整性
                    valid_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict) and tc.get("function") and tc.get("function").get("name"):
                            valid_calls.append(tc)
                    if valid_calls:
                        return _normalize_tool_calls(valid_calls)
            except (ujson.JSONDecodeError, AttributeError):
                pass

            pos = end_pos

    # 尝试3: 解析自然语言函数调用（只在包含特定关键字时）
    if '调用函数' in scannable_text or ('function' in scannable_text and 'arguments' in scannable_text):
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
    # 快速检查：如果没有可能的工具调用，直接返回
    if '```json' not in text and 'tool_calls' not in text:
        return text

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

    # 只有包含 tool_calls 时才处理内联JSON
    if 'tool_calls' not in cleaned_text:
        return cleaned_text.strip()

    # 移除内联工具JSON - 批量处理而非逐字符
    segments = []
    pos = 0
    max_scan = min(len(cleaned_text), 5000)  # 限制扫描范围

    while pos < max_scan:
        brace_pos = cleaned_text.find('{', pos)
        if brace_pos == -1:
            segments.append(cleaned_text[pos:])
            break

        # 添加 '{' 之前的内容
        segments.append(cleaned_text[pos:brace_pos])

        json_result = _find_balanced_json(cleaned_text, brace_pos)
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

        # 不是工具调用JSON，保留这个字符
        segments.append('{')
        pos = brace_pos + 1

    # 如果扫描范围后还有内容，添加上
    if pos < len(cleaned_text):
        segments.append(cleaned_text[pos:])

    return ''.join(segments).strip()


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
        tool_calls = []
        for tc in dify_response["tool_calls"]:
            func = tc.get("function") or {}
            tool_calls.append({
                "id": tc.get("id", f"call_{fast_uuid()}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": ujson.dumps(func.get("arguments", {}), ensure_ascii=False)
                }
            })

    # 处理agent_thoughts - 优化：直接找到第一个非空thought
    if not answer:
        agent_thoughts = dify_response.get("agent_thoughts")
        if agent_thoughts:
            for thought in reversed(agent_thoughts):
                if thought_content := thought.get("thought"):
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

    # 用于累积内容以检测工具调用 - 使用list代替字符串拼接
    accumulated_chunks = []
    accumulated_length = 0
    detected_tools = []
    tool_calls_sent = False
    last_extraction_length = 0  # 跟踪上次提取时的内容长度

    # 预计算timestamp以减少重复调用
    created_time = int(time.time())

    def make_chunk(content: str = "", tool_calls: List[Dict] = None, final=False, role: str = None):
        delta = {}
        if role:
            delta["role"] = role
        if content:
            delta["content"] = content
        if tool_calls:
            delta["tool_calls"] = tool_calls

        chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": "tool_calls" if final and (tool_calls or tool_calls_sent) else ("stop" if final else None)
            }]
        }

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
                            accumulated_chunks.append(answer)
                            accumulated_length += len(answer)

                            # 尝试检测工具调用 - 仅在内容显著增长时检测
                            if TOOL_SUPPORT and not tool_calls_sent:
                                # 内容增长至少TOOL_DETECTION_THRESHOLD字符才重新检测，避免频繁调用
                                if accumulated_length - last_extraction_length >= TOOL_DETECTION_THRESHOLD:
                                    accumulated_content = ''.join(accumulated_chunks)
                                    extracted = extract_tool_invocations(accumulated_content)
                                    last_extraction_length = accumulated_length

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
                            accumulated_chunks.append(thought)
                            accumulated_length += len(thought)
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
                        if TOOL_SUPPORT and not tool_calls_sent and accumulated_chunks:
                            accumulated_content = ''.join(accumulated_chunks)
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
        resp = await model_manager.client.post(
            f"{DIFY_API_BASE}/chat-messages",
            content=ujson.dumps(dify_req, ensure_ascii=False),
            headers={"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"},
            timeout=20.0
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        dify_resp = ujson.loads(resp.content)
        openai_resp = stream_transform(dify_resp, model, stream=False)

        return Response(
            content=ujson.dumps(openai_resp, ensure_ascii=False),
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

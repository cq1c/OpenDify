"""
OpenDify Lite — Roo Code (OpenAI Compatible) ↔ Dify 弱模型代理
针对弱 LLM 优化：精简提示词、XML 格式工具调用、会话令牌防注入、格式化 JSON 输出
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import json_repair
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").strip().upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("opendify")
for _lib in ("httpx", "httpcore", "uvicorn.access"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1").strip()
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

VALID_API_KEYS = frozenset(
    k.strip() for k in os.getenv("VALID_API_KEYS", "").split(",") if k.strip()
)
AUTH_MODE = os.getenv("AUTH_MODE", "required").strip().lower()

DIFY_API_BASE = (os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1") or "").rstrip("/")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "").strip()
DIFY_MODEL_NAME = os.getenv("DIFY_MODEL_NAME", "dify-model").strip()
TIMEOUT = float(os.getenv("TIMEOUT", "120"))

_model_map_raw = os.getenv("DIFY_MODEL_MAP", "").strip()
MODEL_KEY_MAP: Dict[str, str] = {}
if _model_map_raw:
    for part in _model_map_raw.split(","):
        part = part.strip()
        sep = ":" if ":" in part else "=" if "=" in part else ""
        if sep:
            name, key = part.split(sep, 1)
            if name.strip() and key.strip():
                MODEL_KEY_MAP[name.strip()] = key.strip()
if not MODEL_KEY_MAP and DIFY_API_KEY:
    MODEL_KEY_MAP[DIFY_MODEL_NAME] = DIFY_API_KEY


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return v.strip().lower() in ("1", "true", "yes") if v else default


STRIP_SYSTEM_AFTER_FIRST = _env_bool("STRIP_SYSTEM_AFTER_FIRST", False)
SYSTEM_PROMPT_MAX_LENGTH = int(os.getenv("SYSTEM_PROMPT_MAX_LENGTH", "0"))
SIMPLIFIED_TOOL_DEFS = _env_bool("SIMPLIFIED_TOOL_DEFS", True)
TOOL_DESC_MAX_LENGTH = int(os.getenv("TOOL_DESC_MAX_LENGTH", "120"))
ONLY_RECENT_MESSAGES = int(os.getenv("ONLY_RECENT_MESSAGES", "0"))
CONVERSATION_MODE = os.getenv("CONVERSATION_MODE", "auto").strip().lower()

POOL_SIZE = int(os.getenv("POOL_SIZE", "50"))

# ── 请求/响应日志 ──
REQUEST_LOG_ENABLED = _env_bool("REQUEST_LOG_ENABLED", False)
REQUEST_LOG_DIR = os.getenv("REQUEST_LOG_DIR", "logs").strip()
REQUEST_LOG_MAX_SIZE = int(
    os.getenv("REQUEST_LOG_MAX_SIZE", str(50 * 1024 * 1024))
)  # 50MB
REQUEST_LOG_BACKUP_COUNT = int(os.getenv("REQUEST_LOG_BACKUP_COUNT", "5"))
REQUEST_LOG_MAX_BODY = int(os.getenv("REQUEST_LOG_MAX_BODY", "0"))  # 0=不截断


# ═══════════════════════════════════════════════════════════════
#  请求/响应日志记录器
# ═══════════════════════════════════════════════════════════════


class RequestResponseLogger:
    """
    将 OpenAI 客户端请求、Dify 转发请求、Dify 响应、OpenAI 返回响应
    记录到独立的日志文件中，格式化 JSON 便于阅读调试。
    """

    def __init__(self) -> None:
        self._logger: Optional[logging.Logger] = None
        if REQUEST_LOG_ENABLED:
            self._setup()

    def _setup(self) -> None:
        log_dir = Path(REQUEST_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger("opendify.traffic")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # 清除已有 handler 防止重复
        self._logger.handlers.clear()

        handler = RotatingFileHandler(
            str(log_dir / "traffic.log"),
            maxBytes=REQUEST_LOG_MAX_SIZE,
            backupCount=REQUEST_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    @property
    def enabled(self) -> bool:
        return self._logger is not None

    def _truncate_body(self, body: Any) -> Any:
        """可选截断超大 body"""
        if REQUEST_LOG_MAX_BODY <= 0:
            return body
        if isinstance(body, str) and len(body) > REQUEST_LOG_MAX_BODY:
            return body[:REQUEST_LOG_MAX_BODY] + f"... [截断，总长 {len(body)}]"
        if isinstance(body, dict):
            text = json.dumps(body, ensure_ascii=False)
            if len(text) > REQUEST_LOG_MAX_BODY:
                return json.loads(text[:REQUEST_LOG_MAX_BODY] + "}")  # best-effort
        return body

    def _format_entry(self, entry: Dict[str, Any]) -> str:
        sep = "=" * 72
        return f"\n{sep}\n{json.dumps(entry, ensure_ascii=False, indent=2)}\n{sep}\n"

    def log_openai_request(
        self, request_id: str, model: str, body: Dict[str, Any]
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "CLIENT → PROXY",
            "stage": "openai_request",
            "request_id": request_id,
            "model": model,
            "stream": body.get("stream", False),
            "message_count": len(body.get("messages", [])),
            "tools_count": len(body.get("tools", [])),
            "tool_choice": body.get("tool_choice"),
            "body": self._truncate_body(body),
        }
        self._logger.debug(self._format_entry(entry))

    def log_dify_request(
        self,
        request_id: str,
        endpoint: str,
        body: Dict[str, Any],
        conversation_id: Optional[str],
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "PROXY → DIFY",
            "stage": "dify_request",
            "request_id": request_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "query_length": len(body.get("query", "")),
            "response_mode": body.get("response_mode"),
            "body": self._truncate_body(body),
        }
        self._logger.debug(self._format_entry(entry))

    def log_dify_response(
        self,
        request_id: str,
        status_code: int,
        body: Any,
        conversation_id: Optional[str] = None,
        is_stream_summary: bool = False,
    ) -> None:
        if not self._logger:
            return
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "DIFY → PROXY",
            "stage": (
                "dify_response_stream_summary" if is_stream_summary else "dify_response"
            ),
            "request_id": request_id,
            "status_code": status_code,
            "conversation_id": conversation_id,
        }
        if isinstance(body, dict):
            entry["answer_length"] = len(body.get("answer", ""))
            entry["body"] = self._truncate_body(body)
        elif isinstance(body, str):
            entry["answer_length"] = len(body)
            entry["body"] = self._truncate_body(body)
        else:
            entry["body"] = str(body)[:2000] if body else None
        self._logger.debug(self._format_entry(entry))

    def log_openai_response(
        self,
        request_id: str,
        body: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> None:
        if not self._logger:
            return
        choices = body.get("choices", [])
        finish_reason = choices[0].get("finish_reason") if choices else None
        has_tool_calls = bool(
            choices
            and isinstance(choices[0], dict)
            and choices[0].get("message", {}).get("tool_calls")
        )
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "PROXY → CLIENT",
            "stage": "openai_response",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "finish_reason": finish_reason,
            "has_tool_calls": has_tool_calls,
            "usage": body.get("usage"),
            "body": self._truncate_body(body),
        }
        self._logger.debug(self._format_entry(entry))

    def log_stream_complete(
        self,
        request_id: str,
        accumulated_text: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        conversation_id: Optional[str],
        finish_reason: str,
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "PROXY → CLIENT (stream summary)",
            "stage": "openai_stream_complete",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "finish_reason": finish_reason,
            "accumulated_text_length": len(accumulated_text),
            "accumulated_text": self._truncate_body(accumulated_text),
            "tool_calls_count": len(tool_calls) if tool_calls else 0,
            "tool_calls": tool_calls,
        }
        self._logger.debug(self._format_entry(entry))

    def log_error(
        self, request_id: str, stage: str, error: str, status_code: Optional[int] = None
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "ERROR",
            "stage": stage,
            "request_id": request_id,
            "status_code": status_code,
            "error": error,
        }
        self._logger.debug(self._format_entry(entry))


traffic_log = RequestResponseLogger()


# ═══════════════════════════════════════════════════════════════
#  HTTP 客户端
# ═══════════════════════════════════════════════════════════════

http_client: httpx.AsyncClient = None  # type: ignore


def _create_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=TIMEOUT, connect=10.0),
        limits=httpx.Limits(
            max_keepalive_connections=POOL_SIZE,
            max_connections=POOL_SIZE,
            keepalive_expiry=30.0,
        ),
        http2=True,
    )


# ═══════════════════════════════════════════════════════════════
#  会话管理器
# ═══════════════════════════════════════════════════════════════


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _make_key(model: str, system_text: str) -> str:
        digest = hashlib.md5((model + system_text[:300]).encode()).hexdigest()[:12]
        return f"{model}::{digest}"

    def get(self, model: str, system_text: str) -> Dict[str, Any]:
        key = self._make_key(model, system_text)
        if key in self._sessions:
            session = self._sessions[key]
            session["msg_count"] += 1
            return session
        token = secrets.token_hex(3)
        session: Dict[str, Any] = {
            "key": key,
            "token": token,
            "conversation_id": None,
            "msg_count": 1,
            "prev_tokens": [],
        }
        self._sessions[key] = session
        return session

    def update_conversation_id(self, session: Dict[str, Any], cid: str) -> None:
        if cid:
            session["conversation_id"] = cid
            self._sessions[session["key"]] = session

    def rotate_token(self, session: Dict[str, Any]) -> str:
        old = session["token"]
        session["prev_tokens"] = (session.get("prev_tokens") or [])[-4:]
        session["prev_tokens"].append(old)
        new_token = secrets.token_hex(3)
        session["token"] = new_token
        return new_token


sessions = SessionStore()


# ═══════════════════════════════════════════════════════════════
#  工具提示词生成
# ═══════════════════════════════════════════════════════════════


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _render_param_tree(
    properties: Dict[str, Any], required: List[str], indent: int = 0
) -> List[str]:
    lines: List[str] = []
    prefix = "  " * indent
    for pname, pinfo in properties.items():
        if not isinstance(pinfo, dict):
            continue
        ptype = pinfo.get("type", "any")
        req_mark = " [必需]" if pname in required else ""
        desc = _truncate(pinfo.get("description", ""), TOOL_DESC_MAX_LENGTH)
        enum_vals = pinfo.get("enum")
        type_str = ptype
        if enum_vals and isinstance(enum_vals, list):
            type_str = "|".join(str(e) for e in enum_vals)
        if ptype == "object" and "properties" in pinfo:
            lines.append(f"{prefix}- {pname} (object){req_mark}: {desc}")
            sub_req = pinfo.get("required", [])
            lines.extend(_render_param_tree(pinfo["properties"], sub_req, indent + 1))
        else:
            lines.append(f"{prefix}- {pname} ({type_str}){req_mark}: {desc}")
    return lines


def build_tool_definitions_text(tools: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for i, tool in enumerate(tools, 1):
        if tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
        name = func.get("name", "unknown")
        desc = _truncate(func.get("description", ""), TOOL_DESC_MAX_LENGTH * 2)
        params = func.get("parameters") or {}
        if SIMPLIFIED_TOOL_DEFS:
            required = params.get("required", [])
            props = params.get("properties", {})
            param_lines = _render_param_tree(props, required)
            block = f"## {i}. {name}\n{desc}"
            if param_lines:
                block += "\n参数:\n" + "\n".join(param_lines)
            sections.append(block)
        else:
            block = f"## {i}. {name}\n{desc}\n参数 JSON Schema:\n"
            block += json.dumps(params, ensure_ascii=False, indent=2)
            sections.append(block)
    return "\n\n".join(sections)


def generate_tool_prompt(
    tools: List[Dict[str, Any]],
    token: str,
    prev_tokens: Optional[List[str]] = None,
) -> str:
    if not tools:
        return ""
    defs_text = build_tool_definitions_text(tools)
    tag_open = f"<tool-calls-{token}>"
    tag_close = f"</tool-calls-{token}>"
    example_calls = [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "example_tool",
                "arguments": {"param1": "value1", "param2": 42},
            },
        }
    ]
    example_json = json.dumps(example_calls, ensure_ascii=False, indent=2)
    history_note = ""
    if prev_tokens:
        expired = ", ".join(prev_tokens[-3:])
        history_note = f"\n已过期的旧令牌（禁止使用）: {expired}"

    prompt = f"""# 可用工具

{defs_text}

# 工具调用格式

需要调用工具时，在回复文字的最后使用 XML 标签包裹 JSON 数组。

当前会话安全令牌: {token}
使用标签: {tag_open} ... {tag_close}{history_note}

重要规则:
1. 令牌 "{token}" 是本次会话专用的，每次会话都不同。你必须使用上面给出的最新令牌，严禁自行创造令牌。
2.使用XML标签时,必须有开始标签{tag_open}和结束标签{tag_close}，不可省略。
3. JSON 必须使用 2 空格缩进，确保括号完整配对。
4. 工具调用 XML 放在回复文字之后。
5. 标签内只能是合法的 JSON 数组，不要用 markdown 代码块包裹。
6. 每个工具调用需要 id（以 "call_" 开头的唯一字符串）、type（"function"）、function（含 name 和 arguments 对象）。

格式示例:

你的分析文字在这里...

{tag_open}
{example_json}
{tag_close}

多个工具调用时放在同一个数组中:

{tag_open}
[
  {{"id": "call_001", "type": "function", "function": {{"name": "tool_a", "arguments": {{"key": "val"}}}}}},
  {{"id": "call_002", "type": "function", "function": {{"name": "tool_b", "arguments": {{"key": "val"}}}}}}
]
{tag_close}"""
    return prompt.strip()


# ═══════════════════════════════════════════════════════════════
#  消息转换: OpenAI → Dify query
# ═══════════════════════════════════════════════════════════════


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content)


def transform_openai_to_dify(
    openai_req: Dict[str, Any],
    session: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = openai_req.get("messages") or []
    if not messages:
        return None

    tools = openai_req.get("tools") or []
    tool_choice = openai_req.get("tool_choice", "auto")
    token = session["token"]
    conversation_id = session.get("conversation_id")
    is_continuation = session["msg_count"] > 1 and conversation_id is not None

    if is_continuation and CONVERSATION_MODE == "auto":
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if (messages[idx] or {}).get("role") == "assistant":
                last_assistant_idx = idx
                break
        system_msgs = [m for m in messages if (m or {}).get("role") == "system"]
        delta_msgs = (
            messages[last_assistant_idx + 1 :] if last_assistant_idx >= 0 else messages
        )
        selected = system_msgs + [
            m for m in delta_msgs if (m or {}).get("role") != "system"
        ]
    elif ONLY_RECENT_MESSAGES > 0:
        system_msgs = [m for m in messages if (m or {}).get("role") == "system"]
        non_system = [m for m in messages if (m or {}).get("role") != "system"]
        selected = system_msgs + non_system[-ONLY_RECENT_MESSAGES:]
    else:
        selected = messages

    query_parts: List[str] = []
    for msg in selected:
        role = msg.get("role", "")
        content = _extract_text(msg.get("content"))
        if role == "system":
            if STRIP_SYSTEM_AFTER_FIRST and is_continuation:
                query_parts.append("[System]: (参照之前的系统指令)")
                continue
            if SYSTEM_PROMPT_MAX_LENGTH > 0:
                content = _truncate(content, SYSTEM_PROMPT_MAX_LENGTH)
            query_parts.append(f"[System]: {content}")
        elif role == "user":
            query_parts.append(f"[User]: {content}")
        elif role == "assistant":
            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                tc_texts = []
                for call in tc:
                    fn = call.get("function") or {}
                    tc_texts.append(
                        f"调用 {fn.get('name', '?')}({json.dumps(fn.get('arguments', {}), ensure_ascii=False)})"
                    )
                query_parts.append(
                    f"[Assistant]: {content or ''}\n工具调用: {'; '.join(tc_texts)}"
                )
            else:
                query_parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            tool_call_id = msg.get("tool_call_id", "")
            id_info = f" (id={tool_call_id})" if tool_call_id else ""
            query_parts.append(f"[Tool Result - {tool_name}{id_info}]: {content}")

    tool_token: Optional[str] = None
    if tools and tool_choice != "none":
        tool_token = token
        tool_prompt = generate_tool_prompt(tools, token, session.get("prev_tokens"))
        query_parts.append(tool_prompt)
        if tool_choice == "required":
            query_parts.append("\n[重要]: 你必须调用至少一个工具。")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname:
                query_parts.append(f"\n[重要]: 请使用 {fname} 工具。")

    user_query = "\n\n".join(query_parts)
    dify_req: Dict[str, Any] = {
        "inputs": {},
        "query": user_query,
        "response_mode": "streaming" if openai_req.get("stream") else "blocking",
        "user": openai_req.get("user", "roo_user"),
    }
    if conversation_id and CONVERSATION_MODE == "auto":
        dify_req["conversation_id"] = conversation_id
    dify_req["_tool_token"] = tool_token
    return dify_req


# ═══════════════════════════════════════════════════════════════
#  工具调用提取
# ═══════════════════════════════════════════════════════════════

_TOOL_TAG_PATTERN = re.compile(
    r"<tool-calls-([0-9a-fA-F]{4,8})>(.*?)</tool-calls-\1>", re.DOTALL
)


def _robut_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json_repair.loads(text)
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")
    return None


def _robust_json_parse(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if ret := _robut_json_loads(text):
        return ret
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    if ret := _robut_json_loads(fixed):
        return ret
    stack: List[str] = []
    for ch in text:
        if ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack and stack[-1] == ch:
                stack.pop()
    if stack:
        patched = text + "".join(reversed(stack))
        if ret := _robut_json_loads(patched):
            return ret
        patched_fixed = re.sub(r",\s*([}\]])", r"\1", patched)
        if ret := _robut_json_loads(patched_fixed):
            return ret
    return None


def _normalize_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    items: List[Any] = []
    if isinstance(raw, dict):
        tc = raw.get("tool_calls")
        items = tc if isinstance(tc, list) else [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    result: List[Dict[str, Any]] = []
    for tc in items:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        args = func.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, indent=2)
        call_id = tc.get("id") or f"call_{secrets.token_hex(8)}"
        result.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return result


def extract_tool_calls(
    text: str, token: Optional[str] = None
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    if not text:
        return text or "", None
    if token:
        tag_open = f"<tool-calls-{token}>"
        tag_close = f"</tool-calls-{token}>"
        open_pos = text.find(tag_open)
        if open_pos < 0:
            return text, None
        clean_text = text[:open_pos].rstrip()
        after_open = text[open_pos + len(tag_open) :]
        close_pos = after_open.find(tag_close)
        json_part = (
            after_open[:close_pos].strip() if close_pos >= 0 else after_open.strip()
        )
        parsed = _robust_json_parse(json_part)
        if parsed is not None:
            calls = _normalize_tool_calls(parsed)
            if calls:
                return clean_text, calls
        return clean_text, None
    match = _TOOL_TAG_PATTERN.search(text)
    if match:
        clean_text = text[: match.start()].rstrip()
        parsed = _robust_json_parse(match.group(2))
        if parsed is not None:
            calls = _normalize_tool_calls(parsed)
            if calls:
                return clean_text, calls
    return text, None


# ═══════════════════════════════════════════════════════════════
#  Dify → OpenAI 响应转换
# ═══════════════════════════════════════════════════════════════


def _fast_id() -> str:
    return secrets.token_hex(16)


def build_openai_response(
    dify_resp: Dict[str, Any],
    model: str,
    tool_token: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_resp.get("answer", "")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    if answer:
        answer, tool_calls = extract_tool_calls(answer, tool_token)
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": answer.strip() if answer and answer.strip() else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage_raw = (
        (dify_resp.get("metadata") or {}).get("usage")
        if isinstance(dify_resp.get("metadata"), dict)
        else {}
    )
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    resp = {
        "id": f"chatcmpl-{_fast_id()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_dify",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        },
    }
    return resp, dify_resp.get("conversation_id")


# ═══════════════════════════════════════════════════════════════
#  Dify SSE 迭代器
# ═══════════════════════════════════════════════════════════════


async def iter_dify_sse(rsp: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    buf = bytearray()
    async for chunk in rsp.aiter_bytes(4096):
        buf.extend(chunk)
        while b"\n" in buf:
            idx = buf.index(b"\n")
            line = bytes(buf[:idx]).strip()
            buf = buf[idx + 1 :]
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data


# ═══════════════════════════════════════════════════════════════
#  流式处理: Dify SSE → OpenAI SSE（含 traffic 日志）
# ═══════════════════════════════════════════════════════════════


async def _stream_and_capture_cid(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
    tool_token: Optional[str],
    include_usage: bool,
    session: Dict[str, Any],
    request_id: str,
) -> AsyncGenerator[str, None]:
    tag_open = f"<tool-calls-{tool_token}>" if tool_token else None
    HOLDBACK = len(tag_open) if tag_open else 0

    accumulated = ""
    sent_up_to = 0
    tool_mode = False
    cid_captured = False
    captured_cid: Optional[str] = None
    final_tool_calls: Optional[List[Dict[str, Any]]] = None
    final_finish_reason = "stop"
    usage_obj: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _chunk(
        delta_content: Optional[str] = None,
        delta_role: Optional[str] = None,
        delta_tool_calls: Optional[List[Dict]] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        c: Dict[str, Any] = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": "fp_dify",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        d = c["choices"][0]["delta"]
        if delta_role:
            d["role"] = delta_role
        if delta_content is not None:
            d["content"] = delta_content
        if delta_tool_calls is not None:
            d["tool_calls"] = delta_tool_calls
        return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

    try:
        yield _chunk(delta_role="assistant")

        async for data in iter_dify_sse(rsp):
            if not cid_captured:
                cid = data.get("conversation_id")
                if isinstance(cid, str) and cid.strip():
                    sessions.update_conversation_id(session, cid)
                    captured_cid = cid
                    cid_captured = True

            event = data.get("event")

            if event in ("message", "agent_message"):
                delta = data.get("answer", "")
                if not delta:
                    continue
                if not isinstance(delta, str):
                    delta = str(delta)
                accumulated += delta

                if tool_mode:
                    continue

                if tag_open and tag_open in accumulated:
                    tool_mode = True
                    tag_pos = accumulated.index(tag_open)
                    unsent = accumulated[sent_up_to:tag_pos].rstrip()
                    if unsent:
                        yield _chunk(delta_content=unsent)
                    sent_up_to = len(accumulated)
                    continue

                safe_end = (
                    (len(accumulated) - HOLDBACK) if HOLDBACK > 0 else len(accumulated)
                )
                if safe_end > sent_up_to:
                    yield _chunk(delta_content=accumulated[sent_up_to:safe_end])
                    sent_up_to = safe_end

            elif event == "message_end":
                meta_usage = (
                    (data.get("metadata") or {}).get("usage")
                    if isinstance(data.get("metadata"), dict)
                    else None
                )
                if isinstance(meta_usage, dict):
                    usage_obj = {
                        "prompt_tokens": meta_usage.get("prompt_tokens", 0),
                        "completion_tokens": meta_usage.get("completion_tokens", 0),
                        "total_tokens": meta_usage.get("total_tokens", 0),
                    }

                if tool_mode:
                    _, tool_calls = extract_tool_calls(accumulated, tool_token)
                    if tool_calls:
                        final_tool_calls = tool_calls
                        final_finish_reason = "tool_calls"
                        tc_list = [
                            {
                                "index": idx,
                                "id": tc["id"],
                                "type": "function",
                                "function": tc["function"],
                            }
                            for idx, tc in enumerate(tool_calls)
                        ]
                        yield _chunk(delta_tool_calls=tc_list)
                        yield _chunk(finish_reason="tool_calls")
                    else:
                        rest = accumulated[sent_up_to:]
                        if rest:
                            yield _chunk(delta_content=rest)
                        yield _chunk(finish_reason="stop")
                else:
                    rest = accumulated[sent_up_to:]
                    if rest:
                        yield _chunk(delta_content=rest)
                    yield _chunk(finish_reason="stop")

                if include_usage:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "system_fingerprint": "fp_dify",
                                "choices": [],
                                "usage": usage_obj,
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                # 记录 Dify 流式响应摘要
                traffic_log.log_dify_response(
                    request_id,
                    200,
                    {"answer": accumulated, "usage": usage_obj},
                    conversation_id=captured_cid,
                    is_stream_summary=True,
                )

                # 记录最终发送给客户端的结果
                clean_text, _ = (
                    extract_tool_calls(accumulated, tool_token)
                    if tool_token
                    else (accumulated, None)
                )
                traffic_log.log_stream_complete(
                    request_id,
                    accumulated_text=clean_text,
                    tool_calls=final_tool_calls,
                    conversation_id=captured_cid,
                    finish_reason=final_finish_reason,
                )

                yield "data: [DONE]\n\n"
                return

        rest = accumulated[sent_up_to:]
        if rest and not tool_mode:
            yield _chunk(delta_content=rest)
        yield _chunk(finish_reason="stop")

        traffic_log.log_stream_complete(
            request_id,
            accumulated_text=accumulated,
            tool_calls=None,
            conversation_id=captured_cid,
            finish_reason="stop",
        )

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"流处理错误: {e}")
        traffic_log.log_error(request_id, "stream_processing", str(e))
        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"


# ═══════════════════════════════════════════════════════════════
#  错误类型
# ═══════════════════════════════════════════════════════════════


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


# ═══════════════════════════════════════════════════════════════
#  认证
# ═══════════════════════════════════════════════════════════════


async def verify_api_key(request: Request) -> str:
    if AUTH_MODE == "disabled":
        return ""
    auth = request.headers.get("Authorization") or ""
    key = auth[7:] if auth.lower().startswith("bearer ") else auth.strip()
    if not key:
        key = (request.headers.get("X-API-Key") or "").strip()
    if not key or key not in VALID_API_KEYS:
        raise APIError(
            401,
            "Invalid API key",
            error_type="invalid_request_error",
            code="invalid_api_key",
        )
    return key


# ═══════════════════════════════════════════════════════════════
#  上游错误处理
# ═══════════════════════════════════════════════════════════════


def _raise_upstream_error(status_code: int, body: bytes, request_id: str = "") -> None:
    try:
        err = json.loads(body or b"{}")
        msg = err.get("message", body.decode("utf-8", errors="ignore"))
        code = err.get("code", "upstream_error")
    except Exception:
        msg = body.decode("utf-8", errors="ignore") if body else "Upstream error"
        code = "upstream_error"

    traffic_log.log_error(
        request_id, "dify_upstream", str(msg), status_code=status_code
    )

    if status_code == 429:
        etype = "rate_limit_error"
    elif status_code >= 500:
        etype = "server_error"
    else:
        etype = "invalid_request_error"
    raise APIError(status_code, str(msg), error_type=etype, code=str(code))


# ═══════════════════════════════════════════════════════════════
#  FastAPI 应用
# ═══════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http_client
    http_client = _create_client()
    log_status = "开启" if REQUEST_LOG_ENABLED else "关闭"
    logger.info(
        "OpenDify Lite 启动 | 模型: %s | 会话模式: %s | 流量日志: %s",
        list(MODEL_KEY_MAP.keys()),
        CONVERSATION_MODE,
        log_status,
    )
    if REQUEST_LOG_ENABLED:
        logger.info("流量日志目录: %s", os.path.abspath(REQUEST_LOG_DIR))
    yield
    await http_client.aclose()


app = FastAPI(
    title="OpenDify Lite",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)


@app.exception_handler(APIError)
async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


# ────────────────────────────────────────
#  POST /v1/chat/completions
# ────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    request_id = f"req_{secrets.token_hex(6)}"

    try:
        body = await request.body()
        openai_req = json.loads(body)
    except (json.JSONDecodeError, Exception):
        raise APIError(400, "Invalid JSON", code="invalid_json")

    model = openai_req.get("model")
    if not isinstance(model, str) or not model.strip():
        raise APIError(400, "Missing 'model'", code="missing_parameter", param="model")

    messages = openai_req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(
            400, "Missing 'messages'", code="missing_parameter", param="messages"
        )

    # 记录 OpenAI 请求
    traffic_log.log_openai_request(request_id, model, openai_req)

    dify_key = MODEL_KEY_MAP.get(model)
    if not dify_key:
        if len(MODEL_KEY_MAP) == 1:
            dify_key = next(iter(MODEL_KEY_MAP.values()))
        else:
            raise APIError(
                404, f"Model '{model}' not found", code="model_not_found", param="model"
            )

    system_text = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_text = _extract_text(m.get("content"))
            break

    session = sessions.get(model, system_text)

    explicit_cid = request.headers.get("X-Dify-Conversation-Id") or openai_req.get(
        "conversation_id"
    )
    if isinstance(explicit_cid, str) and explicit_cid.strip():
        session["conversation_id"] = explicit_cid

    dify_req = transform_openai_to_dify(openai_req, session)
    if not dify_req:
        raise APIError(400, "Failed to transform request", code="transform_error")

    tool_token = dify_req.pop("_tool_token", None)

    # 记录 Dify 请求
    traffic_log.log_dify_request(
        request_id,
        f"{DIFY_API_BASE}/chat-messages",
        dify_req,
        conversation_id=session.get("conversation_id"),
    )

    headers = {
        "Authorization": f"Bearer {dify_key}",
        "Content-Type": "application/json",
    }
    endpoint = f"{DIFY_API_BASE}/chat-messages"
    stream = bool(openai_req.get("stream", False))

    if stream:
        stream_opts = openai_req.get("stream_options") or {}
        include_usage = bool(
            stream_opts.get("include_usage") if isinstance(stream_opts, dict) else False
        )
        message_id = f"chatcmpl-{_fast_id()}"

        cm = http_client.stream(
            "POST",
            endpoint,
            content=json.dumps(dify_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        )
        upstream = await cm.__aenter__()

        if upstream.status_code != 200:
            err_body = await upstream.aread()
            await cm.__aexit__(None, None, None)
            _raise_upstream_error(upstream.status_code, err_body, request_id)

        return StreamingResponse(
            _stream_and_capture_cid(
                upstream,
                model=model,
                message_id=message_id,
                tool_token=tool_token,
                include_usage=include_usage,
                session=session,
                request_id=request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
            background=BackgroundTask(cm.__aexit__, None, None, None),
        )

    # ── 非流式 ──
    resp = await http_client.post(
        endpoint,
        content=json.dumps(dify_req, ensure_ascii=False),
        headers=headers,
        timeout=TIMEOUT,
    )

    if resp.status_code != 200:
        _raise_upstream_error(resp.status_code, resp.content, request_id)

    dify_resp = json.loads(resp.content)

    # 记录 Dify 响应
    traffic_log.log_dify_response(
        request_id,
        resp.status_code,
        dify_resp,
        conversation_id=dify_resp.get("conversation_id"),
    )

    openai_resp, cid = build_openai_response(dify_resp, model, tool_token)

    if cid:
        sessions.update_conversation_id(session, cid)

    # 记录 OpenAI 响应
    traffic_log.log_openai_response(request_id, openai_resp, conversation_id=cid)

    resp_headers: Dict[str, str] = {"Access-Control-Allow-Origin": "*"}
    if cid:
        resp_headers["X-Dify-Conversation-Id"] = cid
    return JSONResponse(content=openai_resp, headers=resp_headers)


# ────────────────────────────────────────
#  GET /v1/models
# ────────────────────────────────────────


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    ts = int(time.time())
    data = [
        {"id": name, "object": "model", "created": ts, "owned_by": "dify"}
        for name in MODEL_KEY_MAP
    ]
    return JSONResponse(
        content={"object": "list", "data": data},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, api_key: str = Depends(verify_api_key)):
    if model_id not in MODEL_KEY_MAP:
        raise APIError(404, f"Model '{model_id}' not found", code="model_not_found")
    ts = int(time.time())
    return JSONResponse(
        content={"id": model_id, "object": "model", "created": ts, "owned_by": "dify"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ═══════════════════════════════════════════════════════════════
#  启动入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        access_log=False,
        server_header=False,
        date_header=False,
        loop="uvloop" if os.name != "nt" else "asyncio",
    )

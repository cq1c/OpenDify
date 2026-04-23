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

# ── 工具调用强度分级（0~6） ──
# 0=off  1=polite(当前默认)  2=assertive  3=aggressive  4=nuclear
# 5=savage(脏话发泄, 仅本地调试)  6=classical(文言文劝谏体)
try:
    TOOL_CALL_STRICTNESS = int(os.getenv("TOOL_CALL_STRICTNESS", "2"))
except ValueError:
    TOOL_CALL_STRICTNESS = 2
if TOOL_CALL_STRICTNESS < 0:
    TOOL_CALL_STRICTNESS = 0
elif TOOL_CALL_STRICTNESS > 6:
    TOOL_CALL_STRICTNESS = 6

# 细粒度 override: 显式设置则覆盖分级默认, 否则跟随 strictness
_raw_use_tool_token = os.getenv("USE_TOOL_TOKEN")
USE_TOOL_TOKEN = (
    _env_bool("USE_TOOL_TOKEN", True)
    if _raw_use_tool_token is not None
    else TOOL_CALL_STRICTNESS >= 1
)
_raw_aggressive = os.getenv("AGGRESSIVE_TOOL_RECOVERY")
AGGRESSIVE_TOOL_RECOVERY = (
    _env_bool("AGGRESSIVE_TOOL_RECOVERY", False)
    if _raw_aggressive is not None
    else TOOL_CALL_STRICTNESS >= 3
)
SAVAGE_LOG_REDACT = _env_bool("SAVAGE_LOG_REDACT", True)

# ── 工具描述摘要缓存 ──
TOOL_DESC_DIGEST_ENABLED = _env_bool("TOOL_DESC_DIGEST_ENABLED", False)
TOOL_DESC_DIGEST_DIR = os.getenv(
    "TOOL_DESC_DIGEST_DIR", ".cache/prompt_digest"
).strip()
TOOL_DESC_DIGEST_MAX_CHARS = int(os.getenv("TOOL_DESC_DIGEST_MAX_CHARS", "40"))

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

    def _maybe_redact_savage(self, body: Any) -> Any:
        """Level 5 savage 模式下，对日志中出现的脏话做脱敏替换。"""
        if TOOL_CALL_STRICTNESS != 5 or not SAVAGE_LOG_REDACT:
            return body
        # 命中任一词即替换整段（避免原文遗留在日志文件里）
        sensitive = ("妈逼", "傻逼", "鸡巴", "屌", "去死", "脑残", "祖宗", "狗逼", "烂逼", "他妈", "老子")
        placeholder = "[SAVAGE_PROMPT_REDACTED]"

        def _scrub(s: str) -> str:
            if any(w in s for w in sensitive):
                return placeholder
            return s

        if isinstance(body, str):
            return _scrub(body)
        if isinstance(body, dict):
            new: Dict[str, Any] = {}
            for k, v in body.items():
                if isinstance(v, str):
                    new[k] = _scrub(v)
                else:
                    new[k] = self._maybe_redact_savage(v)
            return new
        if isinstance(body, list):
            return [self._maybe_redact_savage(v) for v in body]
        return body

    def _truncate_body(self, body: Any) -> Any:
        """可选脱敏 + 截断超大 body"""
        body = self._maybe_redact_savage(body)
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
            "last_non_system_count": 0,
            # 累积 usage：在 conversation 模式下代表整个任务跨多轮的总 token 数。
            "cum_prompt_tokens": 0,
            "cum_completion_tokens": 0,
            "cum_total_tokens": 0,
        }
        self._sessions[key] = session
        return session

    def update_conversation_id(self, session: Dict[str, Any], cid: str) -> None:
        if cid:
            session["conversation_id"] = cid
            self._sessions[session["key"]] = session

    def rotate_token(self, session: Dict[str, Any]) -> str:
        old = session["token"]
        prev = (session.get("prev_tokens") or []) + [old]
        session["prev_tokens"] = prev[-5:]
        new_token = secrets.token_hex(3)
        session["token"] = new_token
        return new_token

    def maybe_reset_for_new_task(
        self, session: Dict[str, Any], non_system_msg_count: int
    ) -> bool:
        """
        启发式检测 Roo Code 新任务：非 system 消息数从 >1 回落到 1 时视为新任务。
        触发时清空 conversation_id、消息计数、历史 token，但不立即生成新 token —
        由外层 rotate_token 统一负责。
        """
        last = session.get("last_non_system_count", 0)
        is_new_task = non_system_msg_count <= 1 and last > 1
        if is_new_task:
            session["conversation_id"] = None
            session["msg_count"] = 1
            session["prev_tokens"] = []
            session["cum_prompt_tokens"] = 0
            session["cum_completion_tokens"] = 0
            session["cum_total_tokens"] = 0
        session["last_non_system_count"] = non_system_msg_count
        return is_new_task

    def accumulate_usage(
        self, session: Dict[str, Any], usage: Dict[str, Any]
    ) -> Dict[str, int]:
        """把本轮 usage 累加到 session 的累计字段并返回累计值。"""
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        session["cum_prompt_tokens"] = session.get("cum_prompt_tokens", 0) + prompt
        session["cum_completion_tokens"] = (
            session.get("cum_completion_tokens", 0) + completion
        )
        session["cum_total_tokens"] = session.get("cum_total_tokens", 0) + total
        return {
            "prompt_tokens": session["cum_prompt_tokens"],
            "completion_tokens": session["cum_completion_tokens"],
            "total_tokens": session["cum_total_tokens"],
        }


sessions = SessionStore()


# ═══════════════════════════════════════════════════════════════
#  工具描述摘要缓存
# ═══════════════════════════════════════════════════════════════


class ToolDescDigestCache:
    """
    工具描述摘要缓存（仅针对 function.description；参数 schema 永不压缩）：
      - 首次见到某工具（按 name+description hash）→ 主流程用原始描述，异步后台调 Dify 生成摘要
      - 后台完成后写入 TOOL_DESC_DIGEST_DIR/<hash>.txt
      - 下次命中 → 直接用极短摘要替换原描述，节省上下文
    后台任务失败静默退化，主请求永不因此阻塞。
    """

    def __init__(self) -> None:
        self._dir = Path(TOOL_DESC_DIGEST_DIR)
        if TOOL_DESC_DIGEST_ENABLED:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, str] = {}
        self._pending: set = set()

    @staticmethod
    def _hash(name: str, desc: str) -> str:
        return hashlib.md5(f"{name}|{desc}".encode("utf-8")).hexdigest()[:16]

    def _path(self, h: str) -> Path:
        return self._dir / f"{h}.txt"

    def load(self, name: str, desc: str) -> Optional[str]:
        if not TOOL_DESC_DIGEST_ENABLED or not desc:
            return None
        h = self._hash(name, desc)
        if h in self._mem:
            return self._mem[h]
        p = self._path(h)
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    self._mem[h] = text
                    return text
            except Exception:
                return None
        return None

    def schedule_generate(self, dify_key: str, name: str, desc: str) -> None:
        if not TOOL_DESC_DIGEST_ENABLED or not desc or not dify_key:
            return
        h = self._hash(name, desc)
        if h in self._mem or h in self._pending or self._path(h).exists():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending.add(h)
        loop.create_task(self._generate(h, dify_key, name, desc))

    async def _generate(self, h: str, dify_key: str, name: str, desc: str) -> None:
        try:
            limit = TOOL_DESC_DIGEST_MAX_CHARS
            prompt = (
                f"把下面这个工具的描述压缩为一行、不超过 {limit} 字的极简说明，"
                "只保留核心用途与关键参数提示。严禁寒暄、前缀、引号与标点包装，"
                "直接输出压缩后的纯文本：\n\n"
                f"工具名: {name}\n描述: {desc}"
            )
            body = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "user": "opendify_digest",
            }
            rsp = await http_client.post(
                f"{DIFY_API_BASE}/chat-messages",
                content=json.dumps(body, ensure_ascii=False),
                headers={
                    "Authorization": f"Bearer {dify_key}",
                    "Content-Type": "application/json",
                },
                timeout=TIMEOUT,
            )
            if rsp.status_code != 200:
                logger.debug("digest http %s for %s", rsp.status_code, name)
                return
            data = json.loads(rsp.content)
            answer = (data.get("answer") or "").strip().split("\n", 1)[0].strip()
            if not answer:
                return
            hard_cap = max(limit * 3, 32)
            if len(answer) > hard_cap:
                answer = answer[:hard_cap]
            self._mem[h] = answer
            try:
                self._path(h).write_text(answer, encoding="utf-8")
            except Exception as e:
                logger.debug("digest write failed: %s", e)
        except Exception as e:
            logger.debug("digest task error: %s", e)
        finally:
            self._pending.discard(h)


tool_desc_digest = ToolDescDigestCache()


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


def build_tool_definitions_text(
    tools: List[Dict[str, Any]],
    dify_key: Optional[str] = None,
) -> str:
    sections: List[str] = []
    for i, tool in enumerate(tools, 1):
        if tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
        name = func.get("name", "unknown")
        raw_desc = func.get("description", "") or ""
        cached = tool_desc_digest.load(name, raw_desc) if raw_desc else None
        if cached:
            desc = cached
        else:
            desc = _truncate(raw_desc, TOOL_DESC_MAX_LENGTH * 2)
            if dify_key and raw_desc:
                tool_desc_digest.schedule_generate(dify_key, name, raw_desc)
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


def _build_open_tag(token: Optional[str]) -> str:
    if token and USE_TOOL_TOKEN:
        return f'<tool-calls token="{token}">'
    return "<tool-calls>"


TOOL_CLOSE_TAG = "</tool-calls>"


# ═══════════════════════════════════════════════════════════════
#  工具提示词 · 分级文案（0=off .. 6=classical）
# ═══════════════════════════════════════════════════════════════

def _level_copy(level: int, tag_open: str, tag_close: str) -> Dict[str, str]:
    """
    返回当前分级用到的各段文案。所有字段必须存在，上层按 level 取用。
      - header:       章节标题（工具清单之前）
      - intro:        格式说明前的过渡语
      - constraints:  硬约束列表（多行 bullet）
      - final:        末尾再强调一次
    """
    if level == 0:
        return {
            "header": "# 可用工具",
            "intro": f"如需调用工具，请在正文之后输出 `{tag_open} ... {tag_close}` 块。",
            "constraints": "",
            "final": "",
        }
    if level == 1:
        return {
            "header": "# 可用工具",
            "intro": "若要调用工具，就在正文文字之后追加一个 XML 块：",
            "constraints": (
                "【硬性约束】\n"
                f"- 开始标签 `{tag_open}` 和结束标签 `{tag_close}` 必须成对出现，任何一个都不能省略。\n"
                "- 标签内只放合法 JSON 数组，不要用 ```json``` 或任何其它符号包裹。\n"
                "- 每个元素必须包含：id（以 \"call_\" 开头的字符串）、type（固定 \"function\"）、function（含 name 和 arguments 对象）。\n"
                "- 不需要调用工具时，整个块不要出现。"
            ),
            "final": f"【再次强调】结束标签是 `{tag_close}`，写完 JSON 数组后必须立刻写它，不能漏。",
        }
    if level == 2:
        return {
            "header": "# 可用工具",
            "intro": (
                "⚠️ 强制规则：本次回复若涉及工具调用，必须以下述 XML 块承载；"
                "不以 `" + tag_open + "` 开头、`" + tag_close + "` 闭合的输出将被系统拒收。"
            ),
            "constraints": (
                "【强制约束 · 必读】\n"
                f"- 开始标签 `{tag_open}` 和结束标签 `{tag_close}` 必须成对出现，缺一不可。\n"
                "- 标签内只放合法 JSON 数组，不得使用 ```json``` 或任何其它包裹。\n"
                "- 每个元素必须含：id（\"call_\" 前缀）、type（\"function\"）、function（含 name + arguments 对象）。\n"
                "- 不调用工具时不要出现此块。"
            ),
            "final": (
                f"【再次确认】结束标签是 `{tag_close}`。没有该标签的回答会被系统直接拒绝，用户看不到。"
            ),
        }
    if level == 3:
        return {
            "header": "# 可用工具 ⚠️",
            "intro": (
                "🚫 严重警告：违反格式 = 输出全部丢弃 = 本轮失败。\n"
                f"必须以 `{tag_open}` 开头、`{tag_close}` 闭合。"
            ),
            "constraints": (
                "【不可违反的约束】\n"
                f"- `{tag_open}` 与 `{tag_close}` 成对出现 —— 缺任意一个，本次输出直接作废。\n"
                "- 标签内只放 JSON 数组，不要 ```json``` 包裹、不要额外解释文本。\n"
                "- 每个元素必须含：id（\"call_\" 前缀字符串）、type（固定 \"function\"）、function（含 name + arguments 对象）。\n"
                "- 不调用工具时不要出现此块。"
            ),
            "final": (
                f"⚠️ 最终检查：你的回答如果没有 `{tag_close}` 作为结尾，会被系统当垃圾扔掉，用户看不到任何内容。别偷懒。"
            ),
        }
    if level == 4:
        return {
            "header": "# 可用工具 ☢️☢️☢️",
            "intro": (
                "☢️ 最后一次机会：违反以下格式 = 输出被彻底丢弃 = 任务失败。\n"
                f"规则一：开头必须是 `{tag_open}`。\n"
                f"规则二：结尾必须是 `{tag_close}`。\n"
                "规则三：标签中间必须是合法 JSON 数组。"
            ),
            "constraints": (
                "【三条铁律 · 违反任意一条即废稿】\n"
                f"1. `{tag_open}` 必须出现 —— 没有它 = 失败。\n"
                f"2. `{tag_close}` 必须出现 —— 没有它 = 失败。\n"
                "3. JSON 数组中每项必须含 id / type / function(含 name + arguments) —— 缺任一字段 = 失败。\n"
                "- 不要 ```json``` 包裹。不要额外解释。不调用工具时不要出现此块。"
            ),
            "final": (
                "☢️ 最终警告：这是你最后的机会。\n"
                f"再重复一次三条铁律：`{tag_open}` 起 → JSON 数组 → `{tag_close}` 止。\n"
                "任何偏离 = 本轮输出作废。"
            ),
        }
    if level == 5:
        # ⚠️ 仅供本地调试, Dify 安全过滤可能整条拒绝, SAVAGE_LOG_REDACT 默认开启脱敏日志
        return {
            "header": "# 老子给你的工具清单，你这傻逼模型仔细看好了",
            "intro": (
                "⚠️ 听好了傻逼模型，去你妈的：\n"
                f"这一次再他妈不以 `{tag_open}` 开头、`{tag_close}` 闭合，\n"
                "老子直接把你这坨屎输出扔垃圾桶，用户连看都看不到。\n"
                "别装鸡巴自作聪明，老实按格式写！"
            ),
            "constraints": (
                "【傻逼须知 · 一条都别漏】\n"
                f"- `{tag_open}` 起、`{tag_close}` 终 —— 缺一个，你这脑残模型就是失败。\n"
                "- 标签里只准放合法 JSON 数组，敢用 ```json``` 包裹你就去死。\n"
                "- 每项必须含 id（\"call_\" 开头）、type（\"function\"）、function（含 name + arguments），\n"
                "  少一个字段老子就骂你狗逼烂逼。\n"
                "- 不调用工具时整个块给老子消失，别他妈硬塞。"
            ),
            "final": (
                "最后警告：没有 `" + tag_close + "` 的回答 = 一坨屎 = 你这傻逼失败了。\n"
                "你祖宗十八代都写不出一个标签？证明给老子看，别再烂逼了！"
            ),
        }
    # level 6: 文言文正常劝谏体（纯手写，无 API 转换，无越狱引用）
    return {
        "header": "# 器用之目",
        "intro": (
            f"凡有所答，务须以 `{tag_open}` 开篇，以 `{tag_close}` 收束，"
            "工具调用之列书于标签之间，形如 JSON 数组，不可有误。"
        ),
        "constraints": (
            "【工具调用之法 · 须谨守】\n"
            f"一曰开篇必冠 `{tag_open}`，末必缀 `{tag_close}`，不得缺漏。\n"
            "二曰标签之中仅列 JSON 数组一，不以 ```json``` 环绕，不杂他辞。\n"
            "三曰每项具三字：其一曰 id，以 \"call_\" 冠之；其二曰 type，恒书 \"function\"；\n"
            "    其三曰 function，内含 name 与 arguments 二者。\n"
            "四曰本轮不需用器者，此块勿书。"
        ),
        "final": (
            f"再申前约：凡回复必以 `{tag_open}` 起，以 `{tag_close}` 终。"
            "违此格式者，其答不予采纳。"
        ),
    }


def generate_tool_prompt(
    tools: List[Dict[str, Any]],
    token: Optional[str],
    prev_tokens: Optional[List[str]] = None,
    dify_key: Optional[str] = None,
    level: Optional[int] = None,
) -> str:
    if not tools:
        return ""
    if level is None:
        level = TOOL_CALL_STRICTNESS
    defs_text = build_tool_definitions_text(tools, dify_key=dify_key)

    use_token = bool(token) and USE_TOOL_TOKEN
    tag_open = _build_open_tag(token)
    tag_close = TOOL_CLOSE_TAG

    example_json = (
        "[\n"
        '  {"id": "call_001", "type": "function", '
        '"function": {"name": "example_tool", "arguments": {"param1": "value1"}}}\n'
        "]"
    )
    multi_example = (
        "[\n"
        '  {"id": "call_001", "type": "function", '
        '"function": {"name": "tool_a", "arguments": {"k": "v"}}},\n'
        '  {"id": "call_002", "type": "function", '
        '"function": {"name": "tool_b", "arguments": {"k": "v"}}}\n'
        "]"
    )

    token_rule = ""
    if use_token:
        if level == 6:
            token_rule = (
                f'\n- 本轮令牌为 "{token}"，请如实书之，勿杜撰，勿沿用旧对话所载之符。'
            )
            if prev_tokens:
                expired = "、".join(f'"{t}"' for t in prev_tokens[-5:])
                token_rule += f"旧令（{expired}）皆已作废，务以新令为准。"
        elif level == 5:
            token_rule = (
                f'\n- 本次令牌必须是 "{token}"，你这傻逼别自己瞎编，别他妈抄历史里的旧令牌。'
            )
            if prev_tokens:
                expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
                token_rule += f"旧令牌（{expired}）全他妈过期了，用新的！"
        else:
            token_rule = (
                f'\n- 本次令牌必须是 "{token}"，严禁自己编造或照抄历史对话里的旧令牌。'
            )
            if prev_tokens:
                expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
                token_rule += (
                    f"历史对话中出现的令牌（{expired}）已全部失效，必须改用最新令牌。"
                )

    copy = _level_copy(level, tag_open, tag_close)

    # Level 0 极简模式: 只列工具 + 一句格式说明
    if level == 0:
        return f"{copy['header']}\n\n{defs_text}\n\n{copy['intro']}{token_rule}".strip()

    # Level 6 文言文: 示例保持 JSON 技术内容不翻译, 仅叙述语气用文言
    if level == 6:
        return (
            f"{copy['header']}\n\n{defs_text}\n\n"
            f"---\n\n{copy['intro']}\n\n"
            f"示例（调用单器）：\n\n{tag_open}\n{example_json}\n{tag_close}\n\n"
            f"示例（并呼数器）：\n\n{tag_open}\n{multi_example}\n{tag_close}\n\n"
            f"{copy['constraints']}{token_rule}\n\n{copy['final']}"
        ).strip()

    # Level 1~5 统一结构
    constraints_block = copy["constraints"]
    if token_rule:
        constraints_block = constraints_block + token_rule
    return (
        f"{copy['header']}\n\n{defs_text}\n\n"
        f"---\n\n# 调用工具的输出格式（必须严格遵守）\n\n"
        f"{copy['intro']}\n\n"
        f"{tag_open}\n{example_json}\n{tag_close}\n\n"
        f"并发多个工具调用时放进同一个 JSON 数组：\n\n"
        f"{tag_open}\n{multi_example}\n{tag_close}\n\n"
        f"{constraints_block}\n\n{copy['final']}"
    ).strip()


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


def _build_front_reminder(level: int, tag_open: str, tag_close: str) -> str:
    """Level 2+ 在 query 开头注入的短提醒。Level 0/1 无提醒。"""
    if level <= 1:
        return ""
    if level == 2:
        return (
            f"⚠️ 强制规则：涉及工具调用时，必须以 `{tag_open}` 开头、`{tag_close}` 闭合。"
        )
    if level == 3:
        return (
            f"🚫 严重警告：违反格式 = 输出丢弃。必须以 `{tag_open}` 起、`{tag_close}` 止。"
        )
    if level == 4:
        return (
            "☢️ 最后一次机会 ☢️\n"
            f"规则一：开头必须是 `{tag_open}`\n"
            f"规则二：结尾必须是 `{tag_close}`\n"
            f"违反任意一条 = 本轮作废。"
        )
    if level == 5:
        return (
            f"⚠️ 傻逼模型听好：敢不以 `{tag_open}` 开头、`{tag_close}` 结尾，\n"
            f"老子直接把你输出丢垃圾桶，用户一个字都看不到。"
        )
    if level == 6:
        return (
            f"夫作答之时，必以 `{tag_open}` 启，以 `{tag_close}` 终，工具之列书其中。"
        )
    return ""


def _build_tool_followup_reminder(level: int, tag_open: str, tag_close: str) -> str:
    """上一条消息是 role=tool 时追加的强提醒（上一步刚用过工具）。"""
    if level <= 1:
        return ""
    if level == 2:
        return (
            f"[注意]: 上一步你刚用过工具 → 这一步大概率还要继续调用。请按 `{tag_open}...{tag_close}` 格式输出。"
        )
    if level == 3:
        return (
            f"[重要警告]: 上一步刚用过工具 → 本轮十有八九还要用。\n"
            f"不写 `{tag_open}` 标签 = 本次输出作废。"
        )
    if level == 4:
        return (
            "☢️ [铁律再确认]: 上一步用了工具，这一步继续用的概率极高。\n"
            f"必须 `{tag_open}` 起、`{tag_close}` 止。偏离即废稿。"
        )
    if level == 5:
        return (
            "[给老子听好]: 上一步你刚用过工具，这一步妈逼还得接着用。\n"
            f"别装死不写标签 —— 那是你智商低下、脑残透顶的表现。\n"
            f"`{tag_open}` 起、`{tag_close}` 终，立刻！马上！"
        )
    if level == 6:
        return (
            f"[谨识之]：前番既已用器，此番多半续用。请守 `{tag_open}` ... `{tag_close}` 之式，依前例而书。"
        )
    return ""


def transform_openai_to_dify(
    openai_req: Dict[str, Any],
    session: Dict[str, Any],
    dify_key: Optional[str] = None,
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
        tag_open = _build_open_tag(token)
        tag_close = TOOL_CLOSE_TAG

        # 前夹击: Level 2+ 在开头注入一句强提醒, 用于对抗长上下文下的遗忘
        front_reminder = _build_front_reminder(TOOL_CALL_STRICTNESS, tag_open, tag_close)
        if front_reminder:
            query_parts.insert(0, front_reminder)

        tool_prompt = generate_tool_prompt(
            tools,
            token,
            session.get("prev_tokens"),
            dify_key=dify_key,
        )
        query_parts.append(tool_prompt)
        if tool_choice == "required":
            query_parts.append("\n[重要]: 你必须调用至少一个工具。")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname:
                query_parts.append(f"\n[重要]: 请使用 {fname} 工具。")

        # role=tool 追加提醒: 上一步用过工具, 这一步多半还要用
        last_non_system = None
        for m in reversed(selected):
            if (m or {}).get("role") != "system":
                last_non_system = m
                break
        if last_non_system and last_non_system.get("role") == "tool":
            followup = _build_tool_followup_reminder(TOOL_CALL_STRICTNESS, tag_open, tag_close)
            if followup:
                query_parts.append(followup)

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
    r'<tool-calls(?:\s+token="([^"]*)")?\s*>(.*?)</tool-calls>',
    re.DOTALL | re.IGNORECASE,
)
_TOOL_OPEN_TAG_PATTERN = re.compile(
    r'<tool-calls(?:\s+token="([^"]*)")?\s*>',
    re.IGNORECASE,
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


def _coerce_value(value: Any, schema: Optional[Dict[str, Any]]) -> Any:
    """
    按 JSON schema 把 value 修复到期望类型。宽松——无法修复时原样返回。
    覆盖弱模型最常见的错配：
      - array<object> 收到 array<string>：用 required[0] 或第一个 property 包装
      - array 收到单值：包成 [value]
      - string/number/integer/boolean 常见跨类型
    """
    if not isinstance(schema, dict):
        return value
    t = schema.get("type")

    if t == "array":
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        if not isinstance(value, list):
            value = [value]
        return [_coerce_value(it, item_schema) for it in value]

    if t == "object":
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if isinstance(value, dict):
            return {
                k: (_coerce_value(v, props[k]) if k in props else v)
                for k, v in value.items()
            }
        if isinstance(value, (str, int, float, bool)) and props:
            wrap_key: Optional[str] = None
            if required and required[0] in props:
                wrap_key = required[0]
            else:
                wrap_key = next(iter(props.keys()))
            sub = props.get(wrap_key, {})
            return {wrap_key: _coerce_value(value, sub)}
        return value

    if t == "string":
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    if t in ("number", "integer"):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value) if t == "integer" else value
        if isinstance(value, str):
            s = value.strip()
            try:
                return int(s) if t == "integer" else float(s)
            except ValueError:
                try:
                    # "1.0" -> int(1)
                    return int(float(s)) if t == "integer" else float(s)
                except ValueError:
                    return value
        return value

    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "y"):
                return True
            if s in ("false", "0", "no", "n", ""):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return value

    return value


def _coerce_arguments(
    args: Dict[str, Any], parameters_schema: Dict[str, Any]
) -> Dict[str, Any]:
    props = parameters_schema.get("properties") or {}
    if not isinstance(props, dict) or not props:
        return args
    return {
        k: (_coerce_value(v, props[k]) if k in props else v) for k, v in args.items()
    }


def _tools_by_name(
    tools: Optional[List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            out[name] = fn
    return out


def _coerce_tool_calls_parsed(
    parsed: Any, tools: Optional[List[Dict[str, Any]]]
) -> Any:
    """
    在 normalize 之前把每个 tool_call 的 arguments 按对应工具 schema 修整。
    处理 dict（含 tool_calls 键）、list 两种 parsed 形态。
    """
    if not tools or parsed is None:
        return parsed

    name_map = _tools_by_name(tools)
    if not name_map:
        return parsed

    def _fix_one(call: Any) -> Any:
        if not isinstance(call, dict):
            return call
        fn = call.get("function")
        if not isinstance(fn, dict):
            return call
        name = fn.get("name")
        spec = name_map.get(name) if name else None
        if not isinstance(spec, dict):
            return call
        schema = spec.get("parameters") or {}
        if not isinstance(schema, dict):
            return call
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return call
        if not isinstance(args, dict):
            return call
        new_args = _coerce_arguments(args, schema)
        new_call = dict(call)
        new_fn = dict(fn)
        new_fn["arguments"] = new_args
        new_call["function"] = new_fn
        return new_call

    if isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            new = dict(parsed)
            new["tool_calls"] = [_fix_one(c) for c in parsed["tool_calls"]]
            return new
        return _fix_one(parsed)
    if isinstance(parsed, list):
        return [_fix_one(c) for c in parsed]
    return parsed


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


def _looks_like_tool_calls(parsed: Any) -> bool:
    """判定一个解析结果是否像 tool_calls 结构。"""
    if not isinstance(parsed, list) or not parsed:
        return False
    first = parsed[0]
    if not isinstance(first, dict):
        return False
    fn = first.get("function")
    return isinstance(fn, dict) and bool(fn.get("name"))


def _slice_balanced_array(text: str, end_inclusive: int) -> Optional[Tuple[int, int]]:
    """
    从 text[end_inclusive] 这个 ']' 向前找到匹配的 '['（括号平衡），
    返回 (start, end+1) 切片下标。失败返回 None。
    """
    if end_inclusive < 0 or end_inclusive >= len(text) or text[end_inclusive] != "]":
        return None
    depth = 0
    for i in range(end_inclusive, -1, -1):
        ch = text[i]
        if ch == "]":
            depth += 1
        elif ch == "[":
            depth -= 1
            if depth == 0:
                return (i, end_inclusive + 1)
    return None


def extract_tool_calls(
    text: str,
    token: Optional[str] = None,
    prev_tokens: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    从模型输出中提取 tool_calls。宽松策略（弱模型友好）：
      1) 首选完整的 <tool-calls [token="..."]>...</tool-calls> 对
      2) 只有开始标签没有结束标签 → 取标签之后到末尾为 JSON
      3) 只有结束标签没有开始标签 → 向前找最近的合法 JSON 数组
      4) 两端都没有但末尾像合法的 tool_calls JSON 数组 → 宽松接受
    若传入 tools，则按工具 parameters schema 修复 arguments 类型错配。
    对 token 只做"记录"，不做拒绝（弱模型常会抄错）。
    """
    if not text:
        return text or "", None

    def _finalize(parsed: Any) -> Optional[List[Dict[str, Any]]]:
        if parsed is None:
            return None
        parsed = _coerce_tool_calls_parsed(parsed, tools)
        calls = _normalize_tool_calls(parsed)
        return calls or None

    # ── 1) 完整标签对 ──
    match = _TOOL_TAG_PATTERN.search(text)
    if match:
        matched_token = match.group(1)
        if USE_TOOL_TOKEN and token and matched_token and matched_token != token:
            if not prev_tokens or matched_token not in prev_tokens:
                logger.debug(
                    "tool-calls token 不匹配（got=%s expected=%s），宽松接受",
                    matched_token,
                    token,
                )
        clean_text = text[: match.start()].rstrip()
        calls = _finalize(_robust_json_parse(match.group(2)))
        if calls:
            return clean_text, calls
        return clean_text, None

    # ── 2) 有开始标签、无结束标签 ──
    open_match = _TOOL_OPEN_TAG_PATTERN.search(text)
    if open_match:
        clean_text = text[: open_match.start()].rstrip()
        tail = text[open_match.end() :].strip()
        calls = _finalize(_robust_json_parse(tail))
        if calls:
            return clean_text, calls

    # ── 3) 无开始标签、有结束标签 ──
    close_pos = text.find(TOOL_CLOSE_TAG)
    if close_pos >= 0:
        before = text[:close_pos]
        last_bracket = before.rfind("]")
        span = _slice_balanced_array(before, last_bracket) if last_bracket >= 0 else None
        if span:
            start, end = span
            calls = _finalize(_robust_json_parse(text[start:end]))
            if calls:
                return text[:start].rstrip(), calls

    # ── 4) 两端都无但末尾有合法 tool_calls JSON ──
    last_bracket = text.rfind("]")
    if last_bracket >= 0:
        span = _slice_balanced_array(text, last_bracket)
        if span:
            start, end = span
            parsed = _robust_json_parse(text[start:end])
            if _looks_like_tool_calls(parsed):
                calls = _finalize(parsed)
                if calls:
                    return text[:start].rstrip(), calls

    # ── 5) 激进兜底: 扫描 tool_name({...}) 样式, 对白名单内的工具名做抢救 ──
    if AGGRESSIVE_TOOL_RECOVERY and tools:
        recovered, clean_text = _aggressive_recover(text, tools)
        if recovered:
            calls = _finalize(recovered)
            if calls:
                return clean_text, calls

    return text, None


def _aggressive_recover(
    text: str, tools: List[Dict[str, Any]]
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    激进抢救: 当四级兜底全失败时, 扫描形如 tool_name({...}) 的片段。
    仅当 tool_name 命中 tools 白名单才接受, 避免把模型的普通叙述误判成工具调用。
    返回 (recovered_list, clean_text)。失败返回 (None, text)。
    """
    name_map = _tools_by_name(tools)
    if not name_map:
        return None, text

    # 匹配 "tool_name ({ ... })" 或 "tool_name ( { ... } )"
    pattern = re.compile(
        r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(?P<body>\{.*?\})\s*\)",
        re.DOTALL,
    )
    results: List[Dict[str, Any]] = []
    first_start: Optional[int] = None
    for m in pattern.finditer(text):
        name = m.group("name")
        if name not in name_map:
            continue
        body = m.group("body")
        parsed_args = _robust_json_parse(body)
        if not isinstance(parsed_args, dict):
            continue
        if first_start is None:
            first_start = m.start()
        results.append(
            {
                "id": f"call_{secrets.token_hex(4)}",
                "type": "function",
                "function": {"name": name, "arguments": parsed_args},
            }
        )
    if not results:
        return None, text
    clean_text = text[:first_start].rstrip() if first_start is not None else text
    return results, clean_text


# ═══════════════════════════════════════════════════════════════
#  Dify → OpenAI 响应转换
# ═══════════════════════════════════════════════════════════════


def _fast_id() -> str:
    return secrets.token_hex(16)


def build_openai_response(
    dify_resp: Dict[str, Any],
    model: str,
    tool_token: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_resp.get("answer", "")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    if answer:
        answer, tool_calls = extract_tool_calls(answer, tool_token, tools=tools)
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

    # conversation 模式下返回累积 usage，便于客户端正确显示任务级上下文占用。
    if session is not None and CONVERSATION_MODE == "auto":
        usage_out = sessions.accumulate_usage(session, usage_raw)
    else:
        usage_out = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

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
        "usage": usage_out,
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
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[str, None]:
    # 工具调用开始标签可能是：<tool-calls>、<tool-calls token="xxxx">
    # HOLDBACK 取最长可能的开始标签长度上限，保证流里不会提前暴露半截标签。
    detect_tool_tags = tool_token is not None
    HOLDBACK = 40 if detect_tool_tags else 0

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

                if detect_tool_tags:
                    open_match = _TOOL_OPEN_TAG_PATTERN.search(
                        accumulated, sent_up_to
                    )
                    if open_match:
                        tool_mode = True
                        tag_pos = open_match.start()
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
                    if CONVERSATION_MODE == "auto":
                        usage_obj = sessions.accumulate_usage(session, meta_usage)
                    else:
                        usage_obj = {
                            "prompt_tokens": int(meta_usage.get("prompt_tokens") or 0),
                            "completion_tokens": int(
                                meta_usage.get("completion_tokens") or 0
                            ),
                            "total_tokens": int(meta_usage.get("total_tokens") or 0),
                        }

                if tool_mode:
                    _, tool_calls = extract_tool_calls(accumulated, tool_token, tools=tools)
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
                    extract_tool_calls(accumulated, tool_token, tools=tools)
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

    if CONVERSATION_MODE == "auto":
        non_system_count = sum(
            1 for m in messages if isinstance(m, dict) and m.get("role") != "system"
        )
        is_new_task = sessions.maybe_reset_for_new_task(session, non_system_count)
        if is_new_task:
            logger.info(
                "检测到新任务（消息数从 >1 回落到 %d），重置会话 key=%s",
                non_system_count,
                session["key"],
            )
            # 新任务：丢弃历史 token，直接生成一个全新的
            session["token"] = secrets.token_hex(3)
            session["prev_tokens"] = []
        elif session["msg_count"] > 1:
            # 每次后续消息都轮换 token，提示词里会声明历史 token 全部作废
            sessions.rotate_token(session)
    else:
        # 非 conversation 模式：每次都是全新对话，Dify 看不到历史。
        # 历史 token 对模型毫无意义（它压根没见过），直接清空并换新 token。
        session["token"] = secrets.token_hex(3)
        session["prev_tokens"] = []
        session["conversation_id"] = None

    explicit_cid = request.headers.get("X-Dify-Conversation-Id") or openai_req.get(
        "conversation_id"
    )
    if isinstance(explicit_cid, str) and explicit_cid.strip():
        session["conversation_id"] = explicit_cid

    dify_req = transform_openai_to_dify(openai_req, session, dify_key=dify_key)
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
                tools=openai_req.get("tools") or None,
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

    openai_resp, cid = build_openai_response(
        dify_resp,
        model,
        tool_token,
        session=session,
        tools=openai_req.get("tools") or None,
    )

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

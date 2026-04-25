"""环境变量加载、日志配置以及全局常量。"""

import logging
import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return v.strip().lower() in ("1", "true", "yes") if v else default


# ── 日志 ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").strip().upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("opendify")
for _lib in ("httpx", "httpcore", "uvicorn.access"):
    logging.getLogger(_lib).setLevel(logging.ERROR)


# ── 服务器 ──
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1").strip()
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))


# ── 鉴权 ──
VALID_API_KEYS = frozenset(
    k.strip() for k in os.getenv("VALID_API_KEYS", "").split(",") if k.strip()
)
AUTH_MODE = os.getenv("AUTH_MODE", "required").strip().lower()


# ── Dify 上游 ──
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


# ── 消息/提示词裁剪 ──
STRIP_SYSTEM_AFTER_FIRST = _env_bool("STRIP_SYSTEM_AFTER_FIRST", False)
SYSTEM_PROMPT_MAX_LENGTH = int(os.getenv("SYSTEM_PROMPT_MAX_LENGTH", "0"))
SIMPLIFIED_TOOL_DEFS = _env_bool("SIMPLIFIED_TOOL_DEFS", True)
TOOL_DESC_MAX_LENGTH = int(os.getenv("TOOL_DESC_MAX_LENGTH", "120"))
ONLY_RECENT_MESSAGES = int(os.getenv("ONLY_RECENT_MESSAGES", "0"))
CONVERSATION_MODE = os.getenv("CONVERSATION_MODE", "auto").strip().lower()

# CONVERSATION_MODE=auto 下 prompt_tokens 的处理:
#   accumulate (默认, 旧行为): 每轮 Dify 报数累加, 反映"任务级累计输入计费"。
#                              注意: 若 Dify 已把全历史算进每轮 prompt_tokens,
#                              累加会双重计数, 上下文显示会涨得飞快 (O(n²))。
#   passthrough             : 直接透传 Dify 当轮报数, 反映"当前上下文窗口大小"。
#                              推荐在 Dify 报数已含全历史的版本上使用。
# completion_tokens 始终按 AUTO_USAGE_ACCUMULATE 一致策略处理 (输出不重复, 累加无害)。
AUTO_USAGE_MODE = os.getenv("AUTO_USAGE_MODE", "accumulate").strip().lower()
if AUTO_USAGE_MODE not in ("accumulate", "passthrough"):
    AUTO_USAGE_MODE = "accumulate"


# ── 提示词方言 ──
# generic = `[Role]:` + `<tool-calls>[JSON]</tool-calls>` (默认, 最泛化)
# claude  = `<system>/<user>/<assistant>` + `<function_calls><invoke>` (Anthropic 原生方言)
# openai  = ChatML `<|im_start|>` + `<tool_call>{JSON}</tool_call>` (Hermes/Qwen 风格)
PROMPT_DIALECT = os.getenv("PROMPT_DIALECT", "generic").strip().lower()
if PROMPT_DIALECT not in ("generic", "claude", "openai"):
    PROMPT_DIALECT = "generic"


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


# ── 连接池 ──
POOL_SIZE = int(os.getenv("POOL_SIZE", "50"))


# ── 请求/响应日志 ──
REQUEST_LOG_ENABLED = _env_bool("REQUEST_LOG_ENABLED", False)
REQUEST_LOG_DIR = os.getenv("REQUEST_LOG_DIR", "logs").strip()
REQUEST_LOG_MAX_SIZE = int(
    os.getenv("REQUEST_LOG_MAX_SIZE", str(50 * 1024 * 1024))
)  # 50MB
REQUEST_LOG_BACKUP_COUNT = int(os.getenv("REQUEST_LOG_BACKUP_COUNT", "5"))
REQUEST_LOG_MAX_BODY = int(os.getenv("REQUEST_LOG_MAX_BODY", "0"))  # 0=不截断

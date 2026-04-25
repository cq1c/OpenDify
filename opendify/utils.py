"""通用小工具。"""

import os
import secrets
from typing import Any, List

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


def truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def fast_id() -> str:
    return secrets.token_hex(16)


# tiktoken encoder 句柄缓存。第一次调用时按 TIKTOKEN_ENCODING 选编码,
# 失败则记 False 不再重试 (避免每次请求都触发网络下载尝试)。
_TT_ENCODING_NAME = os.getenv("TIKTOKEN_ENCODING", "cl100k_base").strip() or "cl100k_base"
_tt_encoder: Any = None  # None=未初始化, False=不可用, 其它=编码器实例


def _get_tt_encoder() -> Any:
    global _tt_encoder
    if _tt_encoder is not None:
        return _tt_encoder
    if tiktoken is None:
        _tt_encoder = False
        return False
    try:
        _tt_encoder = tiktoken.get_encoding(_TT_ENCODING_NAME)
    except Exception:
        try:
            _tt_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tt_encoder = False
    return _tt_encoder


def _heuristic_tokens(text: str) -> int:
    """tiktoken 不可用时的兜底: CJK 1 token/char, 其它 1 token/4char。"""
    if not text:
        return 0
    cjk = 0
    for c in text:
        cp = ord(c)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3040 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7AF
            or 0xF900 <= cp <= 0xFAFF
        ):
            cjk += 1
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_tokens(text: str) -> int:
    """
    用 tiktoken (OpenAI cl100k_base BPE) 计算 token 数, 同输入→同输出。
    用途: CONVERSATION_MODE=none 时给 prompt_tokens 提供稳定、可复现的基线,
    避免上游 Dify 模板包装造成 usage 抖动。
    若 tiktoken 不可用 (未安装 / 编码下载失败), 退到字符级启发式。
    """
    if not text:
        return 0
    enc = _get_tt_encoder()
    if enc:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return _heuristic_tokens(text)


def extract_text(content: Any) -> str:
    """从 OpenAI message.content 抽出纯文本。支持 str / list[part] / None。"""
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

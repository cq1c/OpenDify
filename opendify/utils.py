"""通用小工具。"""

import secrets
from typing import Any, List


def truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def fast_id() -> str:
    return secrets.token_hex(16)


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

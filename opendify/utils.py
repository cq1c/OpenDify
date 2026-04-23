"""通用小工具。"""

import secrets


def truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def fast_id() -> str:
    return secrets.token_hex(16)
